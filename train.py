#!/usr/bin/env python3
import os, argparse
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
 

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)

from data_utils import DataProcessorPreprocessed

MAX_LENGTH = 4096

# =========================
# Baseline (z0) variants
# =========================

BASELINE_TYPES = ("batch_mean", "none", "running_global", "running_class_conditional")

# Legacy --z0_mode values -> new --baseline_type values (behavior is identical).
_LEGACY_BASELINE_ALIASES = {"global_mean": "batch_mean", "zero": "none"}


def normalize_baseline_type(value: str) -> str:
    """Map legacy z0_mode names onto baseline_type names and validate."""
    if value is None:
        raise ValueError("baseline type cannot be None")
    v = _LEGACY_BASELINE_ALIASES.get(value, value)
    if v not in BASELINE_TYPES:
        raise ValueError(
            f"Unknown baseline type {value!r}. Expected one of {list(BASELINE_TYPES)} "
            f"(or legacy {list(_LEGACY_BASELINE_ALIASES)})."
        )
    return v


# =========================
# Custom Data Collator
# =========================

def kto_data_collator(features):
    """
    Custom data collator for KTO training that handles the specific fields expected.
    """
    import torch
    from transformers.data.data_collator import torch_default_data_collator
    
    # Handle each field explicitly
    batch = {}
    
    # Simple tensor fields that can be stacked directly
    simple_fields = ['input_ids', 'prompt_length', 'completion_length', 'labels']
    
    for field in simple_fields:
        if field in features[0]:
            values = [f[field] for f in features]
            # Convert to tensors if they aren't already
            if not isinstance(values[0], torch.Tensor):
                values = [torch.tensor(v) for v in values]
            batch[field] = torch.stack(values)
    
    # Handle seed field separately (stored as strings to avoid PyArrow overflow)
    if 'seed' in features[0]:
        seed_strings = [f['seed'] for f in features]
        batch['seed'] = seed_strings  # Keep as list of strings for now
    
    # Handle B_ref dictionary field - create separate tensors for each key
    if 'B_ref' in features[0]:
        # B_ref is a dict of the form {"1": <float>, "2": <float>, "4": <float>, "8": <float>}
        # Get all unique keys from the first example
        b_ref_keys = list(features[0]['B_ref'].keys())
        
        # Create a tensor for each key
        for key in b_ref_keys:
            bref_values = []
            for f in features:
                b_ref_dict = f['B_ref']
                if isinstance(b_ref_dict, dict) and key in b_ref_dict:
                    bref_values.append(float(b_ref_dict[key]))
                else:
                    # Handle missing keys gracefully
                    bref_values.append(0.0)
            batch[f'B_ref_{key}'] = torch.tensor(bref_values, dtype=torch.float32)
    
    # Handle 2D tensor fields (l_values, masked_idx_sums)
    tensor_2d_fields = ['l_values', 'masked_idx_sums']
    
    for field in tensor_2d_fields:
        if field in features[0]:
            values = [f[field] for f in features]
            # Convert to tensors if they aren't already
            if not isinstance(values[0], torch.Tensor):
                values = [torch.tensor(v) for v in values]
            batch[field] = torch.stack(values)
    
    return batch

# =========================
# Constants / Small Helpers
# =========================

def init_distributed():
    """Initialize torch.distributed when launched with torchrun."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"[Rank {dist.get_rank()}] on GPU {local_rank}")
    else:
        local_rank = 0
    return local_rank

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

# ======= Deterministic PRNG (must match precompute) =======

def mix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF

def derive_seed(base_seed: int, k: int, tag: int) -> int:
    x = base_seed ^ (k * 0x9E3779B97F4A7C15) ^ (tag * 0xD1342543DE82EF95)
    return mix64(x)

# ==========================================================
# Mask building (fixed-ℓ in response window; must match precompute)
# ==========================================================

def make_fixedl_masks_batched(
    select_seeds,                   # [B] list of ints or tensor
    prompt_lens: torch.Tensor,      # [B] int64
    comp_lens: torch.Tensor,        # [B] int64
    l_vals: torch.Tensor,           # [B] int64, ℓ for this draw
    L: int,
    device: torch.device,
) -> torch.BoolTensor:
    """
    Build masks [B, L] with exactly ℓ masked tokens in response window
    [prompt_len, prompt_len + completion_length). Sampling WITHOUT replacement,
    deterministically via per-example torch.Generator seed. This mirrors precompute.
    """
    # Handle both list and tensor inputs for select_seeds
    if isinstance(select_seeds, torch.Tensor):
        B = select_seeds.numel()
    else:
        B = len(select_seeds)
    
    masks = torch.zeros((B, L), dtype=torch.bool, device=device)

    for b in range(B):
        p = int(prompt_lens[b].item())
        c = int(comp_lens[b].item())
        ell = int(l_vals[b].item())
        if c <= 0 or ell <= 0:
            continue
        ell = min(ell, c)
        start, end = p, p + c  # response window [start, end)

        g = torch.Generator(device=device)
        # Handle both list and tensor inputs for seeds
        if isinstance(select_seeds, torch.Tensor):
            seed_val = int(select_seeds[b].item()) & 0xFFFFFFFFFFFFFFFF
        else:
            seed_val = int(select_seeds[b]) & 0xFFFFFFFFFFFFFFFF
        g.manual_seed(seed_val)
        # deterministic permutation of the response window
        r = torch.rand((c,), generator=g, device=device)   # uniforms on window length
        perm = torch.argsort(r)[:ell]                      # first ℓ positions by sort
        idx = start + perm
        masks[b, idx] = True
    return masks

# ===========================
# KTO Trainer (no ref model)
# ===========================

class ELBOKTOTrainer(Trainer):
    """
    Trainer that:
      - Rebuilds the exact fixed-ℓ masks per draw k using stored seeds + ℓ,
      - Computes B_theta via mean log-prob over masked tokens,
      - Uses precomputed bref_K from dataset,
      - Computes KTO loss with a configurable stop-gradient z0 baseline,
      - Optionally verifies masked index sums match the precompute (debug).

    Baseline variants (``baseline_type``):
      - "batch_mean"                : z0 = mean of the current global batch margins (default).
      - "none"                      : z0 = 0.
      - "running_global"            : z0 = EMA of past global batch margin means.
      - "running_class_conditional" : separate EMAs for desirable / undesirable examples.

    The running variants use the EMA value from *before* the current batch, and are
    updated only after the batch loss has been computed.
    """
    # public knobs (set after construction if desired)
    verify_masks: bool = True   # compare sum(masked idx) with stored values; raise on mismatch
    n_mc_samples: int = 8
    kto_beta: float = 0.1
    kto_lambda_D: float = 1.0
    kto_lambda_U: float = 1.0
    baseline_type: str = "batch_mean"
    baseline_ema_decay: float = 0.99
    z0_mode: Optional[str] = None  # deprecated alias for baseline_type

    # Running baseline state (stop-grad scalars, identical on every rank because they are
    # only ever updated from all-reduced quantities). Initialized to 0.
    _baseline_ema_global: float = 0.0
    _baseline_ema_D: float = 0.0
    _baseline_ema_U: float = 0.0

    # Extra runtime attributes set by runner for generalization
    mask_token_id: Optional[int] = None

    # ---- utilities ----
    def _global_mean_1d(self, x: torch.Tensor):
        """Mean of a 1D tensor across ranks (detached)."""
        x = x.detach()
        if not dist.is_initialized():
            return x.mean()
        s = x.sum()
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        n_local = torch.tensor([x.numel()], device=x.device, dtype=torch.long)
        dist.all_reduce(n_local, op=dist.ReduceOp.SUM)
        return s / n_local.to(s.dtype)

    # ---- baseline helpers ----
    def _resolve_baseline_type(self) -> str:
        """Resolve baseline_type, honoring the deprecated z0_mode alias."""
        legacy = getattr(self, "z0_mode", None)
        if legacy is not None:
            return normalize_baseline_type(legacy)
        return normalize_baseline_type(getattr(self, "baseline_type", "batch_mean"))

    def _global_class_stats(self, r_hat: torch.Tensor, good: torch.Tensor) -> torch.Tensor:
        """
        Reduce [sum_D, n_D, sum_U, n_U] over all ranks in a single collective.

        MUST be called by every rank, unconditionally and in the same order, or NCCL
        deadlocks. Any "is this class present?" decision must therefore be made from the
        *reduced* result, never from rank-local data.
        """
        r = r_hat.detach().float()
        d = good.to(r.dtype)
        u = 1.0 - d
        stats = torch.stack([(r * d).sum(), d.sum(), (r * u).sum(), u.sum()])
        if dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats

    def _baseline_z0(self, r_hat: torch.Tensor, good: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Stop-gradient baseline z0 used to center the margins for this batch.

        Scalar for every mode except "running_class_conditional", which returns [B].
        Running modes deliberately return the EMA value from *before* this batch.
        """
        if mode == "none":
            return torch.zeros((), device=r_hat.device, dtype=r_hat.dtype)
        if mode == "batch_mean":
            return self._global_mean_1d(r_hat)
        if mode == "running_global":
            return torch.full((), float(self._baseline_ema_global),
                              device=r_hat.device, dtype=r_hat.dtype)
        if mode == "running_class_conditional":
            bD = torch.full((), float(self._baseline_ema_D), device=r_hat.device, dtype=r_hat.dtype)
            bU = torch.full((), float(self._baseline_ema_U), device=r_hat.device, dtype=r_hat.dtype)
            return torch.where(good, bD, bU)
        raise ValueError(f"Unknown baseline_type: {mode}")

    def _update_running_baselines(self, stats, mode: str) -> None:
        """
        EMA update b_t = decay * b_{t-1} + (1 - decay) * mean_margin_t.

        `stats` is (sum_D, n_D, sum_U, n_U), already reduced across ranks, so every rank
        takes the same branch and the EMAs stay in lockstep. A class absent from the
        *global* batch is skipped.
        """
        if mode not in ("running_global", "running_class_conditional"):
            return
        decay = float(self.baseline_ema_decay)
        sum_D, n_D, sum_U, n_U = stats

        if mode == "running_global":
            n = n_D + n_U
            if n > 0:
                mean_r = (sum_D + sum_U) / n
                self._baseline_ema_global = decay * self._baseline_ema_global + (1.0 - decay) * mean_r
        else:
            if n_D > 0:
                self._baseline_ema_D = decay * self._baseline_ema_D + (1.0 - decay) * (sum_D / n_D)
            if n_U > 0:
                self._baseline_ema_U = decay * self._baseline_ema_U + (1.0 - decay) * (sum_U / n_U)

    # ---- core math ----
    def _logp_mean_over_mask(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        """
        Mean log-prob over masked positions, per example.
        logits: [B, L, V] (float / bf16 ok, we cast to float for CE)
        targets: [B, L] long
        mask: [B, L] bool
        Returns: [B] float32 (mean log p over masked positions; 0 if none masked)
        """
        B, L, V = logits.shape
        vals = torch.zeros((B,), device=logits.device, dtype=torch.float32)

        for b in range(B):
            m = mask[b]
            if not m.any():
                continue
            idx = torch.nonzero(m, as_tuple=False).squeeze(1)
            # CE = -log p; we want mean log p
            ce = F.cross_entropy(logits[b].index_select(0, idx).float(),
                                 targets[b].index_select(0, idx),
                                 reduction="none")
            vals[b] = (-ce).mean()
        return vals

    # ---- main loss ----
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Expected batch keys:
          input_ids          [B,L] long
          prompt_length      [B]   long
          completion_length  [B]   long
          labels             [B]   long (1 = desired, 0 = undesired)
          seed               [B]   long (base seed per example)
          l_values           [B,Kmax] long  (Kmax >= self.n_mc_samples)
          masked_idx_sums    [B,Kmax] long  (sum of indices per draw; Kmax >= n_mc)
          bref_K             [B]   float  (precomputed B_ref prefix mean for n_mc)
        """
        # hyperparams
        K = getattr(self, "n_mc_samples", 8)
        beta = getattr(self, "kto_beta", 0.1)
        lambda_D = getattr(self, "kto_lambda_D", 1.0)
        lambda_U = getattr(self, "kto_lambda_U", 1.0)
        baseline_type = self._resolve_baseline_type()
        verify = getattr(self, "verify_masks", True)

        # runtime ids
        mask_token_id = getattr(self, "mask_token_id", None)
        if mask_token_id is None:
            raise ValueError("mask_token_id must be set on the trainer for masking.")

        # tensors
        x = inputs["input_ids"].to(model.device).long()          # [B,L]
        pl = inputs["prompt_length"].to(model.device).long()     # [B]
        cl = inputs["completion_length"].to(model.device).long() # [B]
        y  = inputs["labels"].to(model.device).long()            # [B]
        # Convert string seeds back to integers (stored as strings to avoid PyArrow overflow)
        seed_strings = inputs["seed"]  # [B] list of strings 
        # print("seed_strings: ", seed_strings)
        base_seed_ints = [int(s) for s in seed_strings]  # Keep as Python ints to avoid tensor overflow
        # print("base_seed_ints: ", base_seed_ints)
        l_values = inputs["l_values"].to(model.device).long()    # [B,Kmax]
        # print("l_values: ", l_values)
        idx_sums = inputs["masked_idx_sums"].to(model.device).long()  # [B,Kmax]
        # Use B_ref value corresponding to n_mc_samples (K)
        bkey = f"B_ref_{K}"
        if bkey not in inputs:
            available_keys = [k for k in inputs.keys() if k.startswith("B_ref_")]
            raise KeyError(f"Missing precomputed {bkey} in batch. Available: {available_keys}. Ensure precompute_bref.py included this K and that n_mc_samples matches.")
        bref_K   = inputs[bkey].to(model.device).float()     # [B]

        Bsz, L = x.shape
        assert l_values.shape[1] >= K and idx_sums.shape[1] >= K, \
            "Dataset l_values/masked_idx_sums must have at least n_mc_samples columns"

        # cache targets once
        targets = x

        # accumulate B_theta draws across k (we'll average at end) without in-place ops
        btheta_terms = []
        total_masked_tokens = 0

        # loop over draws k=1..K (keeps memory low)
        for k in range(1, K + 1):
            # derive selection seeds (tag=1; must match precompute)
            select_seeds = []
            for b in range(Bsz):
                # Use Python int directly to avoid tensor overflow issues
                base_seed_int = base_seed_ints[b]
                derived_seed = derive_seed(base_seed_int, k, tag=1)
                select_seeds.append(derived_seed)
            
            # Pass seeds as list to avoid overflow when converting to tensor
            
            # Alternative: If you want to use tensors, apply mask first to prevent overflow:
            # masked_seeds = [s & 0x7FFFFFFFFFFFFFFF for s in select_seeds]  # Force to signed int64 range
            # select_seeds = torch.tensor(masked_seeds).to(x.device)

            # take the stored ℓ for this draw
            l_k = l_values[:, k - 1]  # [B]

            # build masks (exactly ℓ masked tokens in response window)
            masks = make_fixedl_masks_batched(
                select_seeds=select_seeds,
                prompt_lens=pl,
                comp_lens=cl,
                l_vals=l_k,
                L=L,
                device=x.device,
            )  # [B,L] bool

            # ===== verification: sum of masked indices per example =====
            if verify:
                # compute sum of absolute indices (must match the precompute bookkeeping)
                sums = torch.zeros((Bsz,), dtype=torch.long, device=x.device)
                nz = masks.nonzero(as_tuple=False)
                if nz.numel() > 0:
                    rows, cols = nz[:, 0], nz[:, 1]
                    sums.index_add_(0, rows, cols)
                # compare to stored masked_idx_sums[:, k-1]
                mism = (sums != idx_sums[:, k - 1])
                if mism.any():
                    # Print first few mismatches to help debug. Raise to catch determinism issues.
                    bad_rows = torch.nonzero(mism, as_tuple=False).squeeze(1)
                    msg = f"[Rank {dist.get_rank() if dist.is_initialized() else 0}] " \
                          f"Mask verification failed at draw k={k} for rows: {bad_rows.tolist()[:5]}"
                    print(msg)
                    raise RuntimeError("Deterministic mask check failed. Ensure loader fields match precompute and code paths identical.")

            # Track if this batch contributes any gradient (any masked token present across draws)
            total_masked_tokens += int(masks.sum().item())

            # forward pass for this draw
            noisy = x.clone()
            noisy[masks] = mask_token_id
            out = model(input_ids=noisy)
            logits = out.logits  # [B,L,V]; dtype could be bf16; CE will cast to float

            # mean log-prob over masked positions (fixed-ℓ estimator)
            b_k = self._logp_mean_over_mask(logits, targets, masks)  # [B]
            # Multiply each value by corresponding comp_len value
            b_k = cl.float() * b_k
            btheta_terms.append(b_k)

        # average across K draws
        if total_masked_tokens == 0:
            # No masked tokens across all draws in this batch; return a zero loss connected to the graph
            zero_loss = next(model.parameters()).sum() * 0.0
            if return_outputs:
                z0 = torch.zeros((), device=x.device, dtype=torch.float32)
                return zero_loss, {"r_mean_local": torch.zeros((), device=x.device), "z0": z0}
        
        B_theta = torch.stack(btheta_terms, dim=0).mean(dim=0)           # [B]
        r_hat = B_theta - bref_K                    # [B]

        good = (y == 1)

        # z0 baseline (stop-grad). Running variants use the value from *before* this batch.
        z0 = self._baseline_z0(r_hat, good, baseline_type)

        s = beta * (r_hat - z0)                     # [B]

        # KTO loss
        v = torch.empty_like(s)
        v[good]  = lambda_D * torch.sigmoid(s[good])
        v[~good] = lambda_U * torch.sigmoid(-s[~good])
        lambdas = lambda_D * good.float() + lambda_U * (~good).float()
        loss = (lambdas - v).mean()

        # ---- running baseline EMA update: strictly after the batch loss is computed ----
        # The collective runs on every rank in every mode so that all ranks issue the same
        # sequence of collectives; only the *use* of the result is mode-dependent.
        class_stats = self._global_class_stats(r_hat, good)
        sum_D, n_D, sum_U, n_U = (float(t) for t in class_stats.tolist())
        if getattr(model, "training", True):
            # Do not let evaluation batches pollute the running baselines.
            self._update_running_baselines((sum_D, n_D, sum_U, n_U), baseline_type)

        with torch.no_grad():
            abs_beta_r = (beta * r_hat).abs().float()          # [B]
            abs_beta_s = s.abs().float()
            abs_r_hat_z0 = (r_hat - z0).abs().float()

            # global means via all_reduce
            mean_abs_beta_r = self._global_mean_1d(abs_beta_r)
            mean_abs_beta_s = self._global_mean_1d(abs_beta_s)
            mean_abs_r_hat_z0 = self._global_mean_1d(abs_r_hat_z0)
            sat6 = self._global_mean_1d((abs_beta_s > 6).float())
            sat8 = self._global_mean_1d((abs_beta_s > 8).float())
            mean_len_resp = self._global_mean_1d(cl.float())
            # expand so a class-conditional (per-example) z0 is example-weighted globally
            z0_used_mean = self._global_mean_1d(z0.detach().float().expand_as(r_hat))

        if is_main_process():
            logs = {
                "abs_beta_r/mean":   mean_abs_beta_r.item(),
                "abs_beta_s/mean":   mean_abs_beta_s.item(),
                "abs_r_hat_z0/mean": mean_abs_r_hat_z0.item(),
                "abs_beta_s/frac>6": sat6.item(),
                "abs_beta_s/frac>8": sat8.item(),
                "resp_len/mean":     mean_len_resp.item(),
                # baseline telemetry
                "baseline/z0_used_mean": z0_used_mean.item(),
                "baseline/n_D": n_D,
                "baseline/n_U": n_U,
                "baseline/margin_mean_D": (sum_D / n_D) if n_D > 0 else float("nan"),
                "baseline/margin_mean_U": (sum_U / n_U) if n_U > 0 else float("nan"),
            }
            if baseline_type == "running_global":
                logs["baseline/ema_global"] = float(self._baseline_ema_global)
            elif baseline_type == "running_class_conditional":
                logs["baseline/ema_D"] = float(self._baseline_ema_D)
                logs["baseline/ema_U"] = float(self._baseline_ema_U)
            self.log(logs)

        if return_outputs:
            return loss, {"r_mean_local": r_hat.detach().mean(), "z0": z0.detach()}
        return loss

# ======================
# CLI / Runner
# ======================

def parse_args():
    p = argparse.ArgumentParser(description="Train diffusion LLM with KTO using precomputed B_ref (fixed-ℓ).")
    p.add_argument("--model_name_or_path", type=str, required=True,
                   help="HF model path or name (policy init). Use the same tokenizer/template as precompute.")
    p.add_argument("--output_dir", type=str, default="kto-train",
                   help="HF Trainer output directory")
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.95)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1, 
                   help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=3000)
    p.add_argument("--save_strategy", type=str, default="steps", choices=["no", "steps", "epoch"],
                   help="Save strategy: 'no' (no saving), 'steps' (save every save_steps), 'epoch' (save every epoch)")
    p.add_argument("--eval_steps", type=int, default=0)

    # FSDP flags (optional)
    p.add_argument("--fsdp", type=str, default="full_shard auto_wrap")
    p.add_argument("--activation_checkpointing", action="store_true", default=True)

    # KTO
    p.add_argument("--n_mc_samples", type=int, default=8)
    p.add_argument("--kto_beta", type=float, default=0.2)
    p.add_argument("--kto_lambda_D", type=float, default=1.0)
    p.add_argument("--kto_lambda_U", type=float, default=1.0)
    p.add_argument("--z0_mode", type=str, default=None, choices=["global_mean", "zero"],
                   help="DEPRECATED alias for --baseline_type ('global_mean'->'batch_mean', 'zero'->'none').")
    p.add_argument("--baseline_type", type=str, default=None, choices=list(BASELINE_TYPES),
                   help="Stop-gradient z0 baseline used to center ELBO margins. "
                        "'batch_mean' (default): mean margin of the current global batch. "
                        "'none': 0. "
                        "'running_global': EMA over past global batch margin means. "
                        "'running_class_conditional': separate EMAs for desirable/undesirable.")
    p.add_argument("--baseline_ema_decay", type=float, default=0.99,
                   help="EMA decay for the running baselines: b_t = decay*b_{t-1} + (1-decay)*mean_margin_t")
    p.add_argument("--verify_masks", action="store_true", default=True)
    p.add_argument("--disable_mask_verification", action="store_true", default=False,
                   help="Disable mask verification for faster training (production mode)")

    # Dataset module path/hints:
    p.add_argument("--train_dataset_path", type=str, required=True,
                   help="Path/name your loader will use to return the preprocessed train dataset")
    p.add_argument("--eval_dataset_path", type=str, default=None,
                   help="Optional eval dataset path")
    
    # Sample filtering
    p.add_argument("--n_D", type=float, default=1.0,
                   help="Ratio of positive samples (labels=1) to use for training. Range: [0.0, 1.0]. E.g., 0.1 uses 10%% of positive samples, 0.0 uses none")
    p.add_argument("--n_U", type=float, default=1.0,
                   help="Ratio of negative samples (labels=0) to use for training. Range: [0.0, 1.0]. E.g., 0.1 uses 10%% of negative samples, 0.0 uses none")
    p.add_argument("--sample_seed", type=int, default=42,
                   help="Random seed for reproducible dataset sampling when n_D or n_U < 1.0")

    return p.parse_args()

def resolve_baseline_args(args):
    """
    Resolve --baseline_type / deprecated --z0_mode into args.baseline_type, and validate
    the EMA decay. Returns the resolved baseline type.
    """
    if args.baseline_type is not None and args.z0_mode is not None:
        resolved = normalize_baseline_type(args.baseline_type)
        if normalize_baseline_type(args.z0_mode) != resolved:
            raise ValueError(
                f"--baseline_type={args.baseline_type} conflicts with deprecated "
                f"--z0_mode={args.z0_mode}. Pass only --baseline_type."
            )
        args.baseline_type = resolved
    elif args.z0_mode is not None:
        args.baseline_type = normalize_baseline_type(args.z0_mode)
        print(f"[WARN] --z0_mode is deprecated; using --baseline_type={args.baseline_type}")
    else:
        args.baseline_type = normalize_baseline_type(args.baseline_type or "batch_mean")

    if not (0.0 <= args.baseline_ema_decay < 1.0):
        raise ValueError(
            f"baseline_ema_decay must be in [0.0, 1.0), got {args.baseline_ema_decay}"
        )
    args.z0_mode = None  # consumed; the trainer reads baseline_type only
    return args.baseline_type


def create_training_args(args) -> TrainingArguments:
    fsdp_cfg = {
        "min_num_params": 100_000_000,
        "mixed_precision": "bf16",
        "activation_checkpointing": args.activation_checkpointing,
        "state_dict_type": "sharded_state_dict",
    }
    
    # Handle save_steps based on save_strategy
    save_steps = args.save_steps if args.save_strategy != "no" else None
    
    return TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=save_steps,
        save_strategy=args.save_strategy,
        remove_unused_columns=False,  # we need custom fields intact
        eval_strategy=("steps" if args.eval_steps > 0 else "no"),
        eval_steps=(args.eval_steps if args.eval_steps > 0 else None),
        lr_scheduler_type="cosine",
        fsdp=args.fsdp,
        fsdp_config=fsdp_cfg,
        report_to=[],  # Disable all logging integrations (MLflow, wandb, etc.)
    )

# ==== Generalization helpers ====
    
def _resolve_mask_token_id(tokenizer: AutoTokenizer, model=None) -> int:
    # Prefer tokenizer's built-in
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return int(tokenizer.mask_token_id)
    # Try model config
    if model is not None and getattr(getattr(model, "config", None), "mask_token_id", None) is not None:
        return int(model.config.mask_token_id)
    # Try special_tokens_map
    mask_tok = getattr(tokenizer, "mask_token", None) or tokenizer.special_tokens_map.get("mask_token")
    if mask_tok:
        tok_id = tokenizer.convert_tokens_to_ids(mask_tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            return int(tok_id)
    raise ValueError("Could not resolve mask_token_id. Provide --mask_id or --mask_token.")


def main():
    args = parse_args()
    
    # Validate sampling ratio arguments
    if not (0.0 <= args.n_D <= 1.0):
        raise ValueError(f"n_D must be between 0.0 and 1.0 (inclusive), got {args.n_D}")
    if not (0.0 <= args.n_U <= 1.0):
        raise ValueError(f"n_U must be between 0.0 and 1.0 (inclusive), got {args.n_U}")
    if args.n_D == 0.0 and args.n_U == 0.0:
        raise ValueError("Both n_D and n_U cannot be 0.0 - at least one sample type must be used")

    # Resolve baseline type (--baseline_type wins; --z0_mode kept as a deprecated alias)
    resolve_baseline_args(args)
    
    init_distributed()

    # Load tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if "llada" in args.model_name_or_path.lower():
        model_kwargs["flash_attention"] = True
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        **model_kwargs
    )
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Load preprocessed datasets (you implement these)
    train_ds = DataProcessorPreprocessed(tok).load_dataset(args.train_dataset_path)["train"]
    # Drop samples with no masked tokens
    train_ds = train_ds.filter(lambda ex: sum(ex["l_values"]) > 0)
    
    # Filter dataset based on n_D and n_U ratios
    if args.n_D < 1.0 or args.n_U < 1.0:
        import random
        random.seed(args.sample_seed)
        
        # Split into positive (desired) and negative (undesired) samples
        positive_indices = [i for i, ex in enumerate(train_ds) if ex["labels"] == 1]
        negative_indices = [i for i, ex in enumerate(train_ds) if ex["labels"] == 0]
        
        original_pos_count = len(positive_indices)
        original_neg_count = len(negative_indices)
        
        # Sample according to ratios
        n_pos_samples = int(original_pos_count * args.n_D)
        n_neg_samples = int(original_neg_count * args.n_U)
        
        # Ensure we have at least 1 sample of each type if the original dataset had them,
        # but only if the ratio is > 0.0 (respect explicit 0.0 ratios)
        if original_pos_count > 0 and n_pos_samples == 0 and args.n_D > 0.0:
            n_pos_samples = 1
        if original_neg_count > 0 and n_neg_samples == 0 and args.n_U > 0.0:
            n_neg_samples = 1
            
                    # Randomly sample indices
        selected_pos_indices = random.sample(positive_indices, min(n_pos_samples, original_pos_count)) if n_pos_samples > 0 else []
        selected_neg_indices = random.sample(negative_indices, min(n_neg_samples, original_neg_count)) if n_neg_samples > 0 else []
        
        # Combine selected indices and sort to maintain some order
        selected_indices = sorted(selected_pos_indices + selected_neg_indices)
        
        # Filter the dataset
        train_ds = train_ds.select(selected_indices)
        
        if is_main_process():
            print(f"Dataset filtering applied:")
            print(f"  Original: {original_pos_count} positive, {original_neg_count} negative")
            print(f"  Filtered: {len(selected_pos_indices)} positive (n_D={args.n_D}), {len(selected_neg_indices)} negative (n_U={args.n_U})")
            print(f"  Total samples: {original_pos_count + original_neg_count} -> {len(train_ds)}")
    else:
        if is_main_process():
            pos_count = len([ex for ex in train_ds if ex["labels"] == 1])
            neg_count = len([ex for ex in train_ds if ex["labels"] == 0])
            print(f"Using full dataset: {pos_count} positive, {neg_count} negative samples")
    # print(len(train_ds))
    # for i in range(1):
    #     print(train_ds[i].keys())
    # # exit()
    eval_ds  = DataProcessorPreprocessed(tok).load_dataset(args.eval_dataset_path) if args.eval_dataset_path else None
    if eval_ds is not None and isinstance(eval_ds, dict) and "train" in eval_ds:
        eval_ds = eval_ds["train"].filter(lambda ex: sum(ex["l_values"]) > 0)

    train_args = create_training_args(args)

    trainer = ELBOKTOTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=kto_data_collator,
        processing_class=tok,
    )

    # KTO knobs
    trainer.n_mc_samples = args.n_mc_samples
    trainer.kto_beta = args.kto_beta
    trainer.kto_lambda_D = args.kto_lambda_D
    trainer.kto_lambda_U = args.kto_lambda_U
    trainer.baseline_type = args.baseline_type
    trainer.baseline_ema_decay = args.baseline_ema_decay
    trainer.z0_mode = None  # resolved into baseline_type above    trainer.verify_masks = args.verify_masks and not args.disable_mask_verification

    # Generalization runtime ids
    trainer.mask_token_id = _resolve_mask_token_id(tok, model)

    if is_main_process():
        print("Starting training...")
        print(f"mask_token_id={trainer.mask_token_id}")
        print(f"baseline_type={trainer.baseline_type} (ema_decay={trainer.baseline_ema_decay})")
    trainer.train()
    if is_main_process():
        print("Training completed!")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
