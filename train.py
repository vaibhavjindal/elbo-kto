#!/usr/bin/env python3
import os, argparse, random
from contextlib import nullcontext
from typing import List, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler
 

from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
)

from data_utils import DataProcessorPreprocessed

MAX_LENGTH = 4096
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
# Global-batch class composition sampler
# ===========================

def _interleave_pattern(n_desired: int, n_undesired: int) -> List[int]:
    """Alternate 1/0 while both classes remain, then append the leftovers."""
    pattern = []
    while n_desired > 0 or n_undesired > 0:
        if n_desired > 0:
            pattern.append(1)
            n_desired -= 1
        if n_undesired > 0:
            pattern.append(0)
            n_undesired -= 1
    return pattern


class ClassCompositionSampler(Sampler):
    """
    Emits a flat index order whose consecutive groups of ``group_size`` are the global
    batches actually seen by ``compute_loss``.

    accelerate's ``BatchSamplerShard(split_batches=False)`` hands per-device batch ``j`` to
    rank ``j % world_size``, so flat position ``j * group_size + r * per_device_bs`` lands on
    rank ``r`` at optimizer step ``j``. Controlling this order therefore controls the exact
    desired/undesired composition of every global batch, which is what determines the
    stop-gradient global baseline ``z0 = mean(r_hat)`` (all-reduced across ranks).

    ``group_size`` must be ``world_size * per_device_train_batch_size``. Gradient accumulation
    is deliberately excluded: ``z0`` is all-reduced once per micro-step, so the micro-step is
    the unit whose composition matters.

    Modes:
      - ``balanced``:    every global batch has ``desired_per_batch`` desired samples.
      - ``alternating``: global batches are homogeneous, alternating all-desired / all-undesired.

    Both classes are consumed fully; once one class is exhausted the remaining batches are
    filled from the other class (off-spec tail), and the trailing remainder that cannot fill a
    whole global batch is dropped.
    """

    def __init__(self, labels, group_size: int, mode: str,
                 desired_per_batch: Optional[int] = None, seed: int = 42):
        if group_size <= 0:
            raise ValueError(f"group_size must be positive, got {group_size}")
        if mode not in ("balanced", "alternating"):
            raise ValueError(f"Unknown composition mode: {mode}")

        self.group_size = int(group_size)
        self.mode = mode
        self.seed = int(seed)
        self.epoch = 0

        labels = [int(l) for l in labels]
        self.desired_idx = [i for i, l in enumerate(labels) if l == 1]
        self.undesired_idx = [i for i, l in enumerate(labels) if l != 1]

        if mode == "balanced":
            if desired_per_batch is None:
                desired_per_batch = self.group_size // 2
            if not (0 <= desired_per_batch <= self.group_size):
                raise ValueError(
                    f"desired_per_batch must be in [0, {self.group_size}], got {desired_per_batch}"
                )
            self.desired_per_batch = int(desired_per_batch)
        else:
            self.desired_per_batch = None

        n_total = len(self.desired_idx) + len(self.undesired_idx)
        self.num_groups = n_total // self.group_size
        self._num_samples = self.num_groups * self.group_size
        self.n_dropped = n_total - self._num_samples

    def set_epoch(self, epoch: int):
        """Called by accelerate's DataLoaderShard so multi-epoch runs reshuffle."""
        self.epoch = int(epoch)

    def __len__(self):
        return self._num_samples

    def _requested_pattern(self, group: int) -> List[int]:
        if self.mode == "balanced":
            return _interleave_pattern(self.desired_per_batch,
                                       self.group_size - self.desired_per_batch)
        return [1] * self.group_size if group % 2 == 0 else [0] * self.group_size

    def build_order(self) -> List[int]:
        """Deterministic given (seed, epoch) so every rank builds the identical order."""
        rng = random.Random(self.seed + 1000003 * self.epoch)
        desired = list(self.desired_idx)
        undesired = list(self.undesired_idx)
        rng.shuffle(desired)
        rng.shuffle(undesired)

        di = ui = 0
        order = []
        for g in range(self.num_groups):
            for want in self._requested_pattern(g):
                if want == 1 and di < len(desired):
                    order.append(desired[di]); di += 1
                elif want == 0 and ui < len(undesired):
                    order.append(undesired[ui]); ui += 1
                elif di < len(desired):          # requested class exhausted -> off-spec fill
                    order.append(desired[di]); di += 1
                else:
                    order.append(undesired[ui]); ui += 1
        return order

    def __iter__(self):
        return iter(self.build_order())

    def describe(self, world_size: int, per_device_bs: int, n_preview: int = 4) -> str:
        """Human-readable realized composition, for verifying an experiment before it burns GPU hours."""
        order = self.build_order()
        desired_set = set(self.desired_idx)
        lines = [
            f"[composition] mode={self.mode} group_size={self.group_size} "
            f"(world_size={world_size} x per_device_bs={per_device_bs})",
            f"[composition] pool: {len(self.desired_idx)} desired, {len(self.undesired_idx)} undesired "
            f"-> {self.num_groups} global batches, {self.n_dropped} trailing samples dropped",
        ]

        off_spec = []
        for g in range(self.num_groups):
            grp = order[g * self.group_size:(g + 1) * self.group_size]
            got = [1 if i in desired_set else 0 for i in grp]
            if got != self._requested_pattern(g):
                off_spec.append(g)

        if off_spec:
            lines.append(
                f"[composition] WARNING: {len(off_spec)} / {self.num_groups} batches are off-spec "
                f"(a class ran out); first off-spec step = {off_spec[0]}"
            )
        else:
            lines.append(f"[composition] all {self.num_groups} batches match the requested composition")

        for g in range(min(n_preview, self.num_groups)):
            grp = order[g * self.group_size:(g + 1) * self.group_size]
            got = [1 if i in desired_set else 0 for i in grp]
            per_rank = [got[r * per_device_bs:(r + 1) * per_device_bs] for r in range(world_size)]
            lines.append(
                f"[composition]   step {g}: n_desired={sum(got)}/{self.group_size} "
                f"per-rank labels={per_rank}"
            )
        return "\n".join(lines)


# ===========================
# KTO Trainer (no ref model)
# ===========================

class ELBOKTOTrainer(Trainer):
    """
    Trainer that:
      - Rebuilds the exact fixed-ℓ masks per draw k using stored seeds + ℓ,
      - Computes B_theta via mean log-prob over masked tokens,
      - Uses precomputed bref_K from dataset,
      - Computes KTO loss with global-mean z0 baseline,
      - Optionally verifies masked index sums match the precompute (debug).
    """
    # public knobs (set after construction if desired)
    verify_masks: bool = True   # compare sum(masked idx) with stored values; raise on mismatch
    n_mc_samples: int = 8
    kto_beta: float = 0.1
    kto_lambda_D: float = 1.0
    kto_lambda_U: float = 1.0
    z0_mode: str = "global_mean"  # or "zero"

    # Global-batch class composition (see ClassCompositionSampler)
    batch_composition: str = "random"       # random | balanced | alternating
    composition_seed: int = 42
    desired_per_batch: Optional[int] = None

    # Extra runtime attributes set by runner for generalization
    mask_token_id: Optional[int] = None

    # ---- data ordering ----
    def _get_train_sampler(self, train_dataset=None) -> Optional[torch.utils.data.Sampler]:
        """Override the default RandomSampler to control per-global-batch class composition.

        accelerate's prepare_data_loader only replaces samplers that are exactly a
        `RandomSampler`, so a custom Sampler subclass is passed through untouched.
        """
        if getattr(self, "batch_composition", "random") == "random":
            return super()._get_train_sampler(train_dataset)

        if train_dataset is None:
            train_dataset = self.train_dataset

        try:
            labels = train_dataset["labels"]
        except (TypeError, KeyError):
            labels = [train_dataset[i]["labels"] for i in range(len(train_dataset))]

        group_size = self.args.world_size * self._train_batch_size
        return ClassCompositionSampler(
            labels=labels,
            group_size=group_size,
            mode=self.batch_composition,
            desired_per_batch=self.desired_per_batch,
            seed=self.composition_seed,
        )

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
        z0_mode = getattr(self, "z0_mode", "global_mean")
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

        # z0 baseline
        if z0_mode == "zero":
            z0 = torch.zeros((), device=x.device, dtype=r_hat.dtype)
        elif z0_mode == "global_mean":
            z0 = self._global_mean_1d(r_hat)
        else:
            raise ValueError(f"Unknown z0_mode: {z0_mode}")

        s = beta * (r_hat - z0)                     # [B]

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
            frac_desired = self._global_mean_1d((y == 1).float())

        if is_main_process():
            self.log({
                "abs_beta_r/mean":   mean_abs_beta_r.item(),
                "abs_beta_s/mean":   mean_abs_beta_s.item(),
                "abs_r_hat_z0/mean": mean_abs_r_hat_z0.item(),
                "abs_beta_s/frac>6": sat6.item(),
                "abs_beta_s/frac>8": sat8.item(),
                "resp_len/mean":     mean_len_resp.item(),
                "batch/frac_desired": frac_desired.item(),
            })

        # KTO loss
        good = (y == 1)
        v = torch.empty_like(s)
        v[good]  = lambda_D * torch.sigmoid(s[good])
        v[~good] = lambda_U * torch.sigmoid(-s[~good])
        lambdas = lambda_D * good.float() + lambda_U * (~good).float()
        loss = (lambdas - v).mean()

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
    p.add_argument("--z0_mode", type=str, default="global_mean", choices=["global_mean", "zero"])
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

    # Global-batch class composition
    p.add_argument("--batch_composition", type=str, default="random",
                   choices=["random", "balanced", "alternating"],
                   help="Desired/undesired composition of each global batch (the unit over which "
                        "z0 is all-reduced). 'random' = stock HF RandomSampler. "
                        "'balanced' = every global batch has --desired_per_batch desired samples. "
                        "'alternating' = homogeneous batches alternating all-desired / all-undesired.")
    p.add_argument("--desired_per_batch", type=int, default=None,
                   help="Number of desired samples per global batch when --batch_composition=balanced. "
                        "Defaults to half the global batch (e.g. 4 of 8 on 8 GPUs with microbatch 1).")
    p.add_argument("--composition_seed", type=int, default=42,
                   help="Seed for the within-class shuffle used by --batch_composition")
    p.add_argument("--print_batch_composition", type=int, default=4,
                   help="Print the realized composition of the first N global batches at startup "
                        "(0 to disable). Verification only; does not affect training.")

    return p.parse_args()

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
        # A partial trailing global batch would make z0 a mean over fewer samples and would
        # trip accelerate's even_batches tail-cycling, which recycles early samples and breaks
        # the requested composition.
        dataloader_drop_last=(args.batch_composition != "random"),
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
    trainer.z0_mode = args.z0_mode
    trainer.verify_masks = args.verify_masks and not args.disable_mask_verification

    # Global-batch class composition knobs (must be set before get_train_dataloader runs)
    trainer.batch_composition = args.batch_composition
    trainer.composition_seed = args.composition_seed
    trainer.desired_per_batch = args.desired_per_batch

    # Generalization runtime ids
    trainer.mask_token_id = _resolve_mask_token_id(tok, model)

    if args.batch_composition != "random" and args.print_batch_composition > 0 and is_main_process():
        sampler = trainer._get_train_sampler(train_ds)
        print(sampler.describe(
            world_size=train_args.world_size,
            per_device_bs=trainer._train_batch_size,
            n_preview=args.print_batch_composition,
        ))

    if is_main_process():
        print("Starting training...")
        print(f"mask_token_id={trainer.mask_token_id}")
    trainer.train()
    if is_main_process():
        print("Training completed!")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
