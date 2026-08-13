#!/usr/bin/env python3
"""
Tests for the z0 baseline variants in train.py.

Runs the *real* ELBOKTOTrainer.compute_loss against a tiny stub mask-predictor, so the
mask rebuild, the verification checksum, the collectives and the EMA bookkeeping are all
exercised without needing LLaDA-8B.

  single process : python tests/test_baselines.py
  8 ranks        : torchrun --nproc_per_node=8 tests/test_baselines.py
"""
import os
import sys
import math

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import TrainingArguments  # noqa: E402

import train as T  # noqa: E402
from precompute_bref import (  # noqa: E402
    mix64,
    derive_seed,
    l_from_seed,
    make_batched_fixedl_masks,
)

VOCAB = 32
SEQ_LEN = 24
MASK_ID = 31
K = 2
DECAY = 0.99
T_GLOBAL_SEED = 0x1A2B3C4D5E6F7788


class StubOut:
    def __init__(self, logits):
        self.logits = logits


class StubModel(nn.Module):
    """Deterministic tiny mask predictor with the (input_ids) -> .logits contract."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, VOCAB)
        with torch.no_grad():
            self.emb.weight.copy_(torch.eye(VOCAB) * 3.0)

    def forward(self, input_ids=None, **kw):
        return StubOut(self.emb(input_ids))

    @property
    def device(self):
        return self.emb.weight.device


def build_example(idx, label, device):
    """Mirror precompute_bref bookkeeping so verify_masks has real values to check."""
    base_seed = mix64(T_GLOBAL_SEED ^ idx)
    prompt_len, comp_len = 4, 10
    g = torch.Generator().manual_seed(idx)
    input_ids = torch.randint(0, VOCAB - 1, (SEQ_LEN,), generator=g)

    l_values, idx_sums = [], []
    for k in range(1, K + 1):
        l = l_from_seed(derive_seed(base_seed, k, tag=0), comp_len)
        sel = derive_seed(base_seed, k, tag=1)
        m = make_batched_fixedl_masks([sel], SEQ_LEN, [prompt_len], [comp_len], [l], device)
        l_values.append(l)
        idx_sums.append(int(m[0].nonzero().squeeze(1).sum().item()))
    return {
        "input_ids": input_ids,
        "prompt_length": prompt_len,
        "completion_length": comp_len,
        "labels": label,
        "seed": str(base_seed),
        "l_values": l_values,
        "masked_idx_sums": idx_sums,
        "B_ref": {str(K): 0.5 * idx},
    }


def make_batch(labels, device, offset=0):
    feats = [build_example(offset + i, lab, device) for i, lab in enumerate(labels)]
    batch = T.kto_data_collator(feats)
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_trainer(baseline_type, device, tmpdir):
    model = StubModel().to(device)
    args = TrainingArguments(
        output_dir=tmpdir,
        per_device_train_batch_size=1,
        report_to=[],
        logging_strategy="no",
        save_strategy="no",
        use_cpu=(device.type == "cpu"),
    )
    tr = T.ELBOKTOTrainer(model=model, args=args, data_collator=T.kto_data_collator)
    tr.baseline_type = baseline_type
    tr.baseline_ema_decay = DECAY
    tr.z0_mode = None
    tr.n_mc_samples = K
    tr.kto_beta = 0.1
    tr.verify_masks = True
    tr.mask_token_id = MASK_ID
    return tr, model


def approx(a, b, tol=1e-5):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def run(device, tmpdir):
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    failures = []

    def check(name, cond, detail=""):
        if rank == 0:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    # Each rank owns a distinct slice so the global batch contains both classes.
    labels = [1, 0]
    offset = rank * 2

    # ---- 1. default behavior is unchanged: batch_mean == legacy global_mean ----
    tr_a, m_a = make_trainer("batch_mean", device, tmpdir)
    loss_a = tr_a.compute_loss(m_a, make_batch(labels, device, offset))
    tr_b, m_b = make_trainer("batch_mean", device, tmpdir)
    tr_b.baseline_type = "irrelevant"
    tr_b.z0_mode = "global_mean"  # deprecated path must win and still work
    loss_b = tr_b.compute_loss(m_b, make_batch(labels, device, offset))
    check("legacy z0_mode=global_mean == baseline_type=batch_mean",
          approx(loss_a.item(), loss_b.item()), f"{loss_a.item():.8f} vs {loss_b.item():.8f}")

    tr_c, m_c = make_trainer("none", device, tmpdir)
    tr_c.z0_mode = "zero"
    loss_c = tr_c.compute_loss(m_c, make_batch(labels, device, offset))
    tr_d, m_d = make_trainer("none", device, tmpdir)
    loss_d = tr_d.compute_loss(m_d, make_batch(labels, device, offset))
    check("legacy z0_mode=zero == baseline_type=none",
          approx(loss_c.item(), loss_d.item()), f"{loss_c.item():.8f} vs {loss_d.item():.8f}")

    # ---- 2. running_global: first batch uses b_0 = 0, EMA updates only afterwards ----
    tr, model = make_trainer("running_global", device, tmpdir)
    check("running_global EMA starts at 0", tr._baseline_ema_global == 0.0)

    loss_rg = tr.compute_loss(model, make_batch(labels, device, offset))
    tr_zero, m_zero = make_trainer("none", device, tmpdir)
    with torch.no_grad():
        loss_zero, out = tr_zero.compute_loss(
            m_zero, make_batch(labels, device, offset), return_outputs=True)
    check("running_global first batch == none (b_0 = 0)",
          approx(loss_rg.item(), loss_zero.item()),
          f"{loss_rg.item():.8f} vs {loss_zero.item():.8f}")

    local_mean = out["r_mean_local"].item()
    gm = torch.tensor([local_mean * len(labels), float(len(labels))],
                      device=device, dtype=torch.float64)
    if dist.is_initialized():
        dist.all_reduce(gm)
    global_mean = (gm[0] / gm[1]).item()
    expected = DECAY * 0.0 + (1 - DECAY) * global_mean
    check("running_global EMA = 0.99*0 + 0.01*global_mean",
          approx(tr._baseline_ema_global, expected, 1e-4),
          f"{tr._baseline_ema_global:.8f} vs {expected:.8f}")

    prev = tr._baseline_ema_global
    tr.compute_loss(model, make_batch(labels, device, offset))
    expected2 = DECAY * prev + (1 - DECAY) * global_mean
    check("running_global EMA compounds across batches",
          approx(tr._baseline_ema_global, expected2, 1e-4),
          f"{tr._baseline_ema_global:.8f} vs {expected2:.8f}")

    # ---- 3. EMA identical on every rank ----
    if dist.is_initialized():
        buf = torch.tensor([tr._baseline_ema_global], device=device, dtype=torch.float64)
        gathered = [torch.zeros_like(buf) for _ in range(world)]
        dist.all_gather(gathered, buf)
        vals = [g.item() for g in gathered]
        check("running_global EMA bit-identical on all ranks",
              all(v == vals[0] for v in vals), f"n_distinct={len(set(vals))}")

    # ---- 4. running_class_conditional ----
    trc, mc = make_trainer("running_class_conditional", device, tmpdir)
    check("class EMAs start at 0", trc._baseline_ema_D == 0.0 and trc._baseline_ema_U == 0.0)
    loss_cc = trc.compute_loss(mc, make_batch(labels, device, offset))
    check("class-conditional first batch == none (b_D = b_U = 0)",
          approx(loss_cc.item(), loss_zero.item()),
          f"{loss_cc.item():.8f} vs {loss_zero.item():.8f}")
    check("both class EMAs moved off 0",
          trc._baseline_ema_D != 0.0 and trc._baseline_ema_U != 0.0,
          f"D={trc._baseline_ema_D:.6f} U={trc._baseline_ema_U:.6f}")
    check("class EMAs track different values",
          not approx(trc._baseline_ema_D, trc._baseline_ema_U),
          f"D={trc._baseline_ema_D:.6f} U={trc._baseline_ema_U:.6f}")

    if dist.is_initialized():
        buf = torch.tensor([trc._baseline_ema_D, trc._baseline_ema_U],
                           device=device, dtype=torch.float64)
        gathered = [torch.zeros_like(buf) for _ in range(world)]
        dist.all_gather(gathered, buf)
        vals = [tuple(g.tolist()) for g in gathered]
        check("class EMAs bit-identical on all ranks",
              all(v == vals[0] for v in vals), f"n_distinct={len(set(vals))}")

    # ---- 5. a class absent from the global batch must not update its EMA ----
    trd, md = make_trainer("running_class_conditional", device, tmpdir)
    trd.compute_loss(md, make_batch([1, 1], device, offset))
    check("absent class U keeps EMA at 0", trd._baseline_ema_U == 0.0, f"U={trd._baseline_ema_U}")
    check("present class D updated", trd._baseline_ema_D != 0.0, f"D={trd._baseline_ema_D:.6f}")

    tru, mu = make_trainer("running_class_conditional", device, tmpdir)
    tru.compute_loss(mu, make_batch([0, 0], device, offset))
    check("absent class D keeps EMA at 0", tru._baseline_ema_D == 0.0, f"D={tru._baseline_ema_D}")
    check("present class U updated", tru._baseline_ema_U != 0.0, f"U={tru._baseline_ema_U:.6f}")

    # ---- 6. eval batches must not move the EMA ----
    tre, me = make_trainer("running_global", device, tmpdir)
    me.eval()
    with torch.no_grad():
        tre.compute_loss(me, make_batch(labels, device, offset))
    check("eval batch leaves EMA untouched", tre._baseline_ema_global == 0.0,
          f"{tre._baseline_ema_global}")
    me.train()

    # ---- 7. baselines are stop-grad, gradients still flow through the policy term ----
    trg, mg = make_trainer("running_class_conditional", device, tmpdir)
    trg._baseline_ema_D, trg._baseline_ema_U = 1.5, -2.5
    z0 = trg._baseline_z0(torch.zeros(2, device=device, requires_grad=True),
                          torch.tensor([True, False], device=device),
                          "running_class_conditional")
    check("class-conditional z0 is stop-grad, per-example, correctly routed",
          (not z0.requires_grad) and tuple(z0.shape) == (2,)
          and approx(z0[0].item(), 1.5) and approx(z0[1].item(), -2.5),
          f"{z0.tolist()}")

    loss_g = trg.compute_loss(mg, make_batch(labels, device, offset))
    loss_g.backward()
    gnorm = sum(p.grad.norm().item() for p in mg.parameters() if p.grad is not None)
    check("gradient flows through the policy ELBO term", gnorm > 0 and math.isfinite(gnorm),
          f"|g|={gnorm:.6f}")

    # ---- 8. invalid config is rejected ----
    trx, mx = make_trainer("nope", device, tmpdir)
    try:
        trx.compute_loss(mx, make_batch(labels, device, offset))
        check("invalid baseline_type raises", False)
    except ValueError:
        check("invalid baseline_type raises", True)

    check("normalize_baseline_type maps legacy names",
          T.normalize_baseline_type("global_mean") == "batch_mean"
          and T.normalize_baseline_type("zero") == "none"
          and T.normalize_baseline_type("running_global") == "running_global")

    return failures


def main():
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if rank == 0:
        print(f"=== baseline variant tests (world_size={world}, device={device}) ===")

    tmpdir = f"/tmp/elbo_kto_test_{rank}"
    failures = run(device, tmpdir)

    nfail = torch.tensor([len(failures)], device=device)
    if dist.is_initialized():
        dist.all_reduce(nfail)
        dist.barrier()
    if rank == 0:
        n = int(nfail.item())
        print(f"=== {'ALL TESTS PASSED' if n == 0 else f'{n} FAILURE(S)'} ===")
    ok = int(nfail.item()) == 0
    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
