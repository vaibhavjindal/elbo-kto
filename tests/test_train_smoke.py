#!/usr/bin/env python3
"""
End-to-end smoke test: runs the real HF Trainer loop (ELBOKTOTrainer.train()) over a
synthetic dataset with a stub mask-predictor, for every baseline variant, and prints the
baseline trajectory captured from the trainer's own log history.

Also checks the CLI resolution of --baseline_type / deprecated --z0_mode.

  torchrun --nproc_per_node=8 tests/test_train_smoke.py
"""
import os
import sys
import argparse

import torch
import torch.distributed as dist
from datasets import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import TrainingArguments  # noqa: E402

import train as T  # noqa: E402
from test_baselines import (  # noqa: E402
    StubModel, build_example, DECAY, K, MASK_ID,
)

N_EXAMPLES = 64
STEPS = 8


def make_dataset(device):
    rows = []
    for i in range(N_EXAMPLES):
        ex = build_example(i, 1 if i % 2 == 0 else 0, device)
        ex["input_ids"] = ex["input_ids"].tolist()
        rows.append(ex)
    return Dataset.from_list(rows)


def run_variant(baseline_type, ds, tmpdir, rank):
    model = StubModel().cuda()
    args = TrainingArguments(
        output_dir=f"{tmpdir}/{baseline_type}",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=STEPS,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )
    tr = T.ELBOKTOTrainer(
        model=model, args=args, train_dataset=ds, data_collator=T.kto_data_collator,
    )
    tr.baseline_type = baseline_type
    tr.baseline_ema_decay = DECAY
    tr.z0_mode = None
    tr.n_mc_samples = K
    tr.kto_beta = 0.1
    tr.verify_masks = True
    tr.mask_token_id = MASK_ID
    tr.train()

    series = {}
    for rec in tr.state.log_history:
        for key in ("baseline/ema_global", "baseline/ema_D", "baseline/ema_U",
                    "baseline/z0_used_mean", "baseline/n_D", "baseline/n_U"):
            if key in rec:
                series.setdefault(key, []).append(rec[key])
    if rank == 0:
        print(f"\n--- {baseline_type} ---")
        for k, v in sorted(series.items()):
            print(f"  {k}: {[round(x, 6) for x in v]}")
    return series, tr


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    tmpdir = "/tmp/elbo_kto_smoke"
    failures = []

    def check(name, cond, detail=""):
        if rank == 0:
            print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
        if not cond:
            failures.append(name)

    ds = make_dataset(torch.device(f"cuda:{local_rank}"))
    if rank == 0:
        print(f"=== train() smoke test: {N_EXAMPLES} examples, {STEPS} steps, "
              f"world_size={dist.get_world_size()} ===")

    results = {}
    for bt in ("batch_mean", "none", "running_global", "running_class_conditional"):
        results[bt] = run_variant(bt, ds, tmpdir, rank)

    if rank == 0:
        print("\n=== assertions ===")

    # ---- log-series assertions (rank 0 only: self.log() is main-process-guarded) ----
    if rank == 0:
        # batch_mean / none never expose running EMAs
        for bt in ("batch_mean", "none"):
            s, _ = results[bt]
            check(f"{bt}: no running-EMA keys logged",
                  "baseline/ema_global" not in s and "baseline/ema_D" not in s)
        check("none: z0 is identically 0",
              all(v == 0.0 for v in results["none"][0].get("baseline/z0_used_mean", [])))
        check("batch_mean: z0 is non-zero (centering active)",
              any(v != 0.0 for v in results["batch_mean"][0].get("baseline/z0_used_mean", [])))

        # running_global: starts at 0, evolves, and step t uses the EMA from step t-1
        s_rg = results["running_global"][0]
        ema = s_rg.get("baseline/ema_global", [])
        check("running_global: EMA logged every step", len(ema) == STEPS, f"{len(ema)} records")
        check("running_global: EMA evolves (not stuck)", len(set(ema)) > 1)
        z0_rg = s_rg.get("baseline/z0_used_mean", [])
        # z0_used_mean round-trips through a float32 tensor; the EMA itself is float64.
        bad = [(i, z0_rg[i], ema[i - 1]) for i in range(1, min(len(z0_rg), len(ema) + 1))
               if abs(z0_rg[i] - ema[i - 1]) > 1e-5 * max(1.0, abs(ema[i - 1]))]
        check("running_global: step 1 used z0 = 0", z0_rg and z0_rg[0] == 0.0, f"{z0_rg[:1]}")
        check("running_global: step t used the EMA from step t-1",
              len(z0_rg) == len(ema) and not bad,
              f"len(z0)={len(z0_rg)} len(ema)={len(ema)} mismatches={bad[:3]}")

        # running_class_conditional
        s_cc = results["running_class_conditional"][0]
        check("class_conditional: both EMAs logged",
              len(s_cc.get("baseline/ema_D", [])) == STEPS
              and len(s_cc.get("baseline/ema_U", [])) == STEPS)
        check("class_conditional: EMAs diverge from each other",
              s_cc["baseline/ema_D"][-1] != s_cc["baseline/ema_U"][-1],
              f"D={s_cc['baseline/ema_D'][-1]} U={s_cc['baseline/ema_U'][-1]}")
        check("class_conditional: both classes present each step",
              all(n > 0 for n in s_cc.get("baseline/n_D", []))
              and all(n > 0 for n in s_cc.get("baseline/n_U", [])))

        # loss must actually differ between variants (the baseline is doing something)
        losses = {}
        for bt, (_, tr) in results.items():
            ls = [r["loss"] for r in tr.state.log_history if "loss" in r]
            losses[bt] = ls[0] if ls else None
        check("variants produce different losses",
              len({round(v, 8) for v in losses.values() if v is not None}) > 1, f"{losses}")

    # ---- cross-rank agreement after a full training run (all ranks participate) ----
    tr_rg = results["running_global"][1]
    tr_cc = results["running_class_conditional"][1]
    buf = torch.tensor([tr_rg._baseline_ema_global, tr_cc._baseline_ema_D,
                        tr_cc._baseline_ema_U], device="cuda", dtype=torch.float64)
    gathered = [torch.zeros_like(buf) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, buf)
    vals = [tuple(g.tolist()) for g in gathered]
    check("all EMAs identical across ranks after full train()",
          all(v == vals[0] for v in vals), f"n_distinct={len(set(vals))}")
    check("running EMAs actually moved during train()",
          tr_rg._baseline_ema_global != 0.0 and tr_cc._baseline_ema_D != 0.0,
          f"global={tr_rg._baseline_ema_global:.6f} D={tr_cc._baseline_ema_D:.6f}")

    # ---- CLI resolution ----
    def mkargs(**kw):
        d = dict(baseline_type=None, z0_mode=None, baseline_ema_decay=0.99)
        d.update(kw)
        return argparse.Namespace(**d)

    a = mkargs(); T.resolve_baseline_args(a)
    check("CLI: default resolves to batch_mean", a.baseline_type == "batch_mean")
    a = mkargs(z0_mode="global_mean"); T.resolve_baseline_args(a)
    check("CLI: legacy global_mean -> batch_mean",
          a.baseline_type == "batch_mean" and a.z0_mode is None)
    a = mkargs(z0_mode="zero"); T.resolve_baseline_args(a)
    check("CLI: legacy zero -> none", a.baseline_type == "none")
    a = mkargs(baseline_type="running_global", z0_mode="zero")
    try:
        T.resolve_baseline_args(a)
        check("CLI: conflicting flags raise", False)
    except ValueError:
        check("CLI: conflicting flags raise", True)
    a = mkargs(baseline_type="batch_mean", z0_mode="global_mean")
    T.resolve_baseline_args(a)
    check("CLI: agreeing flags accepted", a.baseline_type == "batch_mean")
    for bad in (1.0, -0.1, 1.5):
        a = mkargs(baseline_ema_decay=bad)
        try:
            T.resolve_baseline_args(a)
            check(f"CLI: decay={bad} rejected", False)
        except ValueError:
            check(f"CLI: decay={bad} rejected", True)

    nfail = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(nfail)
    dist.barrier()
    n = int(nfail.item())
    if rank == 0:
        print(f"\n=== {'ALL SMOKE TESTS PASSED' if n == 0 else f'{n} FAILURE(S)'} ===")
    dist.destroy_process_group()
    sys.exit(0 if n == 0 else 1)


if __name__ == "__main__":
    main()
