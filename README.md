## ELBO-KTO: Training Code

This repository contains the minimal code to reproduce ELBO-KTO training in two stages:

- Precompute reference estimator values B_ref for each example
- Train the policy using the precomputed values


### 1) Environment
```
pip install -r requirements.txt
```

### 2) Data
This project expects the KTO dataset locally.

Option A: Download the public dataset locally with the Hugging Face CLI:
```
hf download trl-lib/kto-mix-14k --repo-type=dataset --local-dir=data/kto-mix-14k
```

Option B: Point to your own dataset that matches the same JSON structure used here.

### 3) Precompute B_ref
Precompute the deterministic fixed-ℓ reference estimator values per sample and save them to a JSONL file alongside the original fields.

Example:
```
python precompute_bref.py \
  --num_gpus 8 \
  --batch_size 8 \
  --model_path GSAI-ML/LLaDA-8B-Instruct \
  --dataset data/kto-mix-14k \
  --k_vals 1,2,4,8 \
  --output_file data/kto-mix-14k-processed/train.jsonl \
  --split train \
  --max_samples 64
```

Notes:
- `--dataset` should be a local HF dataset directory.
- The output file will contain precomputed fields: `seed`, `l_values`, `B_ref` (per-K), and `masked_idx_sums`.

### 4) Train with ELBO-KTO
Train the policy using the precomputed JSONL produced above:
```
torchrun --nproc_per_node=8 train.py \
  --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
  --train_dataset_path data/kto-mix-14k-processed/train.jsonl \
  --logging_steps 1 \
  --n_mc_samples 8 \
  --z0_mode global_mean \
  --kto_beta 0.1 \
  --learning_rate 1e-6 \
  --warmup_ratio 0.03 \
  --output_dir models/elbo-kto-finetuned
```

Key points:
- `--train_dataset_path` must point to the JSONL created in the precompute step.
- `--n_mc_samples` must be one of the K values you precomputed.

#### Baseline (z0) variants

The KTO value function centers each ELBO margin `r̂ = B̂_πθ − B̂_πref` by a stop-gradient
baseline `z0`. Select it with `--baseline_type`:

| `--baseline_type` | z0 for example *i* | Notes |
| --- | --- | --- |
| `batch_mean` (default) | mean of `r̂` over the current **global** batch | Paper default; variance-optimal among batch-constant baselines (Lemma 1) |
| `none` | `0` | No centering |
| `running_global` | `b_{t-1}` | EMA over past global batch margin means |
| `running_class_conditional` | `b^D_{t-1}` if desirable, else `b^U_{t-1}` | Two independent EMAs |

The running variants use `b_t = decay * b_{t-1} + (1 - decay) * r̄_t`, with `b_0 = 0` and
`decay` set by `--baseline_ema_decay` (default `0.99`). The value applied to batch *t* is
the EMA from **before** that batch; the EMA is updated only after the batch loss has been
computed. Margin means are all-reduced across ranks first, so every rank holds an identical
baseline. For `running_class_conditional`, a class that is absent from the global batch
leaves its EMA untouched. All baselines are stop-gradient.

```
torchrun --nproc_per_node=8 train.py \
  --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
  --train_dataset_path data/kto-mix-14k-processed/train.jsonl \
  --n_mc_samples 8 --kto_beta 0.1 --learning_rate 1e-6 \
  --baseline_type running_class_conditional --baseline_ema_decay 0.99 \
  --output_dir models/elbo-kto-running-cc
```

Baselines are logged every `--logging_steps` for inspection: `baseline/z0_used_mean`
(the baseline actually applied), `baseline/ema_global` or `baseline/ema_D` +
`baseline/ema_U`, plus `baseline/margin_mean_D|U` and `baseline/n_D|n_U`.

`--z0_mode` is still accepted as a deprecated alias (`global_mean` → `batch_mean`,
`zero` → `none`).

Baseline behavior is covered by `tests/test_baselines.py` (unit) and
`tests/test_train_smoke.py` (end-to-end `train()` loop); both run under `torchrun`:
```
torchrun --nproc_per_node=8 tests/test_baselines.py
```

### 5) Reproducibility
- Mask generation is deterministic per example using fixed 64-bit seeds; training re-derives the same per-draw masks and verifies them (configurable).
- BF16 is enabled by default; adjust per hardware if needed.

### 6) Inference (LLADA instruct style models)
We split the test data from kto-mix-14k into chosen and rejected responses and include it under `data/kto-mix-14k-test` to demonstrate inference.

- `data/kto-mix-14k-test/chosen.jsonl`
- `data/kto-mix-14k-test/rejected.jsonl`

Use `inference.py` to generate model responses for the test prompts. Example:
```
python inference.py \
  --model_path=GSAI-ML/LLaDA-8B-Instruct \
  --dataset_path=data/kto-mix-14k-test/chosen.jsonl \
  --max_samples=10 \
  --output_path=generated_responses.jsonl
```

Notes:
- Set `--num_gpus` to leverage multiple GPUs (the script splits the dataset across devices).
- Optional knobs: `--gen_length`, `--steps`, `--block_length`, `--remasking` (defaults are tuned for LLADA).
- Output is a JSONL with records like `{id, prompt, completion: [{role: "assistant", content: "..."}]}`.
- Set `--model_path` to point to your trained checkpoint to generate responses from your trained model.

### 7) Evaluation on downstream tasks
We provide `eval_llada.sh` to run evaluation for LLADA-style models on a few standard downstream tasks using `lm_eval` via our `eval_llada.py` runner.

Quick start:
```
bash eval_llada.sh
```

What it runs:
- mmlu_generative (5-shot)
- gsm8k (5-shot)
- humaneval_instruct_sanitized (with unsafe code confirmation)

Notes:
- The script installs required versions of `transformers`, `lm_eval`, and `accelerate`.
- Set `LLADA_INSTRUCT` inside the script to your model path (e.g., `GSAI-ML/LLaDA-8B-Instruct`).
- Generation knobs like `gen_length`, `steps`, and `block_length` are passed through `--model_args` and can be adjusted per task.

## Cite this work
```
@misc{jindal2025aligningdiffusionlanguagemodels,
      title={Aligning Diffusion Language Models via Unpaired Preference Optimization}, 
      author={Vaibhav Jindal and Hejian Sang and Chun-Mao Lai and Yanning Chen and Zhipeng Wang},
      year={2025},
      eprint={2510.23658},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.23658}, 
}
```
