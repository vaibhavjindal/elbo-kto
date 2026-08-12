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

#### Controlling the desired/undesired composition of each global batch

The global baseline `z0 = mean(r_hat)` is all-reduced across **all ranks**, so its value depends
on the desired/undesired mix of the global batch. That global batch is
`world_size x per_device_train_batch_size` (gradient accumulation is *not* included — `z0` is
all-reduced once per micro-step). With the default 8 GPUs and microbatch 1 the global batch is 8,
matching the paper.

By default the mix is whatever `RandomSampler` produces. `--batch_composition` makes it explicit:

| value | behavior |
| --- | --- |
| `random` (default) | stock HF `RandomSampler` — unchanged baseline behavior |
| `balanced` | every global batch has `--desired_per_batch` desired samples (default: half) |
| `alternating` | homogeneous global batches, alternating all-desired / all-undesired |

Experiment 1 — every global batch is 4 desired + 4 undesired:
```
torchrun --nproc_per_node=8 train.py \
  --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
  --train_dataset_path data/kto-mix-14k-processed/train.jsonl \
  --batch_composition balanced --desired_per_batch 4 \
  --n_mc_samples 8 --z0_mode global_mean --kto_beta 0.1 \
  --learning_rate 1e-6 --warmup_ratio 0.03 --logging_steps 1 \
  --output_dir models/elbo-kto-balanced-4-4
```

Experiment 2 — alternating all-desired (8) / all-undesired (8) global batches:
```
torchrun --nproc_per_node=8 train.py \
  --model_name_or_path GSAI-ML/LLaDA-8B-Instruct \
  --train_dataset_path data/kto-mix-14k-processed/train.jsonl \
  --batch_composition alternating \
  --n_mc_samples 8 --z0_mode global_mean --kto_beta 0.1 \
  --learning_rate 1e-6 --warmup_ratio 0.03 --logging_steps 1 \
  --output_dir models/elbo-kto-alternating
```

Notes:
- Both classes are consumed fully. Once one class runs out, the remaining batches are filled from
  the other class; the run prints how many batches end up off-spec. The trailing samples that
  cannot fill a whole global batch are dropped (`dataloader_drop_last` is forced on for these
  modes so accelerate's `even_batches` tail-cycling cannot recycle early samples).
- `--print_batch_composition N` prints the realized per-rank layout of the first N global batches
  at startup so you can verify a run before it burns GPU hours. `--composition_seed` controls the
  within-class shuffle.
- Every logged step now includes `batch/frac_desired`, so the realized composition is visible in
  the training logs (expect a constant `0.5` for experiment 1, and `1.0`/`0.0` alternating for
  experiment 2).
- In `alternating`, an all-desired batch has `s_i = beta * (r_hat_i - mean(r_hat))` with every
  `s_i` sign `+1` and a zero-mean argument, so roughly half of the desired samples are pushed
  down at each step. This is expected behavior of the batch-mean baseline under a homogeneous
  batch, not a bug — it is precisely what the experiment probes.

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
