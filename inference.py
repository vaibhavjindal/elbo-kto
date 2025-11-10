#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path
from typing import List
import multiprocessing as mp

import torch
import torch.distributed as dist
from tqdm import tqdm

from generate_llada import generate
from transformers import (
    AutoTokenizer, 
    AutoModel, 
)

from datasets import load_dataset

"""
Example usage:
python inference.py \
  --model_path=GSAI-ML/LLaDA-8B-Instruct \
  --dataset_path=data/kto-mix-14k-test/chosen.jsonl \
  --max_samples=10 \
  --output_path=generated_responses.jsonl \
  --num_gpus=1 \
  --gen_length=512 --steps=512 --block_length=32 --remasking=low_confidence
"""

def is_main_process():
    return (not dist.is_initialized()) or dist.get_rank() == 0

def setup_gpu_device(gpu_id: int) -> torch.device:
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")

def init_distributed():
    """Initialize torch.distributed when launched with `torchrun`."""
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        print(f"[Rank {dist.get_rank()}] Running on GPU {local_rank}")
    else:
        local_rank = 0
    return local_rank


def parse_args():
    parser = argparse.ArgumentParser(description="LLADA Inference")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output file")
    parser.add_argument("--gen_length", type=int, default=512, help="Generation length")
    parser.add_argument("--steps", type=int, default=512, help="Number of steps")
    parser.add_argument("--block_length", type=int, default=32, help="Block length")
    parser.add_argument("--remasking", type=str, default="low_confidence", choices=["low_confidence", "random"], help="Remasking strategy")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")


    return parser

def split_work(total_items: int, num_workers: int) -> List[tuple]:
    """Split work evenly across workers. Returns list of (start_idx, end_idx) tuples."""
    items_per_worker = total_items // num_workers
    remainder = total_items % num_workers

    splits = []
    start = 0
    for i in range(num_workers):
        # Distribute remainder across first few workers
        extra = 1 if i < remainder else 0
        end = start + items_per_worker + extra
        splits.append((start, min(end, total_items)))
        start = end

    return splits


def get_available_gpus():
    """Get list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def worker_process(gpu_id: int, model_path: str, data_path: str, work_range: tuple, output_dir: str, worker_id: int, args):
    """Process a subset of the dataset on a specific GPU."""
    # Setup GPU device for this worker
    device = setup_gpu_device(gpu_id)
    print(f"Worker {worker_id} starting on device {device} (examples {work_range[0]}-{work_range[1]-1})")

    # Determinism setup for this worker
    torch.manual_seed(1234 + worker_id)  # Slight offset per worker for independence
    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load dataset (each worker loads independently to avoid shared memory issues)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ds = load_dataset("json", data_files=data_path)["train"]

    # Load model on this GPU
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        flash_attention=True,
    ).to(device)
    model.eval()
    try:
        model.config.use_cache = False
    except Exception:
        pass

    # Prepare output file for this worker
    worker_output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")

    start_idx, end_idx = work_range
    total_examples = end_idx - start_idx
    with open(worker_output_file, "w", encoding="utf-8") as f_out:
        # Process this worker's subset in batches
        for idx in tqdm(range(start_idx, end_idx), desc=f"Worker {worker_id} processing"):
            prompt = tokenizer.apply_chat_template(ds[idx]['prompt'], add_generation_prompt=True, tokenize=False)
            input_ids = tokenizer(prompt)["input_ids"]
            prompt_tensor = torch.tensor(input_ids, device=model.device).unsqueeze(0)

            generated_answer = generate(
                model,
                prompt_tensor,
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_length,
                temperature=0,
                cfg_scale=0,
                remasking=args.remasking,
                mask_id=126336,
            )

            generated_answer = tokenizer.decode(
                generated_answer[0][prompt_tensor.shape[1]:], skip_special_tokens=False
            )
            # remove special tokens
            generated_answer_ids = tokenizer(generated_answer)["input_ids"]
            generated_answer = tokenizer.decode(
                generated_answer_ids, skip_special_tokens=True
            )
            rec = {"id": ds[idx]["id"], "prompt": ds[idx]["prompt"], "completion": [{"content": generated_answer, "role": "assistant"}]}
            f_out.write(json.dumps(rec) + "\n")
    print(f"Worker {worker_id} completed: {total_examples} examples written to {worker_output_file}")
    return worker_output_file

def merge_worker_outputs(output_files: List[str], final_output_path: str):
    """Merge worker output files into final output file, sorted by index."""
    print("Merging worker outputs...")

    # Read all records from worker files
    all_records = []
    for worker_file in output_files:
        if os.path.exists(worker_file):
            with open(worker_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line.strip())
                        all_records.append(record)
            # Clean up worker file
            os.remove(worker_file)


    # Write to final output file
    os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)
    with open(final_output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"Merged {len(all_records)} records into {final_output_path}")

def main(args):
    model_path = args.model_path
    data_path = args.dataset_path
    max_samples = args.max_samples
    output_file_path = args.output_path
    num_gpus = args.num_gpus
    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        print("No GPUs available, using CPU mode")
        available_gpus = [-1]  # Use CPU with device ID -1

    # Determine number of GPUs to use
    if num_gpus is None:
        num_gpus = len(available_gpus)
    else:
        num_gpus = min(num_gpus, len(available_gpus))

    print(f"Using {num_gpus} GPUs: {available_gpus[:num_gpus]}")

    # Load dataset to get total size (lightweight load just for counting)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    ds = load_dataset("json", data_files=data_path)["train"]
    n = len(ds)

    # Apply max_samples limit if specified
    if max_samples is not None and max_samples < n:
        n = max_samples
        print(f"Limiting processing to {n} examples (max_samples={max_samples})")

    # Split work across GPUs
    work_splits = split_work(n, num_gpus)
    print(f"Work distribution across {num_gpus} GPUs:")
    for i, (start, end) in enumerate(work_splits):
        print(f"  GPU {available_gpus[i]}: examples {start}-{end-1} ({end-start} examples)")

    # Create temporary output directory for worker files
    output_dir = Path(output_file_path).parent / f"tmp_workers_{os.getpid()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Temporary worker output directory: {output_dir}")

    # Start worker processes
    processes = []
    for worker_id, (gpu_id, work_range) in enumerate(zip(available_gpus[:num_gpus], work_splits)):
        if work_range[0] >= work_range[1]:  # Skip empty ranges
            continue

        p = mp.Process(
            target=worker_process,
            args=(gpu_id, model_path, data_path, work_range,
                  str(output_dir), worker_id, args)
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    print("Waiting for all workers to complete...")
    worker_output_files = []
    for worker_id, p in enumerate(processes):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Worker {worker_id} failed with exit code {p.exitcode}")

        # Add expected output file
        worker_file = output_dir / f"worker_{worker_id}.jsonl"
        if worker_file.exists():
            worker_output_files.append(str(worker_file))

    # Merge worker outputs
    merge_worker_outputs(worker_output_files, output_file_path)

    # Clean up temporary directory
    try:
        output_dir.rmdir()
    except OSError:
        pass  # Directory not empty or other issues, but that's OK

    print(f"[Done] Multi-GPU processing complete. Results written to {output_file_path}")


if __name__ == "__main__":
    parser = parse_args()
    args = parser.parse_args()

    available_gpus = get_available_gpus()
    
    if args.num_gpus <= 0:
        parser.error("--num_gpus must be positive")
    if args.num_gpus > len(available_gpus) and available_gpus:
        parser.error(f"Requested {args.num_gpus} GPUs but only {len(available_gpus)} available")
    
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main(args)
    
    
    