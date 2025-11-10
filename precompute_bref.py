#!/usr/bin/env python3
import os, json, argparse
import multiprocessing as mp
from typing import List, Dict, Any
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
 

from data_utils import DataProcessor

# ====== Constants (adjust if needed) ======
GLOBAL_SEED = 0x1A2B3C4D5E6F7788  # controls per-example base seeds

# ====== Multi-GPU Utilities ======

def get_available_gpus():
    """Get list of available GPU device IDs."""
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))

def setup_gpu_device(gpu_id: int) -> torch.device:
    """Setup and return device for given GPU ID."""
    if gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        return device
    else:
        return torch.device("cpu")

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

# ====== Deterministic seed/mask helpers ======

def mix64(x: int) -> int:
    """SplitMix64 mix step: deterministic 64-bit scrambling."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF

def derive_seed(base_seed: int, k: int, tag: int) -> int:
    """Derive a new 64-bit seed from base seed, k (>=1), and a small tag (0=t, 1=mask)."""
    x = base_seed ^ (k * 0x9E3779B97F4A7C15) ^ (tag * 0xD1342543DE82EF95)
    return mix64(x)


def l_from_seed(seed: int, L_resp: int) -> int:
    """
    Generate uniform l ∈ {1, ..., L_resp} from a 64-bit seed.
    
    Args:
        seed: 64-bit seed value for deterministic generation
        L_resp: Response length (if L_resp == 0, returns 0)
        
    Returns:
        Uniformly sampled l value in range [1, L_resp], or 0 if L_resp <= 0
    """
    if L_resp <= 0:
        return 0
    return 1 + (mix64(seed) % L_resp)

def make_batched_fixedl_masks(
    select_seeds: List[int],       # [B] seed per example for selecting indices
    L: int,                        # full sequence length (all same)
    prompt_lens: List[int],        # [B]
    comp_lens: List[int],          # [B] completion lengths
    l_values: List[int],           # [B] chosen l for each example
    device: torch.device,
) -> torch.BoolTensor:
    """
    Build batched masks where exactly l tokens are masked in the response window.
    
    Creates [B, L] boolean masks where each example has exactly l_values[b] tokens
    masked within the response window [prompt_len, prompt_len + completion_length).
    Uses deterministic sampling without replacement.
    
    Args:
        select_seeds: List of seeds for deterministic index selection per example
        L: Full sequence length (same for all examples)
        prompt_lens: List of prompt lengths per example
        comp_lens: List of completion lengths per example  
        l_values: List of number of tokens to mask per example
        device: Target device for the mask tensor
        
    Returns:
        Boolean tensor of shape [B, L] with masked positions set to True
    """
    B = len(select_seeds)
    masks = torch.zeros((B, L), dtype=torch.bool, device=device)
    
    for b in range(B):
        p = int(prompt_lens[b])
        c = int(comp_lens[b])
        l = int(l_values[b])
        
        if c <= 0 or l <= 0:
            continue
            
        l = min(l, c)
        start, end = p, p + c              # response window [start, end)
        n = end - start                    # == c
        
        # Deterministic permutation via generator
        g = torch.Generator(device=device)
        g.manual_seed(select_seeds[b] & 0xFFFFFFFFFFFFFFFF)
        r = torch.rand((n,), generator=g, device=device)
        perm = torch.argsort(r)[:l]        # first l indices in window
        idx = start + perm
        masks[b, idx] = True
        
    return masks



@torch.no_grad()
def elbo_batched_fixedl(
    model: AutoModel,
    input_ids_batch: torch.LongTensor,    # [B, L]
    mask_batch: torch.BoolTensor,         # [B, L] (exactly l masked in response window or 0)
) -> List[float]:
    """
    Compute fixed-l ELBO estimator for a batch of sequences.
    
    For each example in the batch, computes:
    B_draw = (1/l) * sum_{i in masked} log p(y_i | y^masked)
    
    Args:
        model: Pre-trained language model
        input_ids_batch: Batch of tokenized sequences [B, L]
        mask_batch: Boolean mask indicating which tokens to mask [B, L]
                   Each example should have exactly l masked tokens in response window
                   
    Returns:
        List of float values (length B), where each value is the mean log probability
        over masked positions for that example. Returns 0.0 if no tokens are masked.
    """
    device = input_ids_batch.device
    B, L = input_ids_batch.shape

    # Create noisy input: replace masked tokens with MASK_ID
    noisy = input_ids_batch.clone()
    noisy[mask_batch] = MASK_ID

    # Forward pass to get logits
    out = model(input_ids=noisy)          # logits: [B, L, V]
    logits = out.logits

    vals: List[float] = []
    for b in range(B):
        m = mask_batch[b]
        if not m.any():
            vals.append(0.0)
            continue
            
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        # Cross-entropy = -log p; we want mean log p over masked positions
        ce = F.cross_entropy(
            logits[b].index_select(0, idx), 
            input_ids_batch[b].index_select(0, idx), 
            reduction="none"
        )
        vals.append(float((-ce).mean().item()))
        
    return vals


@torch.no_grad()
def bref_for_batch(
    model: AutoModel,
    input_ids_batch: torch.LongTensor,    # [B, L]
    prompt_lens: List[int],               # [B]
    comp_lens: List[int],                 # [B]  <-- NEW
    base_seeds: List[int],                # [B]
    K_max: int,
) -> List[Dict[str, Any]]:
    """
    Compute fixed-l estimator with response-only masking for a batch of sequences.
    
    For each example, computes B_ref estimates using deterministic masking within
    the response window. Uses K_max draws with running prefix means.
    
    Args:
        model: Pre-trained language model
        input_ids_batch: Batch of tokenized sequences [B, L]
        prompt_lens: List of prompt lengths per example [B]
        comp_lens: List of completion lengths per example [B]
        base_seeds: List of base seeds for deterministic generation [B]
        K_max: Maximum number of draws K for estimation
        
    Returns:
        List of dictionaries (length B), each containing:
        - "l_values": List of l values used for each draw [l_1, ..., l_K_max]
        - "B_prefix": List of prefix means [mean_1, ..., mean_K_max]  
        - "masked_idx_sums": List of sums of masked indices per draw
    """
    device = input_ids_batch.device
    B, L = input_ids_batch.shape

    # Prepare containers for results
    per_ex = [{"l_values": [], "per_draw": [], "masked_idx_sums": []} for _ in range(B)]
    running = torch.zeros(B, device=device, dtype=torch.float64)

    for k in range(1, K_max + 1):
        # Derive ℓ and selection seed deterministically for each example
        l_values_k, select_seeds_k = [], []
        for b in range(B):
            base = int(base_seeds[b])
            L_resp = int(comp_lens[b])
            l_seed = derive_seed(base, k, tag=0)   # drives l
            sel_seed = derive_seed(base, k, tag=1)  # drives which positions
            l = l_from_seed(l_seed, L_resp)
            l_values_k.append(l)
            select_seeds_k.append(sel_seed)

        # Build masks (exactly ℓ masked positions inside the response window)
        masks = make_batched_fixedl_masks(
            select_seeds_k, L, prompt_lens, comp_lens, l_values_k, device
        )

        # One forward pass; compute mean log p over masked positions (per example)
        draw_vals = elbo_batched_fixedl(model, input_ids_batch, masks)  # List[float]
        # Multiply each value by corresponding comp_len value
        draw_vals = [v * int(comp_lens[b]) for b, v in enumerate(draw_vals)]

        # Accumulate prefix means
        running += torch.tensor(draw_vals, device=device, dtype=torch.float64)
        means = (running / float(k)).tolist()

        # Store results for this draw
        for b in range(B):
            per_ex[b]["l_values"].append(int(l_values_k[b]))
            per_ex[b]["per_draw"].append(float(draw_vals[b]))
            
            # Optional: sum of masked absolute indices
            ms = masks[b].nonzero(as_tuple=False).squeeze(1)
            per_ex[b]["masked_idx_sums"].append(int(ms.sum().item()) if ms.numel() > 0 else 0)

    # Finalize: compute prefix means from per-draw values
    out: List[Dict[str, Any]] = []
    for b in range(B):
        # Recompute prefix means from per_draw for safety/consistency
        s = 0.0
        B_prefix = []
        for v in per_ex[b]["per_draw"]:
            s += v
            B_prefix.append(s / len(B_prefix + [None]))
            
        out.append({
            "l_values": per_ex[b]["l_values"],
            "B_prefix": B_prefix,
            "masked_idx_sums": per_ex[b]["masked_idx_sums"],
        })
        
    return out


def prepare_batch(examples: List[Dict], device: torch.device) -> tuple:
    """
    Prepare a batch of examples for batched processing.
    
    Args:
        examples: List of dataset examples
        device: Target device
        
    Returns:
        Tuple of (input_ids_batch, prompt_lens, comp_lens, indices, orig_data)
    """
    batch_size = len(examples)
    
    # All sequences have the same length, so we can get it from the first example
    L = len(examples[0]["input_ids"])
    
    # Prepare batched tensors (no padding needed)
    input_ids_batch = torch.zeros((batch_size, L), dtype=torch.long, device=device)
    
    prompt_lens = []
    comp_lens = []
    indices = []
    orig_data = []
    
    for i, ex in enumerate(examples):
        input_ids = ex["input_ids"]
        prompt_len = int(ex["prompt_length"])
        comp_len = int(ex["completion_length"])
        
        # Fill in the sequence (all same length, no padding)
        input_ids_batch[i] = torch.tensor(input_ids, device=device, dtype=torch.long)
        
        prompt_lens.append(prompt_len)
        comp_lens.append(comp_len)
        indices.append(ex.get("index", i))
        orig_data.append({
            "orig_prompt": ex.get("orig_prompt", ""),
            "orig_completion": ex.get("orig_completion", ""),  
            "orig_label": ex.get("orig_label", None),
            "labels": ex.get("labels", ex.get("label", None)),
            "prompt_length": ex.get("prompt_length", None),
            "completion_length": ex.get("completion_length", None),
            "text_length": ex.get("text_length", None),
        })
    
    return input_ids_batch, prompt_lens, comp_lens, indices, orig_data


# ====== Multi-GPU Worker Function ======

def resolve_mask_id(tokenizer, model=None, cli_mask_id=None, cli_mask_token=None):
    if cli_mask_id is not None:
        return int(cli_mask_id)
    if cli_mask_token:
        tok_id = tokenizer.convert_tokens_to_ids(cli_mask_token)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            return int(tok_id)
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


def worker_process(gpu_id: int, model_path: str, dataset_arg: str, split: str, K_vals: List[int], 
                   work_range: tuple, batch_size: int, output_dir: str, worker_id: int):
    """
    Worker process for multi-GPU processing. Each worker processes a subset of examples
    on its assigned GPU and writes results to a separate file.
    """
    try:
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
        processor = DataProcessor(tokenizer, caching=False)
        ds = processor.load_dataset(dataset_arg, split=split)
        
        # Load model on this GPU
        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if "llada" in model_path.lower():
            model_kwargs["flash_attention"] = True
        model = AutoModel.from_pretrained(
            model_path,
            **model_kwargs
        ).to(device)
        model.eval()
        try:
            model.config.use_cache = False
        except Exception:
            pass
        
        # Resolve mask_id after loading model
        global MASK_ID
        MASK_ID = resolve_mask_id(
            tokenizer=tokenizer,
            model=model,
            cli_mask_id=getattr(args, "mask_id", None) if 'args' in globals() else None,
            cli_mask_token=getattr(args, "mask_token", None) if 'args' in globals() else None,
        )
        print(f"Worker {worker_id}: using MASK_ID={MASK_ID}")

        # Prepare output file for this worker
        worker_output_file = os.path.join(output_dir, f"worker_{worker_id}.jsonl")
        
        start_idx, end_idx = work_range
        total_examples = end_idx - start_idx
        K_max = max(K_vals)
        
        with open(worker_output_file, "w", encoding="utf-8") as f_out:
            # Process this worker's subset in batches
            for batch_start in range(start_idx, end_idx, batch_size):
                batch_end = min(batch_start + batch_size, end_idx)
                current_batch_size = batch_end - batch_start
                
                batch_in_worker = (batch_start - start_idx) // batch_size + 1
                total_batches_in_worker = (total_examples + batch_size - 1) // batch_size
                print(f"Worker {worker_id}: Processing batch {batch_in_worker}/{total_batches_in_worker} "
                      f"(global examples {batch_start}-{batch_end-1})")
                
                # Prepare batch examples
                batch_examples = []
                batch_base_seeds = []
                
                for idx in range(batch_start, batch_end):
                    ex = ds[int(idx)]
                    ex_with_idx = dict(ex)
                    ex_with_idx["index"] = idx
                    batch_examples.append(ex_with_idx)
                    
                    # Generate deterministic per-example base seed (same as single-GPU version)
                    base_seed = mix64(GLOBAL_SEED ^ idx)
                    batch_base_seeds.append(base_seed)
                
                # Prepare batch tensors
                input_ids_batch, prompt_lens, comp_lens, indices, orig_data = prepare_batch(
                    batch_examples, device
                )
                
                # Compute B_ref values for the entire batch
                batch_results = bref_for_batch(
                    model, input_ids_batch, prompt_lens, comp_lens,
                    batch_base_seeds, K_max
                )
                
                # Write results for each example in the batch
                for i in range(current_batch_size):
                    idx = indices[i]
                    base_seed = batch_base_seeds[i]
                    res = batch_results[i]
                    orig = orig_data[i]
                    
                    # Select only requested K values
                    B_ref_map = {str(K): float(res["B_prefix"][K - 1]) for K in K_vals}
                    
                    rec = {
                        "index": idx,
                        "seed": str(base_seed),
                        "l_values": [int(l) for l in res["l_values"]],
                        "B_ref": B_ref_map,
                        "masked_idx_sums": [int(s) for s in res["masked_idx_sums"]],
                        "prompt": orig["orig_prompt"],
                        "completion": orig["orig_completion"], 
                        "label": orig["orig_label"],
                        "prompt_length": orig["prompt_length"],
                        "completion_length": orig["completion_length"],
                        "text_length": orig["text_length"],
                    }
                    if orig["labels"] is not None:
                        rec["labels"] = int(orig["labels"])
                    
                    f_out.write(json.dumps(rec) + "\n")
        
        print(f"Worker {worker_id} completed: {total_examples} examples written to {worker_output_file}")
        return worker_output_file
        
    except Exception as e:
        print(f"Worker {worker_id} failed with error: {e}")
        raise

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
    
    # Sort by index to maintain original order
    all_records.sort(key=lambda x: x['index'])
    
    # Write to final output file
    os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)
    with open(final_output_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")
    
    print(f"Merged {len(all_records)} records into {final_output_path}")

# ====== Main Functions ======

def main(model_path: str, dataset_arg: str, K_vals: List[int], output_file_path: str, 
         batch_size: int = 1, num_gpus: int = None, max_samples: int = None, split: str = "train"):
    """
    Multi-GPU implementation using multiprocessing. Each GPU gets a separate process
    that handles a subset of the dataset. For single GPU usage, set num_gpus=1.
    """
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
    processor = DataProcessor(tokenizer, caching=False)
    ds = processor.load_dataset(dataset_arg, split=split)
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
    
    # Start worker processes
    processes = []
    for worker_id, (gpu_id, work_range) in enumerate(zip(available_gpus[:num_gpus], work_splits)):
        if work_range[0] >= work_range[1]:  # Skip empty ranges
            continue
            
        p = mp.Process(
            target=worker_process,
            args=(gpu_id, model_path, dataset_arg, split, K_vals, work_range, batch_size, 
                  str(output_dir), worker_id)
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




# ====== CLI ======

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute deterministic B_ref values (dense) with GPU support.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use all available GPUs (default)
  python precompute_bref.py --model_path model/ --dataset data.jsonl --k_vals 1,4,8 --output_file out.jsonl
  
  # Use single GPU
  python precompute_bref.py --num_gpus 1 --model_path model/ --dataset data/ --k_vals 1,4,8 --output_file out.jsonl
  
  # Use only 2 GPUs with larger batches and limit to 1000 samples
  python precompute_bref.py --num_gpus 2 --batch_size 8 --max_samples 1000 --model_path model/ --dataset data/ --k_vals 1,4,8 --output_file out.jsonl
        """
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to model directory or HuggingFace model name")
    parser.add_argument("--dataset", type=str, required=True,
                        help="HF hub name, local HF dataset dir (load_from_disk), or JSON/JSONL file")
    parser.add_argument("--k_vals", type=str, required=True, 
                        help="Comma-separated list of K values, e.g. '1,4,8'")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Output JSONL file path")
    parser.add_argument("--batch_size", type=int, default=1, 
                        help="Batch size per GPU for processing examples (default: 1)")
    parser.add_argument("--num_gpus", type=int, default=None,
                        help="Number of GPUs to use (default: all available). Set to 1 for single GPU.")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    parser.add_argument("--split", type=str, default="train",
                        help="Split to process (default: train)")
    parser.add_argument("--mask_id", type=int, default=None, help="Override mask token id")
    parser.add_argument("--mask_token", type=str, default=None, help="Override mask token string (e.g., '<|mask|>')")
    args = parser.parse_args()

    # Print configuration
    print("=" * 60)
    print("B_REF PRECOMPUTATION")
    print("=" * 60)
    available_gpus = get_available_gpus()
    print(f"Available GPUs: {len(available_gpus)} {available_gpus if available_gpus else '(none - using CPU)'}")
    if args.num_gpus:
        print(f"Requested GPUs: {args.num_gpus}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model_path}")
    print(f"K values: {args.k_vals}")
    print(f"Output: {args.output_file}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print("=" * 60)

    # Parse K values
    K_vals = [int(x) for x in args.k_vals.split(",") if x.strip()]
    
    # Validate arguments
    if args.num_gpus is not None:
        if args.num_gpus <= 0:
            parser.error("--num_gpus must be positive")
        if args.num_gpus > len(available_gpus) and available_gpus:
            parser.error(f"Requested {args.num_gpus} GPUs but only {len(available_gpus)} available")
    
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    # Run main function
    main(args.model_path, args.dataset, K_vals, args.output_file, 
         args.batch_size, args.num_gpus, args.max_samples, args.split)