from datasets import load_dataset, Features, Value
import datasets

# Constants
MAX_LENGTH = 4096
DATA_PATH = "data/kto-mix-14k"


def resolve_eos_id(tokenizer, cli_eos_id=None, cli_eos_token=None):
    """Resolve EOS token id from overrides or tokenizer. Raise if not found."""
    if cli_eos_id is not None:
        return int(cli_eos_id)
    if cli_eos_token:
        tok_id = tokenizer.convert_tokens_to_ids(cli_eos_token)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            return int(tok_id)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        return int(tokenizer.eos_token_id)
    eos_tok = getattr(tokenizer, "eos_token", None) or tokenizer.special_tokens_map.get("eos_token")
    if eos_tok:
        tok_id = tokenizer.convert_tokens_to_ids(eos_tok)
        if tok_id is not None and tok_id != tokenizer.unk_token_id:
            return int(tok_id)
    raise ValueError("Could not resolve eos_token_id. Provide eos_id or eos_token override.")



class DataProcessor:
    """Handles data loading and preprocessing for LLaDA KTO training."""
    
    def __init__(self, tokenizer, caching=True, eos_id=None, eos_token=None):
        self.tokenizer = tokenizer
        self.eos_id = resolve_eos_id(self.tokenizer, cli_eos_id=eos_id, cli_eos_token=eos_token)
        print(f"Using EOS_ID={self.eos_id}")

        if not caching:
            datasets.disable_caching()
    
    def build_example(self, example):
        """Convert a single KTO example to the required format for training."""
        # KTO data typically has 'prompt', 'completion', 'label' fields
        prompt = example['prompt'] # prompt might be a list of dicts
        completion = example['completion'][0]['content'] # completion list will only be a single dict
        label = 1 if example['label'] else 0

        prompt_text = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        
        text = f"{prompt_text}{completion}"
        
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
        
        text_len = len(tokens)
        completion_len = max(min(MAX_LENGTH, text_len) - prompt_len, 0)
        tokens = tokens[:MAX_LENGTH] + [self.eos_id] * (MAX_LENGTH - text_len)
        
        return {
            "orig_prompt": example['prompt'],
            "orig_completion": example['completion'],
            "orig_label": example['label'],
            "prompt": prompt_text,
            "input_ids": tokens,
            "prompt_length": prompt_len,
            "completion": completion,
            "label": label,
            "text_length": text_len,
            "completion_length": completion_len,
        }
    
    def load_dataset(self, data_path=DATA_PATH, split="train"):
        dataset = load_dataset(data_path, split=split).map(self.build_example, num_proc=16)

        return dataset



class DataProcessorPreprocessed:
    """Handles data loading for preprocessed LLaDA KTO training data with precomputed B_ref values."""
    
    def __init__(self, tokenizer, caching=True, eos_id=None, eos_token=None):
        self.tokenizer = tokenizer
        self.eos_id = resolve_eos_id(self.tokenizer, cli_eos_id=eos_id, cli_eos_token=eos_token)
        print(f"Using EOS_ID={self.eos_id}")

        if not caching:
            datasets.disable_caching()
    
    def build_example(self, example):
        """
        Convert a preprocessed KTO example to the required format for training.
        
        This processes the data EXACTLY the same way as DataProcessor.build_example(),
        but starts with the preprocessed data format that includes precomputed B_ref values.
        
        Expected input format (from precompute_bref.py output):
        - "prompt": original prompt (list of dicts)
        - "completion": original completion  
        - "label": original label
        - "seed": precomputed base seed
        - "l_values": precomputed l values for each draw
        - "B_ref": precomputed B_ref values (dict mapping K -> value)
        - "masked_idx_sums": precomputed masked index sums
        """
        # Process prompt and completion exactly like DataProcessor
        prompt = example['prompt']  # prompt might be a list of dicts
        completion = example['completion'][0]['content']  # completion list will only be a single dict
        label = 1 if example['label'] else 0

        prompt_text = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        
        text = f"{prompt_text}{completion}"
        
        tokens = self.tokenizer(text, add_special_tokens=False).input_ids
        prompt_len = len(self.tokenizer(prompt_text, add_special_tokens=False).input_ids)
        
        text_len = len(tokens)
        completion_len = max(min(MAX_LENGTH, text_len) - prompt_len, 0)
        tokens = tokens[:MAX_LENGTH] + [self.eos_id] * (MAX_LENGTH - text_len)
        
        # Combine tokenization results with precomputed fields
        result = {
            "input_ids": tokens,
            "prompt_length": prompt_len,
            "completion_length": completion_len,
            "labels": label,  # Note: using "labels" to match trainer expectations
            # Precomputed fields from B_ref computation
            "seed": str(example["seed"]),  # Store as string to avoid PyArrow overflow with large 64-bit ints
            "l_values": example["l_values"],
            "masked_idx_sums": example["masked_idx_sums"],
            "B_ref": example["B_ref"],
            # Optional fields for debugging/reference
            "prompt": prompt_text,
            "completion": completion,
            "text_length": text_len,
        }
        
        return result
    
    def load_dataset(self, data_path=DATA_PATH):
        """Load preprocessed dataset from JSON file."""
        # features = Features({"seed": Value("string")})
        dataset = load_dataset(
            "json",
            data_files=data_path,
            # features=features
        )
        
        # Apply build_example to process data consistently with DataProcessor
        processed_dataset = dataset.map(self.build_example, num_proc=16)
        
        return processed_dataset 