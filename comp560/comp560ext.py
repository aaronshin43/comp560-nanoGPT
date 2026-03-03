"""
Extensions and utilities for the comp560 package in the nanoGPT project.

This module provides additional functionality to support custom configurations
and extensions without modifying core scripts like train.py, sample.py.
"""

import json
import os
import sys
from ast import literal_eval
import torch
from torch.nn import functional as F

config = {} # Should be overwritten by train.py or sample.py

def configure(globals_dict):
    """
    Parses command line arguments and updates the globals dictionary.
    Replaces the functionality of configurator.py with improved parsing logic.
    """
    for arg in sys.argv[1:]:
        if '=' not in arg:
            # assume it's the name of a config file
            assert not arg.startswith('--')
            config_file = arg
            print(f"Overriding config with {config_file}:")
            with open(config_file) as f:
                print(f.read())
            exec(open(config_file).read(), globals_dict)
        else:
            # assume it's a --key=value argument
            assert arg.startswith('--')
            # FIX: Use maxsplit=1 to handle values containing '=' (e.g. --start="1+1=")
            key, val = arg.split('=', 1)
            key = key[2:]
            if key in globals_dict:
                try:
                    # attempt to eval it it (e.g. if bool, number, or etc)
                    attempt = literal_eval(val)
                except (SyntaxError, ValueError):
                    # if that goes wrong, just use the string
                    attempt = val
                # ensure the types match ok
                assert type(attempt) == type(globals_dict[key])
                # cross fingers
                print(f"Overriding: {key} = {attempt}")
                globals_dict[key] = attempt
            else:
                raise ValueError(f"Unknown config key: {key}")


def get_config_file():
    return os.environ.get("NANOGPT_CONFIG", "configurator.py")


def print_config():
    print(f'comp560ext.config:\n{config}\n-----------------')

def calc_flops_achieved(flops_per_iter, dt):
    return flops_per_iter * (1.0/dt) if dt > 0 else 0.0  # per second

def prepare_stop_token(stop_token, encode):
    """
    Encodes the stop_token string into a token ID using the provided encode function.
    Returns the first token ID or None.
    """
    if stop_token and stop_token != "":
        stop_ids = encode(stop_token)
        if len(stop_ids) > 0:
            return stop_ids[0]
    return None

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    
    Args:
        stop_token (int, optional): If provided, generation stops when this token is generated.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        # append sampled index to the running sequence and continue
        idx = torch.cat((idx, idx_next), dim=1)
        
        # stop response when stop_token is generated
        if stop_token is not None and idx_next.item() == stop_token:
            break

    return idx


# ---------------------------------------------------------------------------
# Benchmarking / Evaluation utilities
# ---------------------------------------------------------------------------

def load_eval_dataset(jsonl_path, max_samples=None):
    """
    Load a JSONL evaluation dataset where each line is a JSON object with
    'input' and 'output' string keys.

    Args:
        jsonl_path:  Path to the .jsonl file.
        max_samples: If given (positive int), only the first `max_samples` valid
                     lines are returned.  None (default) loads the entire file.

    Returns:
        List of {'input': str, 'output': str} dicts.

    Raises:
        FileNotFoundError: If the file does not exist at `jsonl_path`.
        ValueError: If a line contains invalid JSON, or is missing the
                    'input' or 'output' key.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"Eval dataset not found: {jsonl_path}\n"
            "Please ensure train.jsonl / val.jsonl exist in the dataset directory."
        )
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON parse error on line {lineno} of {jsonl_path}: {e}")
            if 'input' not in obj or 'output' not in obj:
                raise ValueError(
                    f"Line {lineno} in {jsonl_path} is missing 'input' or 'output' key."
                )
            samples.append({'input': obj['input'], 'output': obj['output']})
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


@torch.no_grad()
def run_tf_eval(model, eval_data, encode, separator_str, stop_token_str, device, ctx):
    """
    Fast Teacher Forcing (TF) Exact Match Evaluation.

    For each sample the full sequence  (input + separator + output + stop)  is
    fed through the model in a **single forward pass**.  The argmax predictions
    are compared against the target tokens only in the *output region* — i.e.
    the tokens that follow the separator up to and including the stop token.
    A sample counts as an exact match only when every output token is predicted
    correctly.

    Args:
        model:          The raw (non-DDP) model in train/eval compatible state.
        eval_data:      List of {'input': str, 'output': str} dicts.
        encode:         Callable str -> list[int].
        separator_str:  String separating input from output (e.g. '=').
        stop_token_str: String marking end of output (e.g. '\\n').
        device:         torch device string.
        ctx:            Autocast context (nullcontext or torch.amp.autocast).

    Returns:
        (accuracy_pct, exact_matches, total)  where accuracy_pct is 0–100.
    """
    was_training = model.training
    model.eval()
    exact_matches = 0
    total = len(eval_data)

    for sample in eval_data:
        prefix_str = sample['input'] + separator_str
        full_str   = prefix_str + sample['output'] + stop_token_str

        prefix_ids = encode(prefix_str)
        full_ids   = encode(full_str)

        # Need at least 2 tokens to form an (x, y) pair.
        if len(full_ids) < 2:
            total -= 1
            continue

        x      = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)  # (1, T-1)
        y_true = torch.tensor([full_ids[1:]],  dtype=torch.long, device=device)  # (1, T-1)

        with ctx:
            # Pass targets so nanoGPT returns full-sequence logits (1, T-1, vocab_size).
            # Without targets, nanoGPT only returns the last position → (1, 1, vocab_size).
            # The loss returned as the second value is discarded (_); only logits are needed.
            logits, _ = model(x, y_true)  # logits: (1, T-1, vocab_size)
        pred = logits[0].argmax(dim=-1)   # (T-1,)

        # The output region in y_true starts at index (len(prefix_ids) - 1).
        # Derivation (using L = len(prefix_ids)):
        #   full_ids = [p0..p_{L-1}, o0..o_M, stop]
        #   y_true   = full_ids[1:]   =>   y_true[L-1] == full_ids[L] == o0
        out_start = len(prefix_ids) - 1
        if out_start >= len(pred):   # output region is empty; skip
            total -= 1
            continue

        if torch.equal(pred[out_start:], y_true[0, out_start:]):
            exact_matches += 1

    model.train(was_training)   # restore original train/eval state
    accuracy = 100.0 * exact_matches / total if total > 0 else 0.0
    return accuracy, exact_matches, total


@torch.no_grad()
def run_final_gen_eval(model, jsonl_path, encode, decode, separator_str, stop_token_str,
                       max_new_tokens, temperature, top_k, device, ctx,
                       max_samples=None):
    """
    Final autoregressive generation evaluation.

    For each sample, only  (input + separator)  is fed as the prompt to
    `generate`.  Generation stops when `stop_token` is produced (or
    `max_new_tokens` is exhausted).  The decoded continuation is compared
    against the expected output using exact string match.

    Args:
        model:          The raw (non-DDP) model.
        jsonl_path:     Path to the .jsonl eval file.
        encode:         Callable str -> list[int].
        decode:         Callable list[int] -> str.
        separator_str:  String appended to the input before feeding to generate.
        stop_token_str: String whose first encoded token ID terminates generation.
        max_new_tokens: Maximum number of tokens to generate per sample.
        temperature:    Sampling temperature (use 1.0 for greedy-like behaviour).
        top_k:          Top-k filtering (None = disabled).
        device:         torch device string.
        ctx:            Autocast context.
        max_samples:    If given (positive int), evaluate only the first N samples.
                        None (default) evaluates the entire file.

    Returns:
        (accuracy_pct, exact_matches, total)  where accuracy_pct is 0–100.
    """
    eval_data = load_eval_dataset(jsonl_path, max_samples=max_samples)
    model.eval()

    stop_ids      = encode(stop_token_str)
    stop_token_id = stop_ids[0] if stop_ids else None

    exact_matches = 0
    total = len(eval_data)

    for sample in eval_data:
        prompt_str = sample['input'] + separator_str
        prompt_ids = encode(prompt_str)
        if not prompt_ids:
            total -= 1
            continue

        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with ctx:
            out_tensor = generate(
                model, prompt_tensor, max_new_tokens,
                temperature=temperature, top_k=top_k,
                stop_token=stop_token_id,
            )

        # Slice off only the newly generated tokens (after the prompt).
        gen_ids = out_tensor[0, len(prompt_ids):].tolist()

        # Strip trailing stop token if the model produced it.
        if gen_ids and stop_token_id is not None and gen_ids[-1] == stop_token_id:
            gen_ids = gen_ids[:-1]

        generated_str = decode(gen_ids)

        if generated_str == sample['output']:
            exact_matches += 1

    accuracy = 100.0 * exact_matches / total if total > 0 else 0.0
    return accuracy, exact_matches, total


def apply_target_mask(y: torch.Tensor, sep_id: int, stop_id: int) -> torch.Tensor:
    """
    Replace input-region tokens in y with -1 (nanoGPT's cross-entropy ignore_index).

    Tokens from the start of each sample up to and including the separator are
    masked; output tokens and the stop token keep their original IDs.

    Uses fully-vectorised cumsum logic — single pass over (B, T), no Python loops.

    Masking rule at position t:
        stops_seen_before_t >= seps_seen_before_t  →  in 'input phase' → mask

    Edge-case: a window that begins mid-output (before the first stop) will
    incorrectly mask those leading output tokens until the first stop is seen.
    This is a minor effect for short-sample / long-block-size settings.
    """
    B, T = y.shape
    # Inclusive cumulative counts (how many sep/stop seen up to and including t).
    stop_cs = torch.cumsum((y == stop_id).long(), dim=1)   # (B, T)
    sep_cs  = torch.cumsum((y == sep_id ).long(), dim=1)   # (B, T)
    # Shift right by 1 to get *exclusive* counts (seen *before* position t).
    zeros    = torch.zeros(B, 1, dtype=torch.long, device=y.device)
    stop_bef = torch.cat([zeros, stop_cs[:, :-1]], dim=1)  # (B, T)
    sep_bef  = torch.cat([zeros, sep_cs [:, :-1]], dim=1)  # (B, T)
    # We are in 'input phase' when stops_before >= seps_before.
    in_input = stop_bef >= sep_bef                          # (B, T) bool mask
    y = y.clone()
    y[in_input] = -1   # nanoGPT uses ignore_index=-1 in F.cross_entropy (see model.py:189)
    return y


def setup_char_encode_decode(meta, meta_vocab_size):
    """
    Build character-level encode/decode callables from a loaded meta.pkl dict.

    Args:
        meta:            Dict loaded from meta.pkl, or None if meta.pkl was not found.
        meta_vocab_size: vocab_size from meta, or None if meta was not loaded.

    Returns:
        (encode, decode) callables on success; (None, None) if stoi/itos are missing.
        Prints a warning in the failure case.
    """
    if meta is not None and 'stoi' in meta and 'itos' in meta:
        _stoi  = meta['stoi']
        _itos  = meta['itos']
        encode = lambda s: [_stoi[c] for c in s]
        decode = lambda l: ''.join([_itos[i] for i in l])
        print(f"encode/decode ready for benchmarking (vocab_size={meta_vocab_size})")
        return encode, decode
    print(
        "Warning: meta.pkl not found or missing 'stoi'/'itos' — "
        "benchmarking eval (enable_tf_eval / enable_final_eval) will be skipped."
    )
    return None, None


def resolve_mask_token_ids(meta, meta_vocab_size, separator_token, stop_token):
    """
    Resolve token IDs for the separator and stop tokens from a loaded meta.pkl dict.

    Args:
        meta:            Dict loaded from meta.pkl, or None if meta.pkl was not found.
        meta_vocab_size: vocab_size from meta, or None if meta was not loaded.
        separator_token: String separating input from output (e.g. '=').
        stop_token:      String marking end of output (e.g. '\\n').

    Returns:
        (sep_id, stop_id) ints on success; (None, None) if either token is not in vocab.
        Prints a warning in the failure case.
    """
    stoi      = meta.get('stoi', {}) if meta_vocab_size is not None else {}
    sep_char  = separator_token[0] if separator_token else None
    stop_char = stop_token[0]      if stop_token      else None
    if sep_char in stoi and stop_char in stoi:
        sep_id  = stoi[sep_char]
        stop_id = stoi[stop_char]
        print(f"target_mask=True: masking input tokens up to separator "
              f"(sep='{sep_char}' id={sep_id}, stop='{stop_char}' id={stop_id})")
        return sep_id, stop_id
    print("Warning: target_mask=True but separator/stop token not found in vocab — "
          "target masking disabled.")
    return None, None

