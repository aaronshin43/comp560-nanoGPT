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

    Returns a list of {'input': str, 'output': str} dicts.
    Raises FileNotFoundError if the file doesn't exist.
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
            # Without targets, nanoGPT only computes the last position → (1, 1, vocab_size).
            logits, _ = model(x, y_true)  # logits: (1, T-1, vocab_size)
        pred = logits[0].argmax(dim=-1)   # (T-1,)

        # The output region in y_true starts at index (len(prefix_ids) - 1).
        # Explanation:
        #   seq      = [i0..iN, sep, o0..oM, stop]
        #   y_true   = seq[1:]  =>  y_true[len(prefix_ids)-1] == seq[len(prefix_ids)] == o0
        out_start = len(prefix_ids) - 1
        if out_start >= len(pred):
            total -= 1
            continue

        if torch.equal(pred[out_start:], y_true[0, out_start:]):
            exact_matches += 1

    model.train()
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

    model.train()
    accuracy = 100.0 * exact_matches / total if total > 0 else 0.0
    return accuracy, exact_matches, total

