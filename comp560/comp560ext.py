"""
Extensions and utilities for the comp560 package in the nanoGPT project.

This module provides additional functionality to support custom configurations
and extensions without modifying core scripts like train.py, sample.py.
"""

import os
import torch
import json
import pickle
import tiktoken
from torch.nn import functional as F

config = {} # Should be overwritten by train.py or sample.py

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

def get_encoder_decoder(data_dir):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
        decode = lambda l: enc.decode(l)
    return encode, decode

def evaluate_accuracy(model, data_dir, device, config, max_samples=50):
    val_data_path = os.path.join(data_dir, 'val.jsonl')
    if not os.path.exists(val_data_path):
        return None

    encode, decode = get_encoder_decoder(data_dir)
    
    # Determine separator and stop_token from config
    separator = config.get('separator', '=')  # Add separator if defined (e.g. "=")
    # Stop token is crucial for generation to stop cleanly
    stop_token_val = config.get('stop_token', "\n")
    stop_token_id = prepare_stop_token(stop_token_val, encode)

    num_correct = 0
    num_total = 0

    with open(val_data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Simple sampling if file is large, or just take first N
    import random
    if len(lines) > max_samples:
        lines = random.sample(lines, max_samples)

    model.eval()
    with torch.no_grad():
        for line in lines:
            try:
                example = json.loads(line)
                prompt = example['input'] + separator
                target = example['output']
            except (json.JSONDecodeError, KeyError):
                continue
            
            start_ids = encode(prompt)
            x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            
            # Generate
            y = generate(model, x, max_new_tokens=20, temperature=1.0, top_k=1, stop_token=stop_token_id)
            
            # Extract generated part
            # y contains [prompt + generated]
            # we need to decode only the generated part
            generated_ids = y[0].tolist()[len(start_ids):]
            generated_text = decode(generated_ids)
            
            # Strip stop token from generated text if it exists
            if stop_token_val and generated_text.endswith(stop_token_val):
                generated_text = generated_text[:-len(stop_token_val)]
            
            # Compare
            if generated_text.strip() == target.strip():
                num_correct += 1
            num_total += 1
            
    model.train() # Switch back to train mode
    
    return num_correct / num_total if num_total > 0 else 0.0


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

