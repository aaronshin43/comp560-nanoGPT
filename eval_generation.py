"""
Standalone autoregressive generation evaluation script.

Loads a trained nanoGPT checkpoint and runs exact-match generation eval
on one or both JSONL splits, using comp560ext.run_final_gen_eval.

Usage:
    python eval_generation.py \
        --out_dir=out-my-run \
        --dataset=shakespeare_char \
        --benchmark_target=val

    # Evaluate both splits, limit to 200 samples each:
    python eval_generation.py --out_dir=out-add --dataset=addition \
        --benchmark_target=both --eval_max_samples=200

    # Adjust generation settings:
    python eval_generation.py --out_dir=out-add --dataset=addition \
        --temperature=0.8 --top_k=10 --max_new_tokens=32

Required files (relative to the project root):
    data/<dataset>/train.jsonl  and/or  val.jsonl   (one JSON object per line)
    data/<dataset>/meta.pkl                          (must contain 'stoi' and 'itos')
    <out_dir>/ckpt.pt                                (saved by train.py / train_benchmark.py)
"""

import os
import pickle
from contextlib import nullcontext

import torch

from model import GPTConfig, GPT
from comp560 import comp560ext

# -----------------------------------------------------------------------------
# Configuration — all values can be overridden from the command line, e.g.:
#   python eval_generation.py --out_dir=out-my-run --benchmark_target=both
# -----------------------------------------------------------------------------
out_dir          = 'out'         # directory that contains ckpt.pt
dataset          = 'shakespeare_char'
device           = 'cuda'        # 'cpu', 'cuda', 'cuda:0', …; auto-detects if 'cuda' unavailable
dtype            = 'bfloat16'    # 'float32', 'bfloat16', or 'float16'
compile          = False         # set True to torch.compile the model before eval

# Eval targets
benchmark_target = 'val'         # 'train' | 'val' | 'both'
eval_max_samples = 0             # 0 = evaluate the entire split

# Generation settings
separator_token  = '='           # string separating input from output in each sequence
stop_token       = '\n'          # string that marks the end of a generated answer
max_new_tokens   = 64            # upper bound on generated tokens per sample
temperature      = 1.0           # sampling temperature; 1.0 ≈ greedy
top_k            = None          # top-k filtering; None = disabled
# -----------------------------------------------------------------------------

config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
comp560ext.configure(globals())  # apply --key=value CLI overrides

# ── Device setup ──────────────────────────────────────────────────────────────
if device == 'cuda' and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU.")
    device = 'cpu'

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = (nullcontext() if device_type == 'cpu'
       else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

# ── Load checkpoint ───────────────────────────────────────────────────────────
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device)

model_args = checkpoint['model_args']
gptconf    = GPTConfig(**model_args)
model      = GPT(gptconf)

state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)

if compile:
    print("Compiling model…")
    model = torch.compile(model)

ckpt_iter = checkpoint.get('iter_num', '?')
print(f"Model loaded  (checkpoint iter={ckpt_iter}, "
      f"n_layer={model_args['n_layer']}, n_head={model_args['n_head']}, "
      f"n_embd={model_args['n_embd']})")

# ── Load tokenizer ─────────────────────────────────────────────────────────────
data_dir  = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')

if not os.path.exists(meta_path):
    raise FileNotFoundError(
        f"meta.pkl not found at {meta_path}. "
        "Make sure 'dataset' points to a character-level dataset whose prepare.py "
        "produces meta.pkl with 'stoi' and 'itos'."
    )

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)

if 'stoi' not in meta or 'itos' not in meta:
    raise KeyError("meta.pkl must contain 'stoi' and 'itos' mappings.")

# Build encode/decode inline rather than via comp560ext.setup_char_encode_decode().
# Unlike the training script (which soft-fails and disables eval), this script
# cannot proceed without a working tokenizer, so hard failures are correct here.
stoi   = meta['stoi']
itos   = meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
print(f"Tokenizer ready  (vocab_size={meta['vocab_size']})")

# ── Run generation eval ───────────────────────────────────────────────────────
_splits     = ['train', 'val'] if benchmark_target == 'both' else [benchmark_target]
_max        = eval_max_samples if eval_max_samples > 0 else None
_block_size = model_args['block_size']
_gen_tokens = min(max_new_tokens, _block_size)

print()
for split in _splits:
    jsonl_path = os.path.join(data_dir, f'{split}.jsonl')
    print(f"── [{split}] {jsonl_path} ──")

    try:
        acc, em, total = comp560ext.run_final_gen_eval(
            model, jsonl_path, encode, decode,
            separator_str  = separator_token,
            stop_token_str = stop_token,
            max_new_tokens = _gen_tokens,
            temperature    = temperature,
            top_k          = top_k,
            device         = device,
            ctx            = ctx,
            max_samples    = _max,
        )
        print(f"  Exact-match: {acc:.1f}%  ({em}/{total} correct)\n")
    except FileNotFoundError as e:
        print(f"  Skipped — {e}\n")
