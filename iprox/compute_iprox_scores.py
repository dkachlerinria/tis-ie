import os
import sys
import torch
import torch.nn as nn
import numpy as np
import json
import sqlite3
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, get_dataset_config_names

# Add repo root and iprox folder to path
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)
sys.path.append(script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)

from common.data import encode_with_messages_format
from iprox.utils.init_with_ipsvd import init_proxy_model_with_IPSVD, load_proxy_model

# ==================== HELPERS FROM GRADIENT_STOCKING ====================

def ensure_chat_template(tokenizer):
    _CHATML_TEMPLATE = (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        tokenizer.chat_template = _CHATML_TEMPLATE
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_bbh_data(data_dir, n_samples=None, start_index=0):
    import glob
    bbh_dir = os.path.join(data_dir, "bbh")
    prompt_dir = os.path.join(data_dir, "cot-prompts")
    if not os.path.exists(bbh_dir) or not os.path.exists(prompt_dir):
        bbh_dir = os.path.join(data_dir, "bbh")
        prompt_dir = os.path.join(data_dir, "bbh/cot-prompts")

    all_tasks = {}
    task_files = glob.glob(os.path.join(bbh_dir, "*.json"))
    for task_file in task_files:
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]

    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(prompt_dir, "*.txt"))
    for cot_prompt_file in cot_prompt_files:
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            all_prompts[task_name] = "".join(f.readlines()[2:])

    processed = []
    for task_name, examples in all_tasks.items():
        task_prompt = all_prompts.get(task_name, "").strip()
        for ex in examples:
            processed.append({
                "prompt": task_prompt + "\n\nQ: " + ex["input"],
                "response": ex["target"],
                "doc_id": f"bbh_{task_name}_{ex.get('id', len(processed))}"
            })
    processed = sorted(processed, key=lambda x: x["doc_id"])
    if n_samples:
        processed = processed[start_index : start_index + n_samples]
    return processed

def load_data_split(dataset_name, tokenizer, n_samples=None, start_index=0):
    if "tulu" in dataset_name.lower():
        ds = load_dataset(dataset_name, split="train")
        np.random.seed(42)
        index = np.arange(len(ds))
        np.random.shuffle(index)
        target_indices = index[start_index : start_index + n_samples] if n_samples else index[start_index:]
        processed = []
        for i, idx in enumerate(target_indices):
            item = ds[int(idx)]
            if "messages" in item and len(item["messages"]) >= 2:
                msgs = item["messages"]
                if msgs[-1]["role"] == "assistant":
                    processed.append({"prompt": msgs[:-1], "response": msgs[-1]["content"]})
        return processed
    elif dataset_name == "bbh":
        eval_data_dir = os.environ.get("EVAL_DATA_DIR", "data/eval")
        if not os.path.exists(os.path.join(eval_data_dir, "bbh")):
             if os.path.exists("data/bbh"):
                 eval_data_dir = "data"
        return load_bbh_data(eval_data_dir, n_samples=n_samples, start_index=start_index)
    return []

# ==================== IPROX SCORING LOGIC ====================

def compute_proxy_gradient(model, prompt, response, tokenizer, max_seq_length, device, target_modules):
    if isinstance(prompt, list):
        messages = prompt + [{"role": "assistant", "content": response}]
    else:
        messages = [
            {"role": "user", "content": str(prompt).strip()},
            {"role": "assistant", "content": str(response).strip()}
        ]
    
    out = encode_with_messages_format(
        {"messages": messages}, 
        tokenizer, 
        max_seq_length=max_seq_length, 
        include_response=True
    )
    
    input_ids = out["input_ids"].unsqueeze(0).to(device)
    attention_mask = out["attention_mask"].unsqueeze(0).to(device)
    labels = out["labels"].unsqueeze(0).to(device)
    
    model.zero_grad(set_to_none=True)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    
    sample_grads = []
    for name, module in model.named_modules():
        if any(name.endswith(tm) for tm in target_modules):
            if hasattr(module, 'linear_A') and hasattr(module, 'linear_B'):
                if module.linear_A.grad is not None and module.linear_B.grad is not None:
                    grad_A = module.linear_A.grad.detach().cpu().float().flatten().numpy()
                    grad_B = module.linear_B.grad.detach().cpu().float().flatten().numpy()
                    sample_grads.append(np.concatenate([grad_A, grad_B]))
    
    if not sample_grads:
        return None
    return np.concatenate(sample_grads)

def safe_normalize(grad, eps=1e-8):
    norm = np.linalg.norm(grad)
    return grad / max(norm, eps)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy_path", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="bbh")
    parser.add_argument("--train_dataset_name", type=str, default="tulu")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--target_modules", nargs="+", default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    # Load Model & Tokenizer
    print(f"🤖 Loading Proxy Model: {args.proxy_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.proxy_path)
    ensure_chat_template(tokenizer)
    
    # We need a base model to initialize the structure for IPSVD
    # For now, we assume the base model name is in the proxy_path's config
    base_model = AutoModelForCausalLM.from_pretrained(
        args.proxy_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )

    # Re-initialize the LinearSVD structure
    proxy_model = init_proxy_model_with_IPSVD(
        base_model=base_model,
        loader_src=None, # Not needed for loading weights
        sparsity=args.sparsity,
        init_method="RANDOM",
        target_modules=args.target_modules,
        min_rank_multiple=1
    )
    
    # Load the trained weights (the .bin file)
    final_bin = os.path.join(args.proxy_path, "pytorch_model.bin")
    if os.path.exists(final_bin):
        load_proxy_model(proxy_model, final_bin)
    else:
        # Fallback to search in parent or adjacent locations if needed
        parent_final = os.path.join(os.path.dirname(args.proxy_path), "final_pytorch_model.bin")
        if os.path.exists(parent_final):
            load_proxy_model(proxy_model, parent_final)
        else:
            raise FileNotFoundError(f"Could not find proxy weights in {args.proxy_path} or parent.")

    proxy_model.eval()
    for p in proxy_model.parameters():
        p.requires_grad_(True)

    # Load Data
    print(f"📂 Loading Benchmark Anchors: {args.benchmark}")
    anchors = load_data_split(args.benchmark, tokenizer, n_samples=200) # Same size as eval anchors
    
    print(f"📂 Loading Training Pool: {args.train_dataset_name}")
    pool = load_data_split(args.train_dataset_name, tokenizer)

    # Compute Anchor Gradients
    print("🧮 Computing Anchor Gradients...")
    anchor_grads = []
    for a in tqdm(anchors):
        g = compute_proxy_gradient(proxy_model, a["prompt"], a["response"], tokenizer, args.max_seq_length, device, args.target_modules)
        if g is not None:
            anchor_grads.append(safe_normalize(g))
    
    anchor_matrix = np.stack(anchor_grads) # [n_anchors, dim]
    
    # Compute Pool Scores (Streaming)
    print("🧮 Computing Pool Scores...")
    scores = []
    for p in tqdm(pool):
        g = compute_proxy_gradient(proxy_model, p["prompt"], p["response"], tokenizer, args.max_seq_length, device, args.target_modules)
        if g is not None:
            g_norm = safe_normalize(g)
            s = anchor_matrix @ g_norm # [n_anchors]
            scores.append(s.mean()) # Average influence over anchors
        else:
            scores.append(0.0)

    scores_pt = torch.tensor(scores)
    torch.save(scores_pt, os.path.join(args.output_dir, "scores.pt"))
    print(f"✅ Scores saved to {os.path.join(args.output_dir, 'scores.pt')}")

if __name__ == "__main__":
    main()
