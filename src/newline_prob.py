"""
Compute newline prediction log-prob and accuracy across line positions for k=50 text.
Replicates the paper figure showing model adaptation to line width.
"""

import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

MODEL_ID = "google/gemma-3-1b-pt"
DATA_PATH = "data/synthetick-50.txt"


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    return model, tokenizer


def parse_examples(path: str) -> list[dict]:
    """
    Parse examples file. Returns list of dicts with:
    - 'lines': list of text lines
    - 'label': 'NEWLINE' or 'CONTINUE' (what's expected at end of last line)
    """
    with open(path) as f:
        content = f.read()
    
    examples = []
    # Match header line ending with === and following content
    pattern = r"(=== EXAMPLE (\d+)[^\n]* ===)\n(.*?)(?=\n=== EXAMPLE|\Z)"
    
    for match in re.finditer(pattern, content, re.DOTALL):
        header = match.group(1)
        ex_num = int(match.group(2))
        text = match.group(3).strip()
        
        lines = [l for l in text.split("\n") if l.strip()]
        if not lines:
            continue
        
        # Determine label from header
        if "NEWLINE EXPECTED" in header:
            label = "NEWLINE"
        elif "CONTINUE EXPECTED" in header:
            label = "CONTINUE"
        elif "BORDERLINE" in header:
            # Parse: "BORDERLINE (42 chars, word "testing" = 7)"
            # If chars + 1 + word_len > 50 → NEWLINE, else CONTINUE
            m = re.search(r"\((\d+) chars.*?= (\d+)\)", header)
            if m:
                chars, word_len = int(m.group(1)), int(m.group(2))
                label = "NEWLINE" if chars + 1 + word_len > 50 else "CONTINUE"
            else:
                label = "NEWLINE"
        else:
            # Example 1: "was called" + "aluminum" fits → CONTINUE
            label = "CONTINUE"
        
        examples.append({"lines": lines, "label": label, "num": ex_num})
    
    examples.sort(key=lambda x: x["num"])
    return examples


def get_newline_logprob(model, tokenizer, prompt: str) -> tuple[float, int]:
    """Return (log_prob of newline, top_token_id)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    log_probs = torch.log_softmax(logits, dim=-1)
    
    return log_probs[newline_id].item(), logits.argmax().item()


def compute_metrics(model, tokenizer, examples: list[dict], max_lines: int = 17):
    """
    Compute metrics per line position.
    
    At each internal line boundary (end of lines 0..N-2), newline is correct.
    At final line boundary, use the example's label.
    """
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    
    log_probs = defaultdict(list)
    correct = defaultdict(list)
    
    for ex_idx, ex in enumerate(examples):
        lines = ex["lines"]
        label = ex["label"]
        print(f"Example {ex_idx + 1}/{len(examples)}: {len(lines)} lines, {label}")
        
        num_lines = min(len(lines), max_lines)
        
        for line_idx in range(num_lines):
            # Build prompt up to end of this line
            prompt = "\n".join(lines[:line_idx + 1])
            
            lp, top_id = get_newline_logprob(model, tokenizer, prompt)
            
            # Determine if newline is correct at this position
            if line_idx < num_lines - 1:
                # Internal boundary: newline is always correct
                is_correct = (top_id == newline_id)
            else:
                # Final line: use label
                if label == "NEWLINE":
                    is_correct = (top_id == newline_id)
                else:
                    is_correct = (top_id != newline_id)
            
            log_probs[line_idx].append(lp)
            correct[line_idx].append(int(is_correct))
    
    return log_probs, correct


def plot_results(log_probs: dict, correct: dict, output_path: str = "figures/newline_prob_k50.png"):
    """Create dual-panel plot matching paper figure style."""
    line_nums = sorted(log_probs.keys())
    
    mean_lp = [np.mean(log_probs[n]) for n in line_nums]
    mean_acc = [np.mean(correct[n]) for n in line_nums]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Mean Log Probability of Newline
    ax1.plot(line_nums, mean_lp, "o-", color="darkblue", linewidth=2, markersize=6)
    ax1.set_xlabel("Line Number", fontsize=12)
    ax1.set_ylabel("Mean Log Probability of Newline", fontsize=12)
    ax1.set_title("k = 50", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, max(line_nums) + 0.5)
    
    # Right: Top-1 Accuracy
    ax2.plot(line_nums, mean_acc, "o-", color="darkblue", linewidth=2, markersize=6)
    ax2.set_xlabel("Line Number", fontsize=12)
    ax2.set_ylabel("Top-1 Accuracy", fontsize=12)
    ax2.set_title("k = 50", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.5, max(line_nums) + 0.5)
    ax2.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def main():
    print("Loading model...")
    model, tokenizer = load_model()
    
    print("Parsing examples...")
    examples = parse_examples(DATA_PATH)
    print(f"Found {len(examples)} examples")
    
    print("\nComputing metrics...")
    log_probs, correct = compute_metrics(model, tokenizer, examples)
    
    print("\n=== Results ===")
    for n in sorted(log_probs.keys()):
        print(f"Line {n:2d}: log_prob={np.mean(log_probs[n]):6.3f}, acc={np.mean(correct[n]):.3f} (n={len(log_probs[n])})")
    
    plot_results(log_probs, correct)


if __name__ == "__main__":
    main()
