"""
Minimal inference script for linebreaking experiment.
Tests if gemma3-1b-pt can predict newlines at correct positions in k=50 text.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "google/gemma-3-1b-pt"


def get_newline_logit(model, tokenizer, prompt: str) -> tuple[float, str, float]:
    """
    Run inference and return:
    - log probability of newline token
    - predicted next token
    - probability of predicted token
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token position
    
    # Get newline token id
    newline_id = tokenizer.encode("\n", add_special_tokens=False)[0]
    
    # Compute log probs
    log_probs = torch.log_softmax(logits, dim=-1)
    newline_log_prob = log_probs[newline_id].item()
    
    # Get top prediction
    top_id = logits.argmax().item()
    top_token = tokenizer.decode([top_id])
    top_prob = torch.softmax(logits, dim=-1)[top_id].item()
    
    return newline_log_prob, top_token, top_prob


def main():
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"Model loaded on {model.device}")
    
    # Read prompt - aluminum example from paper at k=50
    with open("data/k50_clean.txt", "r") as f:
        text = f.read()
    
    # The prompt ends right before where newline should be predicted
    # Last line is: "produce so cheaply was called"
    # Character count: 30 chars, next word "aluminum" (8) won't fit in 50
    prompt = text.rstrip()  # remove trailing newline for prompt
    
    print("\n" + "="*60)
    print("PROMPT (last 200 chars):")
    print("="*60)
    print(f"...{prompt[-200:]}")
    print("="*60)
    
    # Count chars in last line
    last_line = prompt.split("\n")[-1]
    print(f"\nLast line: '{last_line}'")
    print(f"Last line length: {len(last_line)} chars")
    print(f"Chars remaining for k=50: {50 - len(last_line)}")
    
    # Run inference
    newline_log_prob, top_token, top_prob = get_newline_logit(model, tokenizer, prompt)
    
    print("\n" + "="*60)
    print("INFERENCE RESULTS:")
    print("="*60)
    print(f"Newline log probability: {newline_log_prob:.4f}")
    print(f"Newline probability: {torch.exp(torch.tensor(newline_log_prob)).item():.4f}")
    print(f"Top predicted token: '{repr(top_token)}'")
    print(f"Top token probability: {top_prob:.4f}")


if __name__ == "__main__":
    main()
