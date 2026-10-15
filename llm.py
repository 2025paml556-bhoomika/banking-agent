"""Hugging Face causal LM used for all prompt and decoding experiments."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto").to(DEVICE)


def generate(messages: list[dict], max_new_tokens: int = 64, **decoding) -> str:
    """Run one chat turn. `decoding` is passed to `model.generate`, for example
    do_sample=True, temperature=0.7, top_k=50, top_p=0.9. Default is greedy."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    decoding = {"do_sample": False, **decoding}
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, **decoding)
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


if __name__ == "__main__":
    print(generate([{"role": "user", "content": "I lost my card. What should I do?"}]))
