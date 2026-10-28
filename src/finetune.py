"""LoRA fine-tune of the intent step on the retrieval prompt.

Holds out 500 rows (random_state=0) as validation. Trains on the other rows.
Each prompt has the 5 most similar other training rows (TF-IDF cosine).
Saves the adapter to models/intent-lora-retrieval/.

Run from the project root: uv run -m src.finetune
"""

import os

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER = "models/intent-lora-retrieval"
VAL_N = 500
K = 5
# Same sentence as src/prompts.py CONSTRAINT. Do not import prompts: that loads the model.
CONSTRAINT = "Reply with the exact intent name from the list and nothing else. Always choose exactly one intent."


def task_text(categories: list[str]) -> str:
    """Same sentence as src/baseline.py. Do not import baseline: that loads the model."""
    names = ", ".join(sorted(set(categories)))
    return "Classify the customer message into one of these intents:\n" + names


def user_text(task: str, examples: list[tuple[str, str]], message: str) -> str:
    shots = "".join(f"Message: {text}\nIntent: {label}\n\n" for text, label in examples)
    return f"{task}\n{CONSTRAINT}\n\n{shots}Message: {message}\nIntent:"


def top_indexes(scores: np.ndarray, k: int) -> np.ndarray:
    """For each row, the k highest-scoring columns, best first."""
    pick = np.argpartition(scores, -k, axis=1)[:, -k:]
    row = np.arange(scores.shape[0])[:, None]
    order = np.argsort(scores[row, pick], axis=1)[:, ::-1]
    return pick[row, order]


def examples_from(pool: pd.DataFrame, indexes: np.ndarray) -> list[list[tuple[str, str]]]:
    texts = pool["text"].to_numpy()
    labels = pool["category"].to_numpy()
    return [list(zip(texts[row], labels[row])) for row in indexes]


def token_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    if hasattr(out, "input_ids"):
        out = out["input_ids"]
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


def encode(tokenizer, task: str, examples: list[tuple[str, str]], message: str, label: str) -> dict:
    user = user_text(task, examples, message)
    prompt = token_ids(tokenizer, [{"role": "user", "content": user}], True)
    full = token_ids(
        tokenizer,
        [{"role": "user", "content": user}, {"role": "assistant", "content": label}],
        False,
    )
    if full[: len(prompt)] != prompt:
        raise RuntimeError("prompt tokens are not a prefix of the full turn")
    labels = [-100] * len(prompt) + full[len(prompt) :]
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}


class Rows(torch.utils.data.Dataset):
    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def main():
    data = pd.read_csv("data/train.csv")
    val = data.sample(n=VAL_N, random_state=0)
    fit = data.drop(val.index)
    if not set(fit.index).isdisjoint(val.index):
        raise RuntimeError("train and validation rows overlap")
    fit = fit.reset_index(drop=True)
    val = val.reset_index(drop=True)
    fit.to_csv("data/finetune_train.csv", index=False)
    val.to_csv("data/finetune_val.csv", index=False)
    task = task_text(data["category"])
    print(f"train {len(fit)} rows, validation {len(val)} rows")

    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    fit_vectors = vectorizer.fit_transform(fit["text"])
    fit_scores = cosine_similarity(fit_vectors)
    np.fill_diagonal(fit_scores, -np.inf)
    fit_idx = top_indexes(fit_scores, K)
    if np.any(fit_idx == np.arange(len(fit))[:, None]):
        raise RuntimeError("a training row was retrieved as its own example")
    fit_examples = examples_from(fit, fit_idx)
    val_scores = cosine_similarity(vectorizer.transform(val["text"]), fit_vectors)
    val_examples = examples_from(fit, top_indexes(val_scores, K))

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_rows = Rows([encode(tokenizer, task, shots, text, label) for shots, text, label in zip(fit_examples, fit["text"], fit["category"])])
    val_rows = Rows([encode(tokenizer, task, shots, text, label) for shots, text, label in zip(val_examples, val["text"], val["category"])])
    print(f"tokens in first train row: {len(train_rows[0]['input_ids'])}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=ADAPTER,
            num_train_epochs=3,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            logging_steps=20,
            report_to="none",
            dataloader_num_workers=0,
            seed=0,
        ),
        train_dataset=train_rows,
        eval_dataset=val_rows,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    trainer.save_model(ADAPTER)
    tokenizer.save_pretrained(ADAPTER)
    print(f"adapter saved to {ADAPTER}")


if __name__ == "__main__":
    main()
