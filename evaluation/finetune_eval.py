"""Score the retrieval-prompt LoRA adapter. Development subset first, then the held-out subset.

The prompt is the same one used in training (src/retrieval.py). Decoding is greedy.
Writes new result files. The baseline-prompt scores stay in results_finetune_dev.csv
and results_finetune_heldout.csv.

Run from the project root: uv run -m evaluation.finetune_eval
"""

import os

os.environ["ADAPTER"] = "models/intent-lora-retrieval"

import pandas as pd

from src.baseline import parse_intent
from src.llm import generate
from src.retrieval import build_prompt

# Saved 1.5B scores from the report: retrieval on the development subset,
# final system on the held-out subset.
RETRIEVAL = {"development": 0.84, "held-out": 0.77}
SETS = [
    ("development", "evaluation/eval_subset.csv", "evaluation/results/results_finetune_retrieval_dev.csv"),
    ("held-out", "evaluation/heldout_subset.csv", "evaluation/results/results_finetune_retrieval_heldout.csv"),
]


def score(name: str, source: str, dest: str) -> float:
    data = pd.read_csv(source)
    data["output"] = [generate(build_prompt(text), max_new_tokens=32) for text in data["text"]]
    data["predicted"] = [parse_intent(output) for output in data["output"]]
    data["correct"] = data["predicted"] == data["category"]
    data.to_csv(dest, index=False)
    accuracy = float(data["correct"].mean())
    print(f"{name} accuracy (LoRA): {accuracy:.2f}  retrieval: {RETRIEVAL[name]:.2f}")
    print(f"no valid intent: {(data['predicted'] == '').sum()}")
    return accuracy


def main():
    scores = [(name, score(name, source, dest)) for name, source, dest in SETS]
    lines = [
        "# Fine-tune scores",
        "",
        "Adapter: `models/intent-lora-retrieval`. Prompt: the retrieval intent prompt. Decoding: greedy.",
        "",
        "| Set | LoRA | Retrieval |",
        "|---|---|---|",
    ]
    for name, accuracy in scores:
        lines.append(f"| {name} | {accuracy:.2f} | {RETRIEVAL[name]:.2f} |")
    lines.append("")
    text = "\n".join(lines)
    with open("evaluation/results/finetune_retrieval_scores.md", "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)


if __name__ == "__main__":
    main()
