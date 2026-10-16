"""E1: baseline intent classification with one simple prompt, greedy decoding."""

import pandas as pd

from llm import generate

TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
INTENTS = sorted(pd.read_csv(TEST_URL)["category"].unique())
TASK = "Classify the customer message into one of these intents:\n" + ", ".join(INTENTS)


def build_prompt(message: str) -> list[dict]:
    return [{"role": "user", "content": f"{TASK}\n\nMessage: {message}\nIntent:"}]


def parse_intent(output: str) -> str:
    """Return the longest intent name found in the output, or '' if none."""
    text = output.lower().replace(" ", "_")
    found = [i for i in INTENTS if i.lower() in text]
    return max(found, key=len) if found else ""


if __name__ == "__main__":
    data = pd.read_csv("eval_subset.csv")
    outputs = [generate(build_prompt(t), max_new_tokens=32) for t in data["text"]]
    data["output"] = outputs
    data["predicted"] = [parse_intent(o) for o in outputs]
    data["correct"] = data["predicted"] == data["category"]
    data.to_csv("results_baseline.csv", index=False)

    print(f"accuracy: {data['correct'].mean():.2f}")
    print(f"no intent found: {(data['predicted'] == '').sum()}")
    for _, row in data.head(3).iterrows():
        print(f"\nmessage:  {row['text']}\noutput:   {row['output']!r}\nexpected: {row['category']}")
