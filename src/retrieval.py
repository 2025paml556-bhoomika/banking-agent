"""Adaptation experiment (Req 8): retrieval few-shot. The 5 examples are the train
messages most similar to the customer message (TF-IDF cosine), not 5 random ones."""

import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.baseline import TASK, parse_intent
from src.llm import generate
from src.prompts import CONSTRAINT, TRAIN

K = 5
# Default pool is the real train rows. RETRIEVAL_POOL=vae, gan, or vae+gan adds filtered synthetic rows.
POOL = TRAIN
for _src in filter(None, os.environ.get("RETRIEVAL_POOL", "").split("+")):
    POOL = pd.concat([POOL, pd.read_csv(f"data/synthetic_{_src}.csv")[["text", "category"]]], ignore_index=True)
vectorizer = TfidfVectorizer(ngram_range=(1, 2)).fit(POOL["text"])
TRAIN_VECTORS = vectorizer.transform(POOL["text"])


def retrieve(message: str) -> pd.DataFrame:
    scores = cosine_similarity(vectorizer.transform([message]), TRAIN_VECTORS)[0]
    return POOL.iloc[scores.argsort()[::-1][:K]]


def build_prompt(message: str) -> list[dict]:
    shots = "".join(f"Message: {r.text}\nIntent: {r.category}\n\n" for r in retrieve(message).itertuples())
    return [{"role": "user", "content": f"{TASK}\n{CONSTRAINT}\n\n{shots}Message: {message}\nIntent:"}]


if __name__ == "__main__":
    data = pd.read_csv("evaluation/eval_subset.csv")
    data["output"] = [generate(build_prompt(t), max_new_tokens=32) for t in data["text"]]
    data["predicted"] = [parse_intent(o) for o in data["output"]]
    data["correct"] = data["predicted"] == data["category"]
    # Reference: the answer a plain nearest-neighbour vote would give from the same 5 examples.
    data["knn_vote"] = [retrieve(t)["category"].mode()[0] for t in data["text"]]
    data["knn_correct"] = data["knn_vote"] == data["category"]
    pool_tag = os.environ.get("RETRIEVAL_POOL", "")
    data.to_csv(f"evaluation/results/results_retrieval{'_' + pool_tag if pool_tag else ''}.csv", index=False)

    print(f"accuracy (LLM + retrieved examples): {data['correct'].mean():.2f}")
    print(f"accuracy (5-NN vote, no LLM):        {data['knn_correct'].mean():.2f}")
    print(f"no intent found: {(data['predicted'] == '').sum()}")
    print(f"correct answer present in the 5 examples: {sum(c in retrieve(t)['category'].values for t, c in zip(data['text'], data['category']))}/100")
    for _, row in data.head(3).iterrows():
        print(f"\n  {row['text']!r} -> {row['output']!r} (expected {row['category']})")
