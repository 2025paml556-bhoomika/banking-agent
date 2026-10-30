"""Out-of-scope guard (optional; enable with VAE_GUARD=1).

Two scores are computed for a message, both from the trained text VAE and its vocabulary:
  * elbo : per-token negative ELBO under the VAE (the generative-model score). High = the sentence is
           unlike the banking messages the VAE was trained on.
  * unk  : fraction of words that are not in the VAE vocabulary (1,520 words seen at least twice in
           the banking training data).
A message is flagged when either score is above a threshold. The thresholds are the 99th percentile of the
scores of in-domain messages that were not used for anything else (see evaluation/guard_eval.py),
so about 1 % of real banking messages are flagged by each test.
The check is cheap (one small GRU pass) and runs before the LLM. It does NOT replace the schema
validation: a banking-sounding message that fits no intent is still handled by classify_intent."""

import os
from functools import lru_cache

import torch

from src.text_vocab import UNK, Vocab

CHECKPOINT = os.environ.get("VAE_CHECKPOINT", "models/vae.pt")

# Hand-written out-of-scope messages (no banking content) used to test the guard. Not training data.
OUT_OF_SCOPE = [
    "What is the weather like in Paris tomorrow?",
    "Give me a recipe for chocolate chip cookies.",
    "Who won the football match last night?",
    "Write a python function that reverses a string.",
    "How tall is Mount Everest?",
    "Tell me a joke about cats.",
    "What is the capital of Australia?",
    "Can you recommend a good science fiction movie?",
    "How do I fix a leaking kitchen tap?",
    "Translate good morning into Spanish.",
    "What time does the sun set today?",
    "Explain how photosynthesis works.",
    "Which planet is closest to the sun?",
    "I want to learn how to play the guitar.",
    "How many calories are in a banana?",
    "Suggest a name for my new puppy.",
    "What is the meaning of life?",
    "How do I train for a marathon?",
    "Who painted the Mona Lisa?",
    "Play some relaxing music please.",
    "What is the speed of light?",
    "How do vaccines work?",
    "Recommend a hiking trail near Denver.",
    "Why is the sky blue?",
    "Write a poem about the ocean.",
    "How do I bake sourdough bread?",
    "What year did the second world war end?",
    "Set an alarm for seven in the morning.",
    "Which programming language should I learn first?",
    "How far is the moon from the earth?",
    "What are the symptoms of the flu?",
    "Tell me about the history of the Roman empire.",
    "How do I change a flat tyre on my bicycle?",
    "Is it going to rain this weekend?",
    "Who is the tallest basketball player ever?",
    "How do I grow tomatoes on a balcony?",
    "What does a black hole look like?",
    "Summarise the plot of Hamlet.",
    "Where can I watch the new superhero film?",
    "How many players are in a cricket team?",
]


@lru_cache(maxsize=1)
def _load():
    from src.vae import CVAE, DEVICE

    ck = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model = CVAE(**ck["cfg"]).to(DEVICE)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, Vocab.from_json(ck["vocab"]), ck.get("thresholds", {"elbo": ck.get("threshold", float("inf")), "unk": 1.0})


@torch.no_grad()
def scores(texts: list[str]) -> list[dict]:
    """Per message: {'elbo': per-token negative ELBO, 'unk': share of unknown words}."""
    import torch.nn.functional as F

    from src.text_vocab import PAD
    from src.vae import collate

    model, vocab, _ = _load()
    out = []
    for i in range(0, len(texts), 256):
        x, y = collate([(vocab.encode(t), model.null) for t in texts[i : i + 256]])
        mu, lv = model.encode(x, y)
        logits = model.decode_logits(mu, y, x[:, :-1])
        tgt = x[:, 1:]
        rec = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD, reduction="none").view(tgt.shape).sum(1)
        kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(1)
        n = (tgt != PAD).sum(1)
        unk = (tgt == UNK).sum(1) / n
        out += [{"elbo": float(e), "unk": float(u)} for e, u in zip(((rec + kl) / n).tolist(), unk.tolist())]
    return out


def vae_score(message: str) -> tuple[dict, dict]:
    """Return (scores, thresholds) for one message."""
    _, _, thresholds = _load()
    return scores([message])[0], thresholds


def is_out_of_scope(message: str) -> bool:
    s, t = vae_score(message)
    return s["elbo"] > t["elbo"] or s["unk"] > t["unk"]
