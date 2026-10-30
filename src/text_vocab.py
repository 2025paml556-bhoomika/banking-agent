"""Word-level tokenizer used by the VAE and GAN (not by the LLM, which has its own BPE tokenizer)."""

import collections
import json
import re

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")
PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq: int = 2) -> "Vocab":
        counts = collections.Counter(w for t in texts for w in tokenize(t))
        words = sorted(w for w, c in counts.items() if c >= min_freq)
        return cls(SPECIALS + words)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str, max_len: int = 32) -> list[int]:
        """[bos] w1 ... wn [eos], truncated to max_len words."""
        ids = [self.stoi.get(w, UNK) for w in tokenize(text)][:max_len]
        return [BOS] + ids + [EOS]

    def decode(self, ids) -> str:
        words = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i in (PAD, BOS):
                continue
            words.append(self.itos[i])
        text = " ".join(words)
        text = re.sub(r"\s+([?.!,])", r"\1", text)
        return text.strip()

    def to_json(self) -> str:
        return json.dumps(self.itos)

    @classmethod
    def from_json(cls, s: str) -> "Vocab":
        return cls(json.loads(s))
