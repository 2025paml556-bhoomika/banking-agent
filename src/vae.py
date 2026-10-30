"""Conditional text VAE (GRU encoder/decoder) for BANKING77 messages.

The VAE learns p(message | intent, z). It is used for
  1. generating new labelled customer messages (data augmentation, `src/augment.py`),
  2. scoring how "banking-like" a message is (out-of-scope guard, `src/guard.py`),
  3. providing the latent space in which the GAN (`src/gan.py`) is trained.

Tricks against posterior collapse: word dropout on the decoder input, KL annealing, free bits,
and the latent z is fed to the decoder at every step (not only as initial state).
Label dropout (the "null" intent) lets the same network also act as an unconditional model.
"""

import logging
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text_vocab import BOS, EOS, PAD, UNK, Vocab

log = logging.getLogger("vae")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CVAE(nn.Module):
    def __init__(self, vocab_size, n_labels, emb=128, hid=256, zdim=64, ydim=32, word_dropout=0.3):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, n_labels=n_labels, emb=emb, hid=hid, zdim=zdim, ydim=ydim, word_dropout=word_dropout)
        self.zdim, self.n_labels, self.word_dropout = zdim, n_labels, word_dropout
        self.null = n_labels  # index of the "no label" condition
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=PAD)
        self.y_emb = nn.Embedding(n_labels + 1, ydim)
        self.enc = nn.GRU(emb, hid, batch_first=True, bidirectional=True)
        self.to_mu = nn.Linear(2 * hid + ydim, zdim)
        self.to_logvar = nn.Linear(2 * hid + ydim, zdim)
        self.z_to_h = nn.Linear(zdim + ydim, hid)
        self.dec = nn.GRU(emb + zdim, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab_size)

    # ---- encoder -------------------------------------------------------------------------
    def encode(self, x, y):
        lengths = (x != PAD).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)  # h: (2, B, hid)
        h = torch.cat([h[0], h[1], self.y_emb(y)], dim=-1)
        return self.to_mu(h), self.to_logvar(h)

    # ---- decoder -------------------------------------------------------------------------
    def decode_logits(self, z, y, x_in):
        B, T = x_in.shape
        if self.training and self.word_dropout > 0:
            drop = (torch.rand(B, T, device=x_in.device) < self.word_dropout) & (x_in != PAD) & (x_in != BOS)
            x_in = x_in.masked_fill(drop, UNK)
        h0 = torch.tanh(self.z_to_h(torch.cat([z, self.y_emb(y)], -1))).unsqueeze(0)
        inp = torch.cat([self.emb(x_in), z.unsqueeze(1).expand(B, T, -1)], -1)
        o, _ = self.dec(inp, h0)
        return self.out(o)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        logits = self.decode_logits(z, y, x[:, :-1])
        return logits, mu, logvar

    @torch.no_grad()
    def generate(self, z, y, max_len=30, temperature=0.8, top_k=20):
        """Autoregressive sampling from the decoder given latent codes z and labels y."""
        self.eval()
        B = z.shape[0]
        h = torch.tanh(self.z_to_h(torch.cat([z, self.y_emb(y)], -1))).unsqueeze(0)
        tok = torch.full((B,), BOS, dtype=torch.long, device=z.device)
        out = []
        for _ in range(max_len):
            inp = torch.cat([self.emb(tok), z], -1).unsqueeze(1)
            o, h = self.dec(inp, h)
            logits = self.out(o[:, 0]) / max(temperature, 1e-5)
            logits[:, [PAD, BOS, UNK]] = -1e9
            if top_k:
                kth = logits.topk(top_k, -1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, -1e9)
            tok = torch.multinomial(F.softmax(logits, -1), 1)[:, 0]
            out.append(tok)
        return torch.stack(out, 1)


def collate(batch, device=DEVICE):
    xs, ys = zip(*batch)
    T = max(len(x) for x in xs)
    X = torch.full((len(xs), T), PAD, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, : len(x)] = torch.tensor(x)
    return X.to(device), torch.tensor(ys).to(device)


def elbo_terms(model, x, y, free_bits=0.02):
    logits, mu, logvar = model(x, y)
    tgt = x[:, 1:]
    rec = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD, reduction="none").view(tgt.shape).sum(1)
    kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).mean(0)  # per latent dimension, averaged over batch
    kl_free = torch.clamp(kl_dim, min=free_bits).sum()
    return rec.mean(), kl_dim.sum(), kl_free


def train_vae(texts, labels, vocab: Vocab, n_labels, epochs=40, batch=128, lr=2e-3, anneal_epochs=15, label_dropout=0.15, seed=0, beta_max=0.5, free_bits=0.1, **cfg):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = CVAE(len(vocab), n_labels, **cfg).to(DEVICE)
    data = [(vocab.encode(t), int(l)) for t, l in zip(texts, labels)]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        random.shuffle(data)
        beta = beta_max * min(1.0, ep / anneal_epochs)
        tot = dict(rec=0.0, kl=0.0, n=0)
        for i in range(0, len(data), batch):
            x, y = collate(data[i : i + batch])
            drop = torch.rand(y.shape, device=y.device) < label_dropout
            y = y.masked_fill(drop, model.null)
            rec, kl, kl_free = elbo_terms(model, x, y, free_bits)
            loss = rec + beta * kl_free
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot["rec"] += rec.item() * len(x); tot["kl"] += kl.item() * len(x); tot["n"] += len(x)
        history.append(dict(epoch=ep, beta=beta, rec=tot["rec"] / tot["n"], kl=tot["kl"] / tot["n"]))
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            log.info("VAE epoch %d/%d beta=%.2f recon=%.2f nats/sentence KL=%.2f", ep, epochs, beta, history[-1]["rec"], history[-1]["kl"])
    return model, history


@torch.no_grad()
def encode_dataset(model, texts, labels, vocab, batch=512):
    """Posterior parameters (mu, logvar) of every training message."""
    model.eval()
    data = [(vocab.encode(t), int(l)) for t, l in zip(texts, labels)]
    mus, lvs = [], []
    for i in range(0, len(data), batch):
        x, y = collate(data[i : i + batch])
        mu, lv = model.encode(x, y)
        mus.append(mu); lvs.append(lv)
    return torch.cat(mus), torch.cat(lvs)


@torch.no_grad()
def score_texts(model, texts, vocab, batch=256):
    """Per-token negative ELBO with the null label: high = unlike the training messages."""
    model.eval()
    scores = []
    for i in range(0, len(texts), batch):
        enc = [(vocab.encode(t), model.null) for t in texts[i : i + batch]]
        x, y = collate(enc)
        mu, lv = model.encode(x, y)
        logits = model.decode_logits(mu, y, x[:, :-1])
        tgt = x[:, 1:]
        rec = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=PAD, reduction="none").view(tgt.shape).sum(1)
        kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(1)
        n = (tgt != PAD).sum(1)
        scores.extend(((rec + kl) / n).tolist())
    return np.array(scores)


def sample_messages(model, vocab, label_ids, temperature=0.8, top_k=20, z_scale=1.0, z=None):
    y = torch.tensor(label_ids, device=DEVICE)
    if z is None:
        z = torch.randn(len(label_ids), model.zdim, device=DEVICE) * z_scale
    ids = model.generate(z, y, temperature=temperature, top_k=top_k).cpu().numpy()
    return [vocab.decode(r) for r in ids]
