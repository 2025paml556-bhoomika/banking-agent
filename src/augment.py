"""VAE + GAN experiment: train both generators on BANKING77, generate labelled customer messages,
filter them, and measure whether they help (data augmentation) - in the full-data setting and in a
low-resource setting (10 % of the training data).

Run from the project root:   python -m src.augment            (about 25 minutes on 2 CPU cores)
Writes: models/vae.pt, models/gan.pt, data/synthetic_vae.csv, data/synthetic_gan.csv,
        evaluation/results/vae_gan_results.json
"""

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import torch
from scipy import linalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline

from src.gan import LatentGAN, sample_latents, train_gan
from src.text_vocab import Vocab, tokenize
from src.vae import CVAE, DEVICE, encode_dataset, sample_messages, score_texts, train_vae

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("augment")

TRAIN = pd.read_csv("data/train.csv")
TEST = pd.read_csv("data/test.csv")
DEV = pd.read_csv("evaluation/eval_subset.csv")
HELD = pd.read_csv("evaluation/heldout_subset.csv")
INTENTS = sorted(TRAIN["category"].unique())
I2ID = {c: i for i, c in enumerate(INTENTS)}


def make_validator(df):
    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2)), LogisticRegression(max_iter=1000))
    return clf.fit(df["text"], df["category"])


def accuracy(clf, df):
    return float((clf.predict(df["text"]) == df["category"]).mean())


def frechet(a, b):
    """Fréchet distance between Gaussians fitted to two sets of latent vectors (lower = closer)."""
    m1, m2 = a.mean(0), b.mean(0)
    c1, c2 = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    covmean = linalg.sqrtm(c1 @ c2).real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * covmean))


def text_stats(texts, labels, train_texts, validator):
    toks = [tokenize(t) for t in texts]
    uni = [w for t in toks for w in t]
    bi = [(t[i], t[i + 1]) for t in toks for i in range(len(t) - 1)]
    proba = validator.predict_proba(texts)
    pred = validator.classes_[proba.argmax(1)]
    train_set = set(t.lower() for t in train_texts)
    return {
        "n": len(texts),
        "unique_pct": 100 * len(set(texts)) / len(texts),
        "novel_pct": 100 * np.mean([t.lower() not in train_set for t in texts]),
        "distinct1": len(set(uni)) / max(1, len(uni)),
        "distinct2": len(set(bi)) / max(1, len(bi)),
        "mean_words": float(np.mean([len(t) for t in toks])),
        "intent_fidelity_pct": 100 * float((pred == np.array(labels)).mean()),
        "mean_validator_conf": float(proba.max(1).mean()),
    }


def generate(vae, gan, vocab, per_intent, source, temperature=0.8):
    labels = [c for c in INTENTS for _ in range(per_intent)]
    ids = [I2ID[c] for c in labels]
    out_all = []
    for i in range(0, len(ids), 1000):
        chunk = ids[i : i + 1000]
        if source == "gan":
            z = sample_latents(gan, chunk)
        else:
            z = torch.randn(len(chunk), vae.zdim, device=DEVICE)
        out_all += sample_messages(vae, vocab, chunk, temperature=temperature, z=z)
    return pd.DataFrame({"text": out_all, "category": labels, "source": source})


def filter_synthetic(df, validator, train_texts, min_conf=0.3):
    df = df[df["text"].str.split().str.len() >= 3].copy()
    train_set = set(t.lower() for t in train_texts)
    df = df[~df["text"].str.lower().isin(train_set)].drop_duplicates("text")
    proba = validator.predict_proba(df["text"])
    pred = validator.classes_[proba.argmax(1)]
    keep = (pred == df["category"].values) & (proba.max(1) >= min_conf)
    return df[keep].reset_index(drop=True)


def knn_vote_accuracy(pool, queries, k=5):
    vec = TfidfVectorizer(ngram_range=(1, 2)).fit(pool["text"])
    sims = cosine_similarity(vec.transform(queries["text"]), vec.transform(pool["text"]))
    top = sims.argsort(1)[:, ::-1][:, :k]
    labels = pool["category"].values
    pred = [pd.Series(labels[r]).mode()[0] for r in top]
    return float((np.array(pred) == queries["category"].values).mean())


def fit_generators(df, args, seed, reuse_vae=False):
    ids = [I2ID[c] for c in df["category"]]
    if reuse_vae:  # keep the VAE in models/vae.pt (and its calibrated guard thresholds); train only the GAN
        ck = torch.load("models/vae.pt", map_location=DEVICE, weights_only=False)
        vae = CVAE(**ck["cfg"]).to(DEVICE); vae.load_state_dict(ck["state"]); vae.eval()
        vocab, hist = Vocab.from_json(ck["vocab"]), None
    else:
        vocab = Vocab.build(df["text"], min_freq=2)
        vae, hist = train_vae(df["text"].tolist(), ids, vocab, len(INTENTS), epochs=args.vae_epochs, seed=seed)
    mu, lv = encode_dataset(vae, df["text"].tolist(), ids, vocab)
    gan, ghist = train_gan(mu, lv, ids, len(INTENTS), steps=args.gan_steps, lr=args.gan_lr, seed=seed)
    return vae, gan, vocab, hist, ghist, mu, lv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-epochs", type=int, default=40)
    ap.add_argument("--gan-steps", type=int, default=8000)
    ap.add_argument("--gan-lr", type=float, default=3e-4)
    ap.add_argument("--reuse-vae", action="store_true", help="do not retrain the VAE; train the GAN on models/vae.pt")
    ap.add_argument("--per-intent", type=int, default=30)
    ap.add_argument("--low-frac", type=float, default=0.1)
    ap.add_argument("--low-seeds", type=int, default=3)
    ap.add_argument("--skip-low", action="store_true")
    args = ap.parse_args()
    R = {"args": vars(args)}
    t0 = time.time()

    # ------------------------------------------------------------------ full-data setting
    log.info("== Full data: %d training messages", len(TRAIN))
    vae, gan, vocab, hist, ghist, mu, lv = fit_generators(TRAIN, args, seed=0, reuse_vae=args.reuse_vae)
    if not args.reuse_vae:
        torch.save({"cfg": vae.cfg, "state": vae.state_dict(), "vocab": vocab.to_json(), "intents": INTENTS}, "models/vae.pt")
    torch.save({"cfg": gan.cfg, "state": gan.state_dict()}, "models/gan.pt")
    R["vae_history"] = hist[-1] if hist else json.load(open("evaluation/results/vae_gan_results.json"))["vae_history"]
    R["gan_history"] = ghist[-1]

    validator = make_validator(TRAIN)
    R["reference_lr"] = {"dev": accuracy(validator, DEV), "heldout": accuracy(validator, HELD), "test_full": accuracy(validator, TEST)}
    log.info("reference LR: %s", R["reference_lr"])

    raw = {s: generate(vae, gan, vocab, args.per_intent, s) for s in ("vae", "gan")}
    R["gen_stats"] = {s: text_stats(df["text"].tolist(), df["category"].tolist(), TRAIN["text"], validator) for s, df in raw.items()}
    # Latent-space quality: Fréchet distance to the real posterior samples (lower = better).
    real_z = (mu + torch.randn_like(mu) * torch.exp(0.5 * lv)).cpu().numpy()
    gz = sample_latents(gan, [I2ID[c] for c in TRAIN["category"]]).cpu().numpy()
    pz = torch.randn_like(mu).cpu().numpy()
    R["latent_frechet"] = {"prior_N(0,I)_vs_real": frechet(real_z, pz), "gan_vs_real": frechet(real_z, gz)}
    filt = {s: filter_synthetic(df, validator, TRAIN["text"]) for s, df in raw.items()}
    R["kept_after_filter"] = {s: len(df) for s, df in filt.items()}
    for s, df in filt.items():
        df.to_csv(f"data/synthetic_{s}.csv", index=False)
    log.info("kept after filter: %s", R["kept_after_filter"])

    # Downstream: TF-IDF + LR trained on real vs real + synthetic.
    down = {}
    def eval_pool(name, df):
        clf = make_validator(df)
        down[name] = {"dev": accuracy(clf, DEV), "heldout": accuracy(clf, HELD), "test_full": accuracy(clf, TEST), "n_train": len(df)}
        log.info("downstream %s: %s", name, down[name])
    eval_pool("real", TRAIN)
    eval_pool("real+vae(filtered)", pd.concat([TRAIN, filt["vae"][["text", "category"]]]))
    eval_pool("real+gan(filtered)", pd.concat([TRAIN, filt["gan"][["text", "category"]]]))
    eval_pool("real+vae+gan(filtered)", pd.concat([TRAIN, filt["vae"][["text", "category"]], filt["gan"][["text", "category"]]]))
    eval_pool("real+vae(unfiltered)", pd.concat([TRAIN, raw["vae"][["text", "category"]]]))
    eval_pool("real+gan(unfiltered)", pd.concat([TRAIN, raw["gan"][["text", "category"]]]))
    R["downstream_full"] = down

    # Retrieval pool (what the LLM sees as few-shot examples) - 5-NN vote needs no LLM.
    pools = {"real": TRAIN, "real+vae": pd.concat([TRAIN, filt["vae"][["text", "category"]]]), "real+gan": pd.concat([TRAIN, filt["gan"][["text", "category"]]])}
    R["knn5_vote"] = {n: {"dev": knn_vote_accuracy(p, DEV), "heldout": knn_vote_accuracy(p, HELD)} for n, p in pools.items()}
    log.info("5-NN vote: %s", R["knn5_vote"])

    # ------------------------------------------------------------------ low-resource setting
    if not args.skip_low:
        low = []
        for seed in range(args.low_seeds):
            sub = TRAIN.groupby("category", group_keys=False).sample(frac=args.low_frac, random_state=seed).reset_index(drop=True)
            log.info("== Low-resource seed %d: %d real messages", seed, len(sub))
            a = argparse.Namespace(**{**vars(args), "vae_epochs": max(args.vae_epochs, 80), "gan_steps": 3000})
            v, g, vo, *_ = fit_generators(sub, a, seed)
            val = make_validator(sub)
            rawl = {s: generate(v, g, vo, args.per_intent, s) for s in ("vae", "gan")}
            fl = {s: filter_synthetic(d, val, sub["text"]) for s, d in rawl.items()}
            row = {"seed": seed, "n_real": len(sub), "kept_vae": len(fl["vae"]), "kept_gan": len(fl["gan"])}
            for name, d in {
                "real": sub,
                "real+vae(filtered)": pd.concat([sub, fl["vae"][["text", "category"]]]),
                "real+gan(filtered)": pd.concat([sub, fl["gan"][["text", "category"]]]),
                "real+vae+gan(filtered)": pd.concat([sub, fl["vae"][["text", "category"]], fl["gan"][["text", "category"]]]),
                "real+vae(unfiltered)": pd.concat([sub, rawl["vae"][["text", "category"]]]),
                "real+gan(unfiltered)": pd.concat([sub, rawl["gan"][["text", "category"]]]),
            }.items():
                clf = make_validator(d)
                row[name] = accuracy(clf, TEST)
            row["knn5_real"] = knn_vote_accuracy(sub, TEST.sample(1000, random_state=0))
            row["knn5_real+vae"] = knn_vote_accuracy(pd.concat([sub, fl["vae"][["text", "category"]]]), TEST.sample(1000, random_state=0))
            row["knn5_real+gan"] = knn_vote_accuracy(pd.concat([sub, fl["gan"][["text", "category"]]]), TEST.sample(1000, random_state=0))
            log.info("low-resource: %s", row)
            low.append(row)
        R["low_resource"] = low
        keys = [k for k in low[0] if k not in ("seed",)]
        R["low_resource_mean"] = {k: float(np.mean([r[k] for r in low])) for k in keys}
        R["low_resource_std"] = {k: float(np.std([r[k] for r in low])) for k in keys}

    # ------------------------------------------------------------------ samples for inspection
    for s_, df in raw.items():
        df.sample(60, random_state=0).to_csv(f"evaluation/results/samples_{s_}.csv", index=False)
    # The out-of-scope guard is calibrated separately: python -m evaluation.guard_eval
    R["seconds"] = time.time() - t0
    json.dump(R, open("evaluation/results/vae_gan_results.json", "w"), indent=2)
    log.info("done in %.0fs; wrote evaluation/results/vae_gan_results.json", R["seconds"])


if __name__ == "__main__":
    main()
