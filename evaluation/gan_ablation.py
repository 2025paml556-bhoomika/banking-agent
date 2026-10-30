"""GAN ablation: how many generator steps does the latent GAN need? Uses the trained VAE in models/vae.pt.
Metrics: Fréchet distance of generated latents to real posterior latents (lower = better), intent fidelity of the
decoded messages (share that a TF-IDF+LR validator assigns to the requested intent), novelty and unique share.

Run: python -m evaluation.gan_ablation        (about 6 minutes on 2 CPU cores)"""

import json
import logging

import numpy as np
import pandas as pd
import torch

from src.augment import I2ID, INTENTS, TRAIN, frechet, generate, make_validator, text_stats
from src.gan import sample_latents, train_gan
from src.text_vocab import Vocab
from src.vae import CVAE, DEVICE, encode_dataset

logging.getLogger("gan").setLevel(logging.WARNING)
ck = torch.load("models/vae.pt", map_location=DEVICE, weights_only=False)
vae = CVAE(**ck["cfg"]).to(DEVICE); vae.load_state_dict(ck["state"]); vae.eval()
vocab = Vocab.from_json(ck["vocab"])
ids = [I2ID[c] for c in TRAIN["category"]]
mu, lv = encode_dataset(vae, TRAIN["text"].tolist(), ids, vocab)
real_z = (mu + torch.randn_like(mu) * torch.exp(0.5 * lv)).cpu().numpy()
validator = make_validator(TRAIN)
rows = []
prior = generate(vae, None, vocab, 30, "vae")
s = text_stats(prior["text"].tolist(), prior["category"].tolist(), TRAIN["text"], validator)
rows.append({"model": "VAE prior N(0,I) (no GAN)", "steps": 0, "frechet": frechet(real_z, torch.randn_like(mu).cpu().numpy()), "fidelity_pct": s["intent_fidelity_pct"], "novel_pct": s["novel_pct"], "distinct2": s["distinct2"]})
for steps, lr in [(500, 1e-4), (3000, 1e-4), (8000, 1e-4), (8000, 3e-4)]:
    gan, _ = train_gan(mu, lv, ids, len(INTENTS), steps=steps, lr=lr, seed=0)
    gz = sample_latents(gan, ids).cpu().numpy()
    df = generate(vae, gan, vocab, 30, "gan")
    s = text_stats(df["text"].tolist(), df["category"].tolist(), TRAIN["text"], validator)
    rows.append({"model": f"latent GAN, lr {lr:g}", "steps": steps, "frechet": frechet(real_z, gz), "fidelity_pct": s["intent_fidelity_pct"], "novel_pct": s["novel_pct"], "distinct2": s["distinct2"]})
    print(rows[-1], flush=True)
json.dump(rows, open("evaluation/results/gan_ablation.json", "w"), indent=2)
print(pd.DataFrame(rows).to_string(index=False))
