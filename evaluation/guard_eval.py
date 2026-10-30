"""Calibrate and evaluate the VAE out-of-scope guard.

Protocol (no leakage):
  * calibration set  : test rows that are NOT in the development or held-out subsets (about 2,880 in-domain messages);
  * evaluation       : the 100 held-out messages (false rejections) and 40 hand-written out-of-scope messages (catch rate);
  * thresholds       : the P-th percentile (default 99) of the calibration scores, stored in models/vae.pt.

Run from the project root:  python -m evaluation.guard_eval  [--percentile 99]"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.guard import CHECKPOINT, OUT_OF_SCOPE, scores

ap = argparse.ArgumentParser()
ap.add_argument("--percentile", type=float, default=99)
a = ap.parse_args()

test = pd.read_csv("data/test.csv")
dev = pd.read_csv("evaluation/eval_subset.csv")
held = pd.read_csv("evaluation/heldout_subset.csv")
cal = test[~test["text"].isin(set(dev["text"]) | set(held["text"]))]
col = lambda rows, k: np.array([r[k] for r in rows])
S_cal, S_held, S_oos = scores(cal["text"].tolist()), scores(held["text"].tolist()), scores(OUT_OF_SCOPE)

res = {"n_calibration": len(cal), "n_heldout": len(held), "n_oos": len(OUT_OF_SCOPE), "percentile": a.percentile, "auroc": {}, "curve": []}
for k in ("elbo", "unk"):
    y = np.r_[np.zeros(len(S_cal)), np.ones(len(S_oos))]
    res["auroc"][k] = float(roc_auc_score(y, np.r_[col(S_cal, k), col(S_oos, k)]))
    res[f"mean_in_domain_{k}"] = float(col(S_cal, k).mean()); res[f"mean_oos_{k}"] = float(col(S_oos, k).mean())

for p in (90, 95, 99):
    t = {k: float(np.percentile(col(S_cal, k), p)) for k in ("elbo", "unk")}
    flag = lambda S: np.array([(r["elbo"] > t["elbo"]) or (r["unk"] > t["unk"]) for r in S])
    only = lambda S, k: np.array([r[k] > t[k] for r in S])
    res["curve"].append({"percentile": p, "thresholds": t,
                         "false_reject_heldout_pct": 100 * float(flag(S_held).mean()), "oos_caught_pct": 100 * float(flag(S_oos).mean()),
                         "elbo_only": {"false_reject_pct": 100 * float(only(S_held, "elbo").mean()), "oos_caught_pct": 100 * float(only(S_oos, "elbo").mean())},
                         "unk_only": {"false_reject_pct": 100 * float(only(S_held, "unk").mean()), "oos_caught_pct": 100 * float(only(S_oos, "unk").mean())}})
chosen = {k: float(np.percentile(col(S_cal, k), a.percentile)) for k in ("elbo", "unk")}
res["chosen_thresholds"] = chosen
ck = torch.load(CHECKPOINT, weights_only=False); ck["thresholds"] = chosen; torch.save(ck, CHECKPOINT)
json.dump(res, open("evaluation/results/guard_eval.json", "w"), indent=2)
print(json.dumps(res, indent=2))
