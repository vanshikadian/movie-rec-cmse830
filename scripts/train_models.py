"""Split ratings, fit baselines + matrix factorization, tune on validation, evaluate on test once.

    python scripts/train_models.py [--seed 830] [--processed-dir data/processed] [--models-dir models]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as d  # noqa: E402
from src import models as m  # noqa: E402

BIAS_GRID = list(
    itertools.product([2, 5, 10, 25], [2, 5, 10, 25])
)  # (item regularization, user regularization)
MF_GRID = list(itertools.product([50, 100], [0.02, 0.05, 0.1]))  # (factors, regularization)
MF_EPOCHS, MF_LR = 20, 0.005


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=830)
    p.add_argument("--processed-dir", type=Path, default=d.PROCESSED_DIR)
    p.add_argument("--models-dir", type=Path, default=m.MODELS_DIR)
    a = p.parse_args()
    a.models_dir.mkdir(parents=True, exist_ok=True)

    ratings = d.load_ratings(a.processed_dir)
    split = m.make_split(ratings, seed=a.seed)
    parts = m.split_frames(ratings, split)
    train, val, test = parts["train"], parts["val"], parts["test"]

    print("split checks")
    key = lambda df: set(zip(df.userId, df.movieId))
    check(
        len(train) + len(val) + len(test) == len(ratings), "train+val+test rows equal total ratings"
    )
    check(
        not (key(train) & key(val)) and not (key(train) & key(test)) and not (key(val) & key(test)),
        "no user-movie pair appears in two splits",
    )
    check(
        set(train.userId) == set(ratings.userId),
        "every user has training ratings (no unseen users)",
    )
    check(m.make_split(ratings, seed=a.seed).equals(split), "split reproduces for the same seed")
    split.to_parquet(a.processed_dir / "split.parquet", index=False)
    split_cfg = {
        "seed": a.seed,
        "val_frac": 0.1,
        "test_frac": 0.2,
        "scheme": "random within each user (NOT chronological)",
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (a.models_dir / "split_config.json").write_text(json.dumps(split_cfg, indent=2))

    model_a = m.fit_global_mean(train)

    # Choose regularization using validation ratings.
    t0 = time.time()
    bias_rows = []
    for li, lu in BIAS_GRID:
        r = m.evaluate(m.fit_bias(train, li, lu), val)
        bias_rows.append(
            {
                "model": "bias",
                "lam_item": li,
                "lam_user": lu,
                "val_rmse": r["all"]["rmse"],
                "val_mae": r["all"]["mae"],
            }
        )
    best_b = min(bias_rows, key=lambda r: r["val_rmse"])
    model_b = m.fit_bias(train, best_b["lam_item"], best_b["lam_user"])
    print(
        f"bias tuning done in {time.time() - t0:.0f}s: best lam_item={best_b['lam_item']}, lam_user={best_b['lam_user']}"
    )

    # Retain the best fitted model and its reference predictor.
    mf_rows = []
    best_mf = model_mf = algo_chk = None
    for nf, reg in MF_GRID:
        t0 = time.time()
        model, algo = m.fit_mf(train, nf, reg, MF_EPOCHS, MF_LR, seed=a.seed)
        r = m.evaluate(model, val)
        mf_rows.append(
            {
                "model": "mf",
                "n_factors": nf,
                "reg": reg,
                "val_rmse": r["all"]["rmse"],
                "val_mae": r["all"]["mae"],
            }
        )
        if best_mf is None or mf_rows[-1]["val_rmse"] < best_mf["val_rmse"]:
            best_mf, model_mf, algo_chk = mf_rows[-1], model, algo
        print(
            f"  mf factors={nf} reg={reg}: val RMSE {r['all']['rmse']:.4f} ({time.time() - t0:.0f}s)"
        )
    pd.DataFrame(bias_rows + mf_rows).to_csv(a.models_dir / "tuning_results.csv", index=False)

    # sanity: our numpy prediction equals surprise's own prediction
    sample = val.head(2000)
    ref = np.array(
        [
            algo_chk.predict(int(u), int(i), clip=False).est
            for u, i in zip(sample.userId, sample.movieId)
        ]
    )
    ours = m.predict(model_mf, sample.userId.to_numpy(), sample.movieId.to_numpy(), clip=False)
    check(
        np.allclose(ref, ours, atol=1e-6),
        "numpy prediction matches surprise's own predict (2000 val ratings)",
    )

    # Score the selected models on test ratings.
    final = {
        "baseline_global_mean": model_a,
        "baseline_bias": model_b,
        "matrix_factorization": model_mf,
    }
    metrics = {
        "split": split_cfg,
        "selected": {"bias": best_b, "mf": best_mf},
        "unseen_item_handling": "prediction = mu + bu[user] (item terms omitted); clipped to [0.5, 5.0]",
        "val": {k: m.evaluate(v, val) for k, v in final.items()},
        "test": {k: m.evaluate(v, test) for k, v in final.items()},
    }
    (a.models_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for name, model in final.items():
        m.save_model(model, a.models_dir / f"{name}.npz")
    print("\ntest metrics (clipped predictions)")
    for k, r in metrics["test"].items():
        print(
            f"  {k:22s} n={r['n_ratings']:,} MAE {r['all']['mae']:.4f} RMSE {r['all']['rmse']:.4f} | warm-only MAE {r['warm_only']['mae']:.4f} "
            f"RMSE {r['warm_only']['rmse']:.4f} | warm coverage {r['warm_coverage']:.4f} unseen items {r['n_unseen_items']} unseen users {r['n_unseen_users']}"
        )


if __name__ == "__main__":
    main()
