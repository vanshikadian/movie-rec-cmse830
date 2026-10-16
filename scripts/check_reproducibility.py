"""Run the preparation twice into scratch folders and compare the outputs.

    python scripts/check_reproducibility.py [--n-users 5000] [--seed 830]
"""

import argparse
import json
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--n-users", default="5000")
p.add_argument("--seed", default="830")
a = p.parse_args()

with tempfile.TemporaryDirectory() as tmp:
    outs = []
    for i in (1, 2):
        out = Path(tmp) / f"run{i}"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "prepare_data.py"),
                "--n-users",
                a.n_users,
                "--seed",
                a.seed,
                "--out-dir",
                str(out),
                "--report",
                str(Path(tmp) / f"report{i}.md"),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        outs.append(out)
    for name in ["ratings", "movies", "movie_genres"]:
        assert_frame_equal(
            pd.read_parquet(outs[0] / f"{name}.parquet"),
            pd.read_parquet(outs[1] / f"{name}.parquet"),
        )
        print(f"ok: {name}.parquet identical across runs")
    for name in ["selected_user_ids.csv", "audit_full_source.json", "audit_subset.json"]:
        assert (outs[0] / name).read_text() == (outs[1] / name).read_text(), name
        print(f"ok: {name} identical across runs")
    ids = lambda o: set(pd.read_csv(o / "selected_user_ids.csv")["userId"])
    assert ids(outs[0]) == set(
        pd.read_csv(ROOT / "data/processed/selected_user_ids.csv")["userId"]
    ), "differs from the saved run"
    print("ok: sample matches the existing data/processed run")

import numpy as np

sys.path.insert(0, str(ROOT))
from src import models as m  # noqa: E402

if (ROOT / "models" / "metrics.json").exists():
    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp) / "processed"
        processed.mkdir()
        shutil.copy2(ROOT / "data/processed/ratings.parquet", processed / "ratings.parquet")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "train_models.py"),
                "--processed-dir",
                str(processed),
                "--models-dir",
                tmp,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        assert json.loads((Path(tmp) / "metrics.json").read_text()) == json.loads(
            (ROOT / "models" / "metrics.json").read_text()
        )
        print("ok: retrained metrics.json identical to saved metrics")
        for name in ["baseline_global_mean", "baseline_bias", "matrix_factorization"]:
            a_, b_ = m.load_model(Path(tmp) / f"{name}.npz"), m.load_model(
                ROOT / "models" / f"{name}.npz"
            )
            assert (
                all(np.array_equal(a_[k], b_[k]) for k in ("bu", "bi", "P", "Q"))
                and a_["mu"] == b_["mu"]
            ), name
            print(f"ok: retrained {name} parameters identical")
