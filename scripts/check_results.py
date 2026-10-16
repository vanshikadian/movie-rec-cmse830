"""Check saved rating metrics, ID mappings and split validation without fitting models."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as d, group as g, models as m  # noqa: E402


def main():
    ratings = d.load_ratings()
    split = pd.read_parquet(d.PROCESSED_DIR / "split.parquet")
    parts = m.split_frames(ratings, split)
    movies, genres = d.load_movies(), d.load_movie_genres()
    ctx = g.build_context(ratings, split, movies, genres)
    metrics = json.loads((m.MODELS_DIR / "metrics.json").read_text())
    assert set(ratings.movieId) <= set(movies.movieId)
    assert movies.movieId.is_unique
    assert not genres.duplicated(["movieId", "genre"]).any()
    assert m.make_split(ratings, seed=metrics["split"]["seed"]).equals(split)
    for name in metrics["test"]:
        model = m.load_model(m.MODELS_DIR / f"{name}.npz")
        assert np.array_equal(model["user_ids"], np.sort(parts["train"].userId.unique()))
        assert np.array_equal(model["item_ids"], np.sort(parts["train"].movieId.unique()))
        for part in ("val", "test"):
            actual = m.evaluate(model, parts[part])
            expected = metrics[part][name]
            for key in ("n_ratings", "n_unseen_users", "n_unseen_items"):
                assert actual[key] == expected[key], (name, part, key)
            for scope in ("all", "warm_only", "unseen_item_only"):
                for metric in ("mae", "rmse"):
                    assert np.isclose(
                        actual[scope][metric], expected[scope][metric], rtol=0, atol=1e-12
                    ), (name, part, scope, metric)
        user = int(model["user_ids"][0])
        direct = m.predict(model, np.full(len(model["item_ids"]), user), model["item_ids"])
        assert np.allclose(direct, m.predict_user_all_items(model, user), rtol=0, atol=1e-12)
        unknown = int(max(model["user_ids"].max(), model["item_ids"].max())) + 1
        assert m.predict(model, [unknown], [unknown])[0] == np.clip(model["mu"], 0.5, 5)
        print(f"ok: {name} metrics and ID mappings")
    for invalid in (split.iloc[:-1], split.assign(split="invalid")):
        for validate in (
            lambda: m.split_frames(ratings, invalid),
            lambda: g.build_context(ratings, invalid, movies, genres),
        ):
            try:
                validate()
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid split was accepted")
    model = m.load_model(m.MODELS_DIR / "matrix_factorization.npz")
    short_ctx = dict(ctx, user_train_n=ctx["user_train_n"].copy())
    short_ctx["user_train_n"].loc[157] = 9
    try:
        g.build_candidates(model, short_ctx, 157, 21008)
    except ValueError as e:
        assert "too little history" in str(e)
    else:
        raise AssertionError("Insufficient training history was accepted")
    print("ok: incomplete splits, invalid labels and insufficient history rejected")


if __name__ == "__main__":
    main()
