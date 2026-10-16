"""Checks of the group-ranking logic: formulas, exclusions, shared pool, ties, errors.

    python scripts/check_group.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as d, group as g, models as m  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


ratings, movies, genres = d.load_ratings(), d.load_movies(), d.load_movie_genres()
split = pd.read_parquet(d.PROCESSED_DIR / "split.parquet")
ctx = g.build_context(ratings, split, movies, genres)
model = m.load_model(m.MODELS_DIR / "matrix_factorization.npz")
A, B = 59, 83
c = g.build_candidates(model, ctx, A, B)

print("candidate pool")
tr = ctx["train"]
seen_train = set(tr[tr.userId.isin([A, B])].movieId)
check(
    not set(c.movieId) & seen_train, "no movie from either user's training history is a candidate"
)
c_all = g.build_candidates(model, ctx, A, B, exclude="all_recorded")
seen_all = set(ratings[ratings.userId.isin([A, B])].movieId)
check(
    not set(c_all.movieId) & seen_all and set(c_all.movieId) < set(c.movieId),
    "demo mode also drops val/test-rated movies (strict subset)",
)
check(
    c.movieId.is_unique and (c.pred_a.between(0.5, 5) & c.pred_b.between(0.5, 5)).all(),
    "unique movies, predictions clipped to [0.5, 5]",
)
direct = m.predict(model, np.full(len(c), A), c.movieId.to_numpy())
check(
    np.allclose(direct, c.pred_a), "candidate predictions equal models.predict for the same pairs"
)
check((c.item_train_n >= 10).all(), "minimum item training count respected")

print("formulas and shared pool")
avg, low, thr = (g.rank_candidates(c, s, 3.5, 4.0) for s in g.STRATEGIES)
check(np.allclose(avg.score, (avg.pred_a + avg.pred_b) / 2), "average score = (A + B) / 2")
check(np.allclose(low.score, np.minimum(low.pred_a, low.pred_b)), "lower score = min(A, B)")
check(len(avg) == len(low) == len(c), "average and lower rank the entire shared pool")
check(
    (thr.pred_a >= 3.5).all() and (thr.pred_b >= 4.0).all() and set(thr.movieId) <= set(c.movieId),
    "threshold keeps only movies meeting both minimums, from the same pool",
)
expected = c[(c.pred_a >= 3.5) & (c.pred_b >= 4.0)]
check(
    set(thr.movieId) == set(expected.movieId) and np.allclose(thr.score, thr.avg_pred),
    "threshold retains exactly the qualifying movies, scored by average",
)
for r in (avg, low, thr):
    check(
        r["score"].is_monotonic_decreasing and (r["rank"] == np.arange(1, len(r) + 1)).all(),
        "scores descending, ranks 1..n",
    )

print("deterministic ties")
tied = c.head(6).copy()
tied[["pred_a", "pred_b", "avg_pred", "min_pred"]] = 4.0
tied = tied.sort_values("movieId", ascending=False)
check(
    g.rank_candidates(tied, "average").movieId.tolist() == sorted(tied.movieId),
    "full ties resolve by ascending movieId, regardless of input order",
)
check(
    g.rank_candidates(c.sample(frac=1, random_state=1), "lower").movieId.tolist()
    == low.movieId.tolist(),
    "ranking independent of row order",
)

print("honest failures")
try:
    g.build_candidates(model, ctx, A, 999999999)
except ValueError as e:
    check("not a trained profile" in str(e), "unknown user ID raises a clear error")
else:
    raise AssertionError("unknown user ID was accepted")
check(
    len(g.rank_candidates(c, "threshold", 5.0, 5.0)) == 0,
    "impossible threshold returns an empty table (no fallback)",
)
same = g.build_candidates(model, ctx, A, A)
check(
    np.allclose(same.pred_a, same.pred_b)
    and len(g.compare_strategies(same, 10)[2]["average"]) == 10,
    "identical users: equal predictions, strategies still computed",
)
check(
    len(g.build_candidates(model, ctx, A, B, genre_filter=["NoSuchGenre"])) == 0,
    "unmatched genre filter gives an empty pool",
)

print("split integrity and comparison")
key = lambda s: set(zip(*[ctx["all"][ctx["all"].split == s][k] for k in ("userId", "movieId")]))
check(
    not (key("train") & key("val"))
    and not (key("train") & key("test"))
    and not (key("val") & key("test")),
    "train/val/test rating pairs do not overlap",
)
summary, overlap, tops = g.compare_strategies(c, k=10)
check(
    all(overlap.loc[s, s] == len(tops[s]) for s in g.STRATEGIES),
    "overlap diagonal equals number selected",
)
print("done")
