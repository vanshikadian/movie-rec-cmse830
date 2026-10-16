"""Group ranking for two existing (trained) users, strategy comparison, simulated-pair evaluation.

Predictions are model ESTIMATES of what each person would rate a movie, clipped to the valid
rating range [0.5, 5.0] by `models.predict*`. The strategies are decision rules over those
estimates; they do not measure enjoyment while watching together.
"""

import numpy as np
import pandas as pd

from src import models as m

STRATEGIES = ("average", "lower", "threshold")
STRATEGY_LABELS = {
    "average": "Average",
    "lower": "Lower score (least misery)",
    "threshold": "Mutual threshold",
}
MIN_TRAIN_RATINGS = 10  # fewer training ratings than this: profile refused as insufficient
SHORT_HISTORY = 30  # fewer than this: profile flagged as short (estimates lean on movie averages)
EXCLUDE_MODES = ("training", "all_recorded")


def build_context(ratings, split, movies, genres):
    """Everything the ranking needs besides the model (no fitting happens here)."""
    r = ratings.merge(split, on=["userId", "movieId"], validate="one_to_one")
    if len(r) != len(ratings) or len(split) != len(ratings):
        raise ValueError("The split must contain exactly one row for every rating")
    if not r["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Split labels must be train, val or test")
    train = r[r["split"] == "train"]
    genre_text = genres.groupby("movieId")["genre"].agg(lambda s: ", ".join(sorted(s)))
    titles = movies.set_index("movieId")[["title"]].join(genre_text.rename("genres"))
    return {
        "all": r[["userId", "movieId", "rating", "split"]],
        "train": train,
        "item_train_n": train.groupby("movieId").size(),
        "user_train_n": train.groupby("userId").size(),
        "user_all_n": r.groupby("userId").size(),
        "titles": titles,
        "genres": genres,
    }


def profile_summary(ctx, user_id):
    t = ctx["train"][ctx["train"].userId == user_id]
    return {
        "userId": int(user_id),
        "n_train": int(len(t)),
        "n_recorded": int(ctx["user_all_n"].get(user_id, 0)),
        "train_mean": float(t.rating.mean()) if len(t) else None,
        "train_rating_counts": t.rating.value_counts().sort_index(),
    }


def build_candidates(
    model, ctx, user_a, user_b, exclude="training", min_item_train=10, genre_filter=None
):
    """One shared candidate pool for both viewers (and for every strategy).

    exclude="training"     drop movies either user rated in their TRAINING history (default).
    exclude="all_recorded" drop movies either user has any recorded rating for (demo discovery mode;
                           never use this to build held-out evaluation candidates).
    min_item_train: movie must have at least this many training ratings in the whole subset.
    genre_filter: optional list; keep movies having any of these genres.
    Raises ValueError for unknown users or users with insufficient history.
    """
    if exclude not in EXCLUDE_MODES:
        raise ValueError(f"exclude must be one of {EXCLUDE_MODES}")
    for u in (user_a, user_b):
        n = int(ctx["user_train_n"].get(u, 0))
        if u not in set(model["user_ids"]) or n == 0:
            raise ValueError(f"user {u} is not a trained profile in this subset")
        if n < MIN_TRAIN_RATINGS:
            raise ValueError(
                f"user {u} has only {n} training ratings (< {MIN_TRAIN_RATINGS}); too little history"
            )
    items = model["item_ids"]
    keep = np.ones(len(items), dtype=bool)
    source = ctx["train"] if exclude == "training" else ctx["all"]
    seen = set(source.loc[source.userId.isin([user_a, user_b]), "movieId"])
    keep &= ~np.isin(items, list(seen))
    n_item = ctx["item_train_n"].reindex(items).fillna(0).to_numpy()
    keep &= n_item >= min_item_train
    if genre_filter:
        keep &= np.isin(
            items, ctx["genres"].loc[ctx["genres"].genre.isin(genre_filter), "movieId"].unique()
        )
    pa, pb = m.predict_user_all_items(model, user_a), m.predict_user_all_items(model, user_b)
    c = pd.DataFrame(
        {
            "movieId": items[keep],
            "pred_a": pa[keep],
            "pred_b": pb[keep],
            "item_train_n": n_item[keep].astype(int),
        }
    )
    c["avg_pred"] = (c.pred_a + c.pred_b) / 2
    c["min_pred"] = np.minimum(c.pred_a, c.pred_b)
    c["gap"] = (c.pred_a - c.pred_b).abs()
    c = c.join(ctx["titles"], on="movieId")
    return c.sort_values("movieId").reset_index(drop=True)


def rank_candidates(cands, strategy, min_a=3.5, min_b=3.5):
    """Apply one strategy to the shared candidate pool; returns rows sorted best-first (may be empty).

    average:   score = (pred_a + pred_b) / 2
    lower:     score = min(pred_a, pred_b)
    threshold: keep pred_a >= min_a and pred_b >= min_b, then score = average
    Ties: score desc, then average desc, then lower desc, then movieId asc (fully deterministic).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}")
    out = cands
    if strategy == "threshold":
        out = out[(out.pred_a >= min_a) & (out.pred_b >= min_b)]
    out = out.assign(score=out["min_pred"] if strategy == "lower" else out["avg_pred"])
    out = out.sort_values(
        ["score", "avg_pred", "min_pred", "movieId"], ascending=[False, False, False, True]
    )
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def display_table(ranked):
    cols = [
        "rank",
        "movieId",
        "title",
        "genres",
        "pred_a",
        "pred_b",
        "avg_pred",
        "min_pred",
        "score",
        "item_train_n",
    ]
    return ranked[cols]


def compare_strategies(cands, k=10, min_a=3.5, min_b=3.5):
    """Model-based descriptive comparison of the three strategies on ONE candidate pool."""
    tops = {s: rank_candidates(cands, s, min_a, min_b).head(k) for s in STRATEGIES}
    summary = pd.DataFrame(
        {
            "n_selected": {s: len(t) for s, t in tops.items()},
            "mean_avg_pred": {s: t.avg_pred.mean() for s, t in tops.items()},
            "mean_lower_pred": {s: t.min_pred.mean() for s, t in tops.items()},
            "mean_abs_gap": {s: t.gap.mean() for s, t in tops.items()},
        }
    )
    overlap = pd.DataFrame(
        {
            a: {b: len(set(tops[a].movieId) & set(tops[b].movieId)) for b in STRATEGIES}
            for a in STRATEGIES
        }
    )
    return summary, overlap, tops


def select_pairs(ctx, min_train=50, pool_size=1000, min_test_shared=5, max_pairs=500, seed=830):
    """Outcome-blind pair selection.

    Pool: the `pool_size` lowest userIds with >= `min_train` TRAINING ratings (training histories only).
    Eligible pair: at least `min_test_shared` movies rated by both in the TEST split (a coverage count;
    rating values are never consulted). Up to `max_pairs` eligible pairs are drawn with a fixed seed.
    """
    from scipy import sparse

    n_tr = ctx["train"].groupby("userId").size()
    pool = np.array(sorted(n_tr[n_tr >= min_train].index)[:pool_size])
    test = ctx["all"][(ctx["all"].split == "test") & ctx["all"].userId.isin(pool)]
    ui = np.searchsorted(pool, test.userId.to_numpy())
    movie_codes, mi = np.unique(test.movieId.to_numpy(), return_inverse=True)
    B = sparse.csr_matrix(
        (np.ones(len(test), dtype=np.int32), (ui, mi)), shape=(len(pool), len(movie_codes))
    )
    S = sparse.triu(B @ B.T, k=1).tocoo()
    ok = S.data >= min_test_shared
    pairs = pd.DataFrame(
        {"user_a": pool[S.row[ok]], "user_b": pool[S.col[ok]], "n_test_shared": S.data[ok]}
    )
    pairs = pairs.sort_values(["user_a", "user_b"]).reset_index(drop=True)
    n_eligible = len(pairs)
    if n_eligible > max_pairs:
        pairs = pairs.iloc[
            np.sort(np.random.default_rng(seed).choice(n_eligible, max_pairs, replace=False))
        ].reset_index(drop=True)
    return pairs, {
        "pool_users": len(pool),
        "pairs_possible": len(pool) * (len(pool) - 1) // 2,
        "pairs_eligible": n_eligible,
        "pairs_used": len(pairs),
    }


def evaluate_pairs(models_by_name, ctx, pairs, k=3, min_a=3.5, min_b=3.5):
    """For each pair, candidates = TEST movies rated by both users (not excluded for being in training:
    a test rating is by construction not a training rating). Rank the candidates by predicted ratings, then
    judge the top-k by the pair's OBSERVED held-out ratings. Compared with the mean over all candidates
    (what picking k candidates at random would give on average)."""
    test = ctx["all"][ctx["all"].split == "test"]
    rows = []
    for _, p in pairs.iterrows():
        ta = test[test.userId == p.user_a].set_index("movieId").rating
        tb = test[test.userId == p.user_b].set_index("movieId").rating
        obs = pd.DataFrame({"obs_a": ta, "obs_b": tb}).dropna().reset_index()
        obs["obs_avg"] = (obs.obs_a + obs.obs_b) / 2
        obs["obs_min"] = obs[["obs_a", "obs_b"]].min(axis=1)
        pool_row = {
            "user_a": p.user_a,
            "user_b": p.user_b,
            "n_candidates": len(obs),
            "pool_mean_obs_avg": obs.obs_avg.mean(),
            "pool_mean_obs_min": obs.obs_min.mean(),
        }
        for name, model in models_by_name.items():
            c = obs.assign(
                pred_a=m.predict(model, np.full(len(obs), p.user_a), obs.movieId.to_numpy()),
                pred_b=m.predict(model, np.full(len(obs), p.user_b), obs.movieId.to_numpy()),
            )
            c["avg_pred"], c["min_pred"] = (c.pred_a + c.pred_b) / 2, np.minimum(c.pred_a, c.pred_b)
            for s in STRATEGIES:
                top = rank_candidates(c, s, min_a, min_b).head(k)
                pool_row[f"{name}|{s}|n"] = len(top)
                pool_row[f"{name}|{s}|obs_avg"] = top.obs_avg.mean() if len(top) else np.nan
                pool_row[f"{name}|{s}|obs_min"] = top.obs_min.mean() if len(top) else np.nan
        rows.append(pool_row)
    return pd.DataFrame(rows)
