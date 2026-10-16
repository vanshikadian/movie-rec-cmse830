"""Anonymous MovieLens users as viewer profiles.

Demo profiles are chosen by a fixed, outcome-blind rule: for each genre in DEMO_GENRES (in order),
take the lowest userId with at least MIN_TRAIN training ratings and at least MIN_SHARE of them on
movies of that genre, skipping users already chosen. No predictions or strategy results are used.
"""

DEMO_GENRES = ["Drama", "Comedy", "Action", "Sci-Fi", "Romance", "Horror", "Animation", "Fantasy"]
MIN_TRAIN = 100
MIN_SHARE = 0.35
NOT_REAL_GENRES = {"(no genres listed)", "IMAX"}


def describe_user(ctx, user_id):
    """Plain facts about one user's TRAINING history (counts, recorded mean, frequently rated genres)."""
    mine = ctx["train"][ctx["train"].userId == user_id]
    mean = float(mine.rating.mean())
    g = mine.merge(ctx["genres"], on="movieId")
    g = g[~g.genre.isin(NOT_REAL_GENRES)]
    top = (
        (g.groupby("genre").size() / len(mine)).sort_values(ascending=False, kind="stable").head(3)
    )
    pop = mine.movieId.map(ctx["item_train_n"])
    fav = (
        mine.assign(pop=pop)
        .sort_values(["rating", "pop", "movieId"], ascending=[False, False, True])
        .head(3)
        .movieId
    )
    return {
        "userId": int(user_id),
        "n_train": int(len(mine)),
        "train_mean": mean,
        "top_genres": [(k, float(v)) for k, v in top.items()],
        "favourites": [ctx["titles"].loc[m, "title"] for m in fav],
    }


def demo_profile_ids(ctx):
    """[(genre, userId), ...] by the fixed rule above."""
    tr = ctx["train"]
    n = tr.groupby("userId").size()
    active = n[n >= MIN_TRAIN].index
    counts = (
        tr[tr.userId.isin(active)]
        .merge(ctx["genres"], on="movieId")
        .groupby(["userId", "genre"])
        .size()
    )
    share = counts.div(n.reindex(counts.index.get_level_values(0)).to_numpy()).unstack(fill_value=0)
    chosen, taken = [], set()
    for genre in DEMO_GENRES:
        ok = share.index[(share[genre] >= MIN_SHARE) & ~share.index.isin(taken)]
        if len(ok):
            taken.add(int(min(ok)))
            chosen.append((genre, int(min(ok))))
    return chosen
