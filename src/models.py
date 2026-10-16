"""Rating-prediction models: split, baselines, matrix factorization, metrics.

All models share one plain representation (a dict of numpy arrays) so a single
`predict` function serves the global mean, the bias model and matrix factorization:

    r_hat(u, i) = mu + bu[u] + bi[i] + P[u] . Q[i]      (clipped to [0.5, 5.0])

Unknown users contribute no user terms and unknown items no item terms, so an
unseen item is predicted from mu + bu[u] only.
"""

import json
from pathlib import Path

import numpy as np

RATING_MIN, RATING_MAX = 0.5, 5.0
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def make_split(ratings, seed=830, val_frac=0.1, test_frac=0.2):
    """Random split WITHIN each user (rows of one user are shuffled with a seeded generator).

    Every user keeps ~70% of their ratings for training. This estimates how well held-out
    ratings of known users and mostly-known movies are predicted; it is NOT a chronological
    (future-forecasting) split. `ratings` must be in a fixed order (sorted by userId, movieId).
    """
    df = ratings[["userId", "movieId"]].copy()
    df["key"] = np.random.default_rng(seed).random(len(df))
    rank = df.groupby("userId")["key"].rank(method="first")
    n = df.groupby("userId")["key"].transform("size")
    n_test, n_val = np.round(test_frac * n), np.round(val_frac * n)
    df["split"] = np.where(rank <= n_test, "test", np.where(rank <= n_test + n_val, "val", "train"))
    return df[["userId", "movieId", "split"]]


def split_frames(ratings, split):
    merged = ratings.merge(split, on=["userId", "movieId"], validate="one_to_one")
    if len(merged) != len(ratings) or len(split) != len(ratings):
        raise ValueError("The split must contain exactly one row for every rating")
    if not merged["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Split labels must be train, val or test")
    return {
        name: merged[merged["split"] == name].drop(columns="split").reset_index(drop=True)
        for name in ("train", "val", "test")
    }


def fit_global_mean(train):
    """Baseline A: every prediction is the training mean."""
    users, items = np.sort(train.userId.unique()), np.sort(train.movieId.unique())
    return {
        "kind": "global_mean",
        "mu": float(train.rating.mean()),
        "user_ids": users,
        "item_ids": items,
        "bu": np.zeros(len(users)),
        "bi": np.zeros(len(items)),
        "P": np.zeros((len(users), 0)),
        "Q": np.zeros((len(items), 0)),
    }


def fit_bias(train, lam_item, lam_user):
    """Baseline B: regularized user/item biases (damped means, fitted on training data only).

    b_i = sum(r - mu) / (lam_item + n_i);  b_u = sum(r - mu - b_i) / (lam_user + n_u)
    """
    model = fit_global_mean(train)
    mu = model["mu"]
    g = train.groupby("movieId")["rating"].agg(["sum", "size"])
    b_i = (g["sum"] - g["size"] * mu) / (lam_item + g["size"])
    resid = train["rating"] - mu - train["movieId"].map(b_i)
    h = resid.groupby(train["userId"]).agg(["sum", "size"])
    b_u = h["sum"] / (lam_user + h["size"])
    model.update(
        kind="bias",
        lam_item=lam_item,
        lam_user=lam_user,
        bu=b_u.reindex(model["user_ids"]).to_numpy(),
        bi=b_i.reindex(model["item_ids"]).to_numpy(),
    )
    return model


def fit_mf(train, n_factors, reg, n_epochs=20, lr=0.005, seed=830):
    """Biased matrix factorization trained by SGD (scikit-surprise `SVD`) on observed ratings only.

    Missing user-movie pairs are never used as targets. `reg` is surprise's `reg_all`
    (L2 on biases and factors).
    """
    from surprise import SVD, Dataset, Reader  # imported here so the app does not need surprise

    data = Dataset.load_from_df(
        train[["userId", "movieId", "rating"]], Reader(rating_scale=(RATING_MIN, RATING_MAX))
    )
    trainset = data.build_full_trainset()
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr, reg_all=reg, random_state=seed)
    algo.fit(trainset)
    users, items = np.sort(train.userId.unique()), np.sort(train.movieId.unique())
    iu = np.array([trainset.to_inner_uid(int(u)) for u in users])
    ii = np.array([trainset.to_inner_iid(int(i)) for i in items])
    return {
        "kind": "mf",
        "mu": float(trainset.global_mean),
        "user_ids": users,
        "item_ids": items,
        "bu": algo.bu[iu].astype(float),
        "bi": algo.bi[ii].astype(float),
        "P": algo.pu[iu].astype(float),
        "Q": algo.qi[ii].astype(float),
        "n_factors": n_factors,
        "reg": reg,
        "n_epochs": n_epochs,
        "lr": lr,
        "seed": seed,
    }, algo


def _lookup(ids, wanted):
    wanted = np.asarray(wanted)
    pos = np.searchsorted(ids, wanted)
    pos_c = np.minimum(pos, len(ids) - 1)
    return pos_c, (pos < len(ids)) & (ids[pos_c] == wanted)


def predict(model, user_ids, item_ids, clip=True):
    """Predictions for aligned arrays of (user, item) pairs."""
    pu, ku = _lookup(model["user_ids"], user_ids)
    pi, ki = _lookup(model["item_ids"], item_ids)
    est = np.full(len(pu), model["mu"], dtype=float)
    est += np.where(ku, model["bu"][pu], 0.0) + np.where(ki, model["bi"][pi], 0.0)
    if model["P"].shape[1]:
        est += np.where(ku & ki, np.einsum("ij,ij->i", model["P"][pu], model["Q"][pi]), 0.0)
    return np.clip(est, RATING_MIN, RATING_MAX) if clip else est


def predict_user_all_items(model, user_id, clip=True):
    """One trained user's prediction for every movie in the model (vectorised)."""
    pu, ku = _lookup(model["user_ids"], [user_id])
    if not ku[0]:
        raise KeyError(f"user {user_id} is not a trained profile")
    est = model["mu"] + model["bu"][pu[0]] + model["bi"]
    if model["P"].shape[1]:
        est = est + model["Q"] @ model["P"][pu[0]]
    return np.clip(est, RATING_MIN, RATING_MAX) if clip else est


def evaluate(model, df):
    """MAE/RMSE on clipped predictions, with counts and warm-start coverage."""
    _, ku = _lookup(model["user_ids"], df.userId.to_numpy())
    _, ki = _lookup(model["item_ids"], df.movieId.to_numpy())
    err = predict(model, df.userId.to_numpy(), df.movieId.to_numpy()) - df.rating.to_numpy()
    warm = ku & ki
    stats = lambda e: (
        {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e**2).mean()))}
        if len(e)
        else {"mae": None, "rmse": None}
    )
    return {
        "n_ratings": int(len(df)),
        "n_unseen_users": int((~ku).sum()),
        "n_unseen_items": int((~ki).sum()),
        "warm_coverage": float(warm.mean()),
        "all": stats(err),
        "warm_only": stats(err[warm]),
        "unseen_item_only": stats(err[~ki]),
    }


def save_model(model, path):
    arrays = {k: v for k, v in model.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in model.items() if not isinstance(v, np.ndarray)}
    np.savez_compressed(path, meta_json=np.array(json.dumps(meta)), **arrays)


def load_model(path):
    with np.load(path, allow_pickle=False) as z:
        model = {k: z[k] for k in z.files if k != "meta_json"}
        model.update(json.loads(str(z["meta_json"])))
    return model
