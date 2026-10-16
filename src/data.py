"""Loading, auditing and preparation helpers for the MovieLens 32M project.

Raw files are never modified; prepared tables are written
to data/processed/.
"""

import hashlib
import json
import random
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ml-32m"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SOURCE_URL = "https://grouplens.org/datasets/movielens/32m/"
ARCHIVE_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
# Official MD5 of ml-32m.zip, read from ARCHIVE_URL + ".md5" (the archive itself was not on disk).
ARCHIVE_MD5 = "d472be332d4daa821edc399621853b57"
# Per-file MD5s published in the official README.txt of the release.
OFFICIAL_FILE_MD5 = {
    "links.csv": "8f033867bcb4e6be8792b21468b4fa6e",
    "movies.csv": "0df90835c19151f9d819d0822e190797",
    "ratings.csv": "cf12b74f9ad4b94a011f079e26d4270a",
    "tags.csv": "963bf4fa4de6b8901868fddd3eb54567",
}

# Official README: 5-star scale, half-star increments (0.5 - 5.0).
RATING_VALUES = np.arange(0.5, 5.01, 0.5)
# README: data span 1995-01-09 .. 2023-10-12, generated 2023-10-13 (UTC epoch seconds).
TS_MIN = int(pd.Timestamp("1995-01-01", tz="UTC").timestamp())
TS_MAX = int(pd.Timestamp("2023-10-14", tz="UTC").timestamp())
YEAR_MIN, YEAR_MAX = 1874, 2023
NO_GENRE = "(no genres listed)"
CHUNK_ROWS = 4_000_000


def file_md5(path, block=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def verify_raw_files(raw_dir=RAW_DIR):
    """Compare local file MD5s with the official README values."""
    out = {}
    for name, expected in OFFICIAL_FILE_MD5.items():
        path = Path(raw_dir) / name
        got = file_md5(path) if path.exists() else None
        out[name] = {"md5": got, "matches_official": got == expected}
    return out


def read_ratings_chunks(raw_dir=RAW_DIR, chunk_rows=CHUNK_ROWS):
    return pd.read_csv(Path(raw_dir) / "ratings.csv", chunksize=chunk_rows)


def parse_release_year(title):
    """Year from a trailing '(YYYY)' only; the title itself is never changed.

    Returns (year or <NA>, status) with status in ok / missing_suffix / out_of_range.
    """
    year = title.str.extract(r"\((\d{4})\)\s*$")[0]
    year = pd.to_numeric(year, errors="coerce")
    status = pd.Series("ok", index=title.index)
    status[year.isna()] = "missing_suffix"
    out_of_range = year.notna() & ~year.between(YEAR_MIN, YEAR_MAX)
    status[out_of_range] = "out_of_range"
    year[out_of_range] = np.nan
    return year.astype("Int16"), status


def prepare_movies(movies_raw):
    movies = movies_raw.rename(columns={"genres": "genres_raw"}).copy()
    movies["release_year"], movies["year_status"] = parse_release_year(movies["title"])
    movies["no_genre_listed"] = movies["genres_raw"].eq(NO_GENRE) | movies["genres_raw"].isna()
    return movies


def explode_genres(movies):
    """One row per movie-genre membership. Movies without genres keep one NO_GENRE row."""
    g = movies[["movieId", "genres_raw"]].copy()
    g["genre"] = g["genres_raw"].fillna(NO_GENRE).str.split("|")
    g = g.explode("genre")[["movieId", "genre"]].reset_index(drop=True)
    return g


def audit_full_source(raw_dir=RAW_DIR, chunk_rows=CHUNK_ROWS):
    """One chunked pass over ALL of ratings.csv. Returns (audit dict, unique user ids).

    Relies on the documented ordering (by userId, then movieId) for duplicate
    detection, and verifies that ordering as it goes.
    """
    movie_ids = pd.read_csv(Path(raw_dir) / "movies.csv", usecols=["movieId"])["movieId"].to_numpy()
    known_movie = np.zeros(int(movie_ids.max()) + 1, dtype=bool)
    known_movie[movie_ids] = True

    n_rows = 0
    nulls = {}
    rating_counts = {}
    bad_increment = bad_range = bad_ts = 0
    ts_min, ts_max = None, None
    dup_pairs = dup_exact = order_violations = orphan_rows = 0
    user_counts = np.zeros(1, dtype=np.int64)
    movie_counts = np.zeros(1, dtype=np.int64)
    prev = None  # last row of previous chunk (userId, movieId, rating, timestamp)

    for chunk in read_ratings_chunks(raw_dir, chunk_rows):
        n_rows += len(chunk)
        for c, v in chunk.isna().sum().items():
            nulls[c] = nulls.get(c, 0) + int(v)
        chunk = chunk.dropna()
        u = chunk["userId"].to_numpy(np.int64)
        m = chunk["movieId"].to_numpy(np.int64)
        r = chunk["rating"].to_numpy(np.float64)
        t = chunk["timestamp"].to_numpy(np.int64)

        for val, cnt in zip(*np.unique(r, return_counts=True)):
            rating_counts[float(val)] = rating_counts.get(float(val), 0) + int(cnt)
        bad_range += int(((r < 0.5) | (r > 5.0)).sum())
        bad_increment += int((np.mod(r * 2, 1) != 0).sum())
        bad_ts += int(((t < TS_MIN) | (t > TS_MAX)).sum())
        ts_min = int(t.min()) if ts_min is None else min(ts_min, int(t.min()))
        ts_max = int(t.max()) if ts_max is None else max(ts_max, int(t.max()))
        orphan_rows += int(
            (~known_movie[np.minimum(m, len(known_movie) - 1)] | (m >= len(known_movie))).sum()
        )

        # prepend the previous chunk's last row so boundary pairs are compared
        if prev is not None:
            u, m, r, t = (np.concatenate([[p], a]) for p, a in zip(prev, (u, m, r, t)))
        same_user = u[1:] == u[:-1]
        same_pair = same_user & (m[1:] == m[:-1])
        dup_pairs += int(same_pair.sum())
        dup_exact += int((same_pair & (r[1:] == r[:-1]) & (t[1:] == t[:-1])).sum())
        order_violations += int(((u[1:] < u[:-1]) | (same_user & (m[1:] < m[:-1]))).sum())
        prev = (u[-1], m[-1], r[-1], t[-1])

        uc = np.bincount(chunk["userId"].to_numpy(np.int64))
        mc = np.bincount(chunk["movieId"].to_numpy(np.int64))
        if len(uc) > len(user_counts):
            user_counts = np.pad(user_counts, (0, len(uc) - len(user_counts)))
        if len(mc) > len(movie_counts):
            movie_counts = np.pad(movie_counts, (0, len(mc) - len(movie_counts)))
        user_counts[: len(uc)] += uc
        movie_counts[: len(mc)] += mc

    user_ids = np.flatnonzero(user_counts)
    per_user = user_counts[user_ids]
    per_movie = movie_counts[movie_counts > 0]
    audit = {
        "scope": "FULL SOURCE: all rows of ratings.csv and movies.csv",
        "ratings_rows": n_rows,
        "null_counts": nulls,
        "rating_value_counts": {str(k): v for k, v in sorted(rating_counts.items())},
        "ratings_out_of_range_0.5_to_5": bad_range,
        "ratings_not_half_star_increment": bad_increment,
        "timestamps_outside_1995_to_2023-10-14": bad_ts,
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "duplicate_user_movie_pairs": dup_pairs,
        "exact_duplicate_rows": dup_exact,
        "sort_order_violations": order_violations,
        "ratings_without_movie_metadata": orphan_rows,
        "distinct_users": int(len(user_ids)),
        "distinct_movies_rated": int(len(per_movie)),
        "ratings_per_user": {
            "min": int(per_user.min()),
            "median": float(np.median(per_user)),
            "max": int(per_user.max()),
        },
        "ratings_per_movie": {
            "min": int(per_movie.min()),
            "median": float(np.median(per_movie)),
            "max": int(per_movie.max()),
        },
    }
    return audit, user_ids


def audit_movies_source(movies_raw, movies):
    return {
        "scope": "FULL SOURCE: all rows of movies.csv",
        "movies_rows": len(movies_raw),
        "null_counts": {k: int(v) for k, v in movies_raw.isna().sum().items()},
        "exact_duplicate_rows": int(movies_raw.duplicated().sum()),
        "movieId_unique": bool(movies_raw["movieId"].is_unique),
        "year_status_counts": movies["year_status"].value_counts().to_dict(),
        "movies_no_genre_listed": int(movies["no_genre_listed"].sum()),
    }


def audit_subset(ratings, movies_subset, genres_subset, movies_full):
    """Checks that cover ONLY the selected-user subset."""
    in_movies = ratings["movieId"].isin(movies_full["movieId"])
    per_user = ratings.groupby("userId").size()
    per_movie = ratings.groupby("movieId").size()
    return {
        "scope": "SUBSET ONLY: ratings of the selected users and the movies they rated",
        "ratings_rows": len(ratings),
        "ratings_dtypes": {k: str(v) for k, v in ratings.dtypes.items()},
        "null_counts": {k: int(v) for k, v in ratings.isna().sum().items()},
        "exact_duplicate_rows": int(ratings.duplicated().sum()),
        "duplicate_user_movie_pairs": int(ratings.duplicated(["userId", "movieId"]).sum()),
        "rating_value_counts": {
            str(k): int(v) for k, v in ratings["rating"].value_counts().sort_index().items()
        },
        "ratings_out_of_range": int((~ratings["rating"].between(0.5, 5.0)).sum()),
        "ratings_not_half_star_increment": int((np.mod(ratings["rating"] * 2, 1) != 0).sum()),
        "timestamps_outside_expected_range": int(
            (~ratings["timestamp"].between(TS_MIN, TS_MAX)).sum()
        ),
        "ratings_without_movie_metadata": int((~in_movies).sum()),
        "distinct_users": int(per_user.size),
        "distinct_movies": int(per_movie.size),
        "ratings_per_user": per_user.describe().round(2).to_dict(),
        "ratings_per_movie": per_movie.describe().round(2).to_dict(),
        "movies_rows": len(movies_subset),
        "movieId_unique": bool(movies_subset["movieId"].is_unique),
        "movies_year_status_counts": movies_subset["year_status"].value_counts().to_dict(),
        "movies_no_genre_listed": int(movies_subset["no_genre_listed"].sum()),
        "genre_membership_rows": len(genres_subset),
        "distinct_genres": int(genres_subset["genre"].nunique()),
        "movies_with_zero_ratings_in_subset": int(
            (~movies_subset["movieId"].isin(per_movie.index)).sum()
        ),
    }


def sample_user_ids(all_user_ids, n_users, seed):
    """Uniform sample without replacement from the sorted unique user IDs."""
    ids = sorted(int(i) for i in all_user_ids)
    if n_users >= len(ids):
        return ids
    return sorted(random.Random(seed).sample(ids, n_users))


def read_user_ratings(user_ids, raw_dir=RAW_DIR, chunk_rows=CHUNK_ROWS):
    """All recorded ratings for the given users, read in chunks."""
    keep = set(user_ids)
    parts = [c[c["userId"].isin(keep)] for c in read_ratings_chunks(raw_dir, chunk_rows)]
    return pd.concat(parts, ignore_index=True)


def resolve_duplicates(ratings):
    """Explicit rule, only used if duplicate user-movie pairs exist: keep the latest timestamp
    (ties: last row in file order). Returns (resolved ratings, number of rows dropped)."""
    out = ratings.sort_values(["userId", "movieId", "timestamp"], kind="stable")
    out = out.drop_duplicates(["userId", "movieId"], keep="last")
    return out.reset_index(drop=True), len(ratings) - len(out)


def load_ratings(processed_dir=PROCESSED_DIR):
    return pd.read_parquet(Path(processed_dir) / "ratings.parquet")


def load_movies(processed_dir=PROCESSED_DIR):
    return pd.read_parquet(Path(processed_dir) / "movies.parquet")


def load_movie_genres(processed_dir=PROCESSED_DIR):
    return pd.read_parquet(Path(processed_dir) / "movie_genres.parquet")


def load_json(name, processed_dir=PROCESSED_DIR):
    with open(Path(processed_dir) / name) as f:
        return json.load(f)


def find_first_eligible_pair(ratings, min_ratings=100, pool_size=200, min_shared=20):
    """Deterministic, outcome-blind pair search.

    Pool: the `pool_size` lowest user IDs having >= `min_ratings` ratings in the data.
    Pairs are scanned in ascending (userA, userB) ID order; the first pair with
    >= `min_shared` co-rated movies is returned. Agreement is never consulted.
    """
    counts = ratings.groupby("userId").size()
    pool = sorted(counts[counts >= min_ratings].index)[:pool_size]
    movies_by_user = ratings[ratings["userId"].isin(pool)].groupby("userId")["movieId"].agg(set)
    n_pairs = n_eligible = 0
    first = None
    for a, b in combinations(pool, 2):
        n_pairs += 1
        shared = len(movies_by_user[a] & movies_by_user[b])
        if shared >= min_shared:
            n_eligible += 1
            if first is None:
                first = (int(a), int(b), shared)
    return {
        "pool_users": len(pool),
        "pairs_scanned": n_pairs,
        "pairs_eligible": n_eligible,
        "pair": first,
        "pool_ids_range": (int(pool[0]), int(pool[-1])) if pool else None,
    }
