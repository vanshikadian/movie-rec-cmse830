"""Prepare the MovieLens 32M development subset.

    python scripts/prepare_data.py [--n-users 5000] [--seed 830]
                                   [--raw-dir data/raw/ml-32m] [--out-dir data/processed]
                                   [--report docs/data_quality.md]
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as d  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-users", type=int, default=5000)
    p.add_argument("--seed", type=int, default=830)
    p.add_argument("--raw-dir", type=Path, default=d.RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=d.PROCESSED_DIR)
    p.add_argument("--report", type=Path, default=d.PROJECT_ROOT / "docs" / "data_quality.md")
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    print("1. verifying raw files against official MD5s")
    md5 = d.verify_raw_files(a.raw_dir)
    for name, r in md5.items():
        print(f"  {name}: {'match' if r['matches_official'] else 'MISMATCH or missing'}")
    if not all(r["matches_official"] for r in md5.values()):
        sys.exit(
            "Raw files missing or differ from the official release; see README.md for download steps."
        )

    archive = a.raw_dir.parent / "ml-32m.zip"
    archive_verified = archive.is_file() and d.file_md5(archive) == d.ARCHIVE_MD5
    if archive.is_file() and not archive_verified:
        sys.exit("MovieLens archive checksum mismatch.")

    print("2. full-source audit (chunked pass over all ratings)")
    full_audit, all_users = d.audit_full_source(a.raw_dir)
    movies_raw = pd.read_csv(a.raw_dir / "movies.csv")
    movies_all = d.prepare_movies(movies_raw)
    movies_audit = d.audit_movies_source(movies_raw, movies_all)

    print(f"3. sampling {a.n_users} of {len(all_users)} users (seed {a.seed})")
    selected = d.sample_user_ids(all_users, a.n_users, a.seed)
    ratings = d.read_user_ratings(selected, a.raw_dir)
    dup_info = {"dropped_by_resolution_rule": 0}
    if ratings.isna().any().any():
        sys.exit("Missing values in selected ratings; decide an explicit rule before continuing.")
    if ratings.duplicated(["userId", "movieId"]).any():
        ratings, dup_info["dropped_by_resolution_rule"] = d.resolve_duplicates(ratings)
    ratings = ratings.astype(
        {"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"}
    )
    ratings = ratings.sort_values(["userId", "movieId"]).reset_index(drop=True)

    movies = movies_all[movies_all["movieId"].isin(ratings["movieId"].unique())].reset_index(
        drop=True
    )
    movies["movieId"] = movies["movieId"].astype("int32")
    genres = d.explode_genres(movies)

    print("4. subset audit and verification")
    sub_audit = d.audit_subset(ratings, movies, genres, movies_all)
    sub_audit["duplicate_resolution"] = dup_info
    # many-to-one join check on IDs only (no wide expanded table)
    joined = ratings[["movieId"]].merge(
        movies[["movieId"]], on="movieId", how="left", validate="many_to_one", indicator=True
    )
    check(
        len(joined) == len(ratings) and joined["_merge"].eq("both").all(),
        "ratings->movies merge is many-to-one and keeps every rating row",
    )
    check(
        ratings["userId"].nunique() == len(selected) and set(ratings["userId"]) == set(selected),
        "every selected user present in subset",
    )
    expected = full_audit["ratings_rows"]
    check(
        len(ratings)
        == len(d.read_user_ratings(selected, a.raw_dir)) - dup_info["dropped_by_resolution_rule"],
        "subset rows equal a fresh read of the selected users' rows",
    )
    check(len(ratings) <= expected, "subset row count does not exceed the source row count")
    check(ratings["rating"].isin(d.RATING_VALUES).all(), "ratings are 0.5-5.0 in half-star steps")
    check(movies["movieId"].is_unique, "movieId unique in movies table")
    check(
        genres["movieId"].nunique() == len(movies),
        "genre expansion keeps every movie exactly represented",
    )
    check(
        len(genres) == movies["genres_raw"].fillna(d.NO_GENRE).str.count(r"\|").add(1).sum(),
        "genre rows equal sum of pipe-separated memberships",
    )
    check(
        set(movies["movieId"]) == set(ratings["movieId"]), "movies table = movies rated in subset"
    )

    print("5. writing outputs")
    ratings.to_parquet(a.out_dir / "ratings.parquet", index=False)
    movies.to_parquet(a.out_dir / "movies.parquet", index=False)
    genres.to_parquet(a.out_dir / "movie_genres.parquet", index=False)
    pd.Series(selected, name="userId").to_csv(a.out_dir / "selected_user_ids.csv", index=False)
    config = {
        "dataset": "MovieLens 32M (ml-32m)",
        "source_url": d.SOURCE_URL,
        "archive_url": d.ARCHIVE_URL,
        "official_archive_md5": d.ARCHIVE_MD5,
        "archive_checksum_verified_locally": archive_verified,
        "archive_note": (
            "Archive MD5 verified."
            if archive_verified
            else "Archive unavailable; per-file MD5s verified instead."
        ),
        "file_md5": md5,
        "prepared_on": date.today().isoformat(),
        "n_users_requested": a.n_users,
        "seed": a.seed,
        "sampling_method": "random.Random(seed).sample(sorted unique userIds from full ratings.csv, n_users), then sorted",
        "n_source_users": len(all_users),
        "n_selected_users": len(selected),
    }
    for name, obj in [
        ("sampling_config.json", config),
        ("audit_full_source.json", {"ratings": full_audit, "movies": movies_audit}),
        ("audit_subset.json", sub_audit),
    ]:
        (a.out_dir / name).write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))

    write_report(a.report, config, full_audit, movies_audit, sub_audit)
    print(f"done: {len(ratings):,} ratings, {len(movies):,} movies, {len(genres):,} genre rows")


def fmt(x):
    return f"{x:,}" if isinstance(x, int) else str(x)


def write_report(path, cfg, full, mov, sub):
    ts = lambda t: pd.to_datetime(t, unit="s", utc=True).strftime("%Y-%m-%d")
    L = []
    L += [
        "# Data quality report",
        "",
        f"Generated by `scripts/prepare_data.py` on {cfg['prepared_on']}.",
        "",
        "## Scope of each check",
        "",
        "- **Full source**: one chunked pass over every row of `ratings.csv` (and all of `movies.csv`).",
        f"- **Subset**: {cfg['n_selected_users']:,} users sampled uniformly (seed {cfg['seed']}) with all of their ratings, "
        "plus metadata for the movies they rated. Subset checks do **not** describe the other users.",
        "- Not audited: `links.csv`, `tags.csv` (preserved only; MD5 verified).",
        "",
        "## Source identity",
        "",
        f"- Release: MovieLens 32M, {cfg['source_url']}",
        f"- Official archive MD5 (from `{cfg['archive_url']}.md5`): `{cfg['official_archive_md5']}`. "
        + (
            "Archive MD5 verified locally."
            if cfg["archive_checksum_verified_locally"]
            else "Archive unavailable; its MD5 was not recomputed locally."
        ),
        "- Per-file MD5 vs. official README: "
        + ", ".join(
            f"{k} {'match' if v['matches_official'] else 'MISMATCH'}"
            for k, v in cfg["file_md5"].items()
        ),
        f"- Data prepared on: {cfg['prepared_on']} (original retrieval date not recorded; files carry a 2023-10-13 modification date).",
        "",
        "## Full-source checks",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Ratings rows | {fmt(full['ratings_rows'])} |",
        f"| Distinct users / movies rated | {fmt(full['distinct_users'])} / {fmt(full['distinct_movies_rated'])} |",
        f"| Movies rows in movies.csv (of which never rated) | {fmt(mov['movies_rows'])} ({fmt(mov['movies_rows'] - full['distinct_movies_rated'])}) |",
        f"| Missing values (ratings) | {full['null_counts']} |",
        f"| Missing values (movies) | {mov['null_counts']} |",
        f"| Duplicate user-movie pairs (adjacent-row test; sort order verified) | {fmt(full['duplicate_user_movie_pairs'])} |",
        f"| Exact duplicate rows | {fmt(full['exact_duplicate_rows'])} |",
        f"| Sort-order violations (userId, movieId) | {fmt(full['sort_order_violations'])} |",
        f"| Ratings outside 0.5-5.0 / not half-star | {fmt(full['ratings_out_of_range_0.5_to_5'])} / {fmt(full['ratings_not_half_star_increment'])} |",
        f"| Timestamps outside 1995-01-01..2023-10-14 | {fmt(full['timestamps_outside_1995_to_2023-10-14'])} (range {ts(full['timestamp_min'])} to {ts(full['timestamp_max'])}) |",
        f"| Ratings with no row in movies.csv | {fmt(full['ratings_without_movie_metadata'])} |",
        f"| movieId unique in movies.csv | {mov['movieId_unique']} |",
        f"| Exact duplicate movie rows | {fmt(mov['exact_duplicate_rows'])} |",
        f"| Ratings per user (min / median / max) | {full['ratings_per_user']['min']} / {full['ratings_per_user']['median']} / {full['ratings_per_user']['max']} |",
        f"| Ratings per movie (min / median / max) | {full['ratings_per_movie']['min']} / {full['ratings_per_movie']['median']} / {full['ratings_per_movie']['max']} |",
        f"| Release-year parse status (all movies) | {mov['year_status_counts']} |",
        f"| Movies with `(no genres listed)` | {fmt(mov['movies_no_genre_listed'])} |",
        "",
        "## Subset checks",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| Ratings rows (selected users' complete histories) | {fmt(sub['ratings_rows'])} |",
        f"| Distinct users / movies | {fmt(sub['distinct_users'])} / {fmt(sub['distinct_movies'])} |",
        f"| Column types | {sub['ratings_dtypes']} |",
        f"| Missing values | {sub['null_counts']} |",
        f"| Exact duplicate rows / duplicate user-movie pairs | {sub['exact_duplicate_rows']} / {sub['duplicate_user_movie_pairs']} |",
        f"| Duplicate resolution rule applied (rows dropped) | {sub['duplicate_resolution']['dropped_by_resolution_rule']} |",
        f"| Ratings out of range / not half-star | {sub['ratings_out_of_range']} / {sub['ratings_not_half_star_increment']} |",
        f"| Timestamps outside expected range | {sub['timestamps_outside_expected_range']} |",
        f"| Ratings without movie metadata | {sub['ratings_without_movie_metadata']} |",
        f"| Ratings per user (describe) | {sub['ratings_per_user']} |",
        f"| Ratings per movie (describe) | {sub['ratings_per_movie']} |",
        f"| Movies rows / movieId unique | {fmt(sub['movies_rows'])} / {sub['movieId_unique']} |",
        f"| Release-year parse status (subset movies) | {sub['movies_year_status_counts']} |",
        f"| Subset movies with `(no genres listed)` | {fmt(sub['movies_no_genre_listed'])} |",
        f"| Genre membership rows / distinct genre labels | {fmt(sub['genre_membership_rows'])} / {sub['distinct_genres']} |",
        "",
        "## Decisions and notes",
        "",
        "- Nothing was deleted or repaired. IDs and titles are kept exactly as supplied; `release_year` is a new column parsed only from a trailing `(YYYY)` "
        f"and set to missing (with `year_status`) otherwise or if outside {d.YEAR_MIN}-{d.YEAR_MAX}.",
        "- If duplicate user-movie pairs were ever found, the rule is: keep the latest timestamp (ties: last file row) and report the number dropped (row above).",
        "- Missing user-movie ratings are unobserved, not zero and not dislikes.",
        "- The official README lists the genre label `Children's`, but `movies.csv` uses `Children`; labels are kept as found.",
        "- `(no genres listed)` is kept as an explicit genre label so every movie keeps at least one membership row.",
        "- The movie-genre table has one row per membership, so its row count exceeds the movie count; ratings are never joined to it row-by-row.",
        "- Ratings-to-movies validity is checked with an ID-only many-to-one merge; no wide joined table is stored.",
        "",
    ]
    Path(path).write_text("\n".join(L))


if __name__ == "__main__":
    main()
