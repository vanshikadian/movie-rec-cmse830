"""Headless checks of the Streamlit app (streamlit.testing AppTest) against the group functions.

    python3 scripts/check_app.py
"""

import sys
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data as d, group as g, models as m, profiles as pr  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok: {msg}")


def run(at):
    at.run(timeout=120)
    check(not at.exception, "no exception raised")
    return at


ctx = g.build_context(
    d.load_ratings(),
    pd.read_parquet(d.PROCESSED_DIR / "split.parquet"),
    d.load_movies(),
    d.load_movie_genres(),
)
model = m.load_model(m.MODELS_DIR / "matrix_factorization.npz")
demos = pr.demo_profile_ids(ctx)
by_genre = {genre: u for genre, u in demos}
A, B = by_genre["Action"], by_genre["Romance"]  # the app's default pair
widget = lambda items, label: next(w for w in items if w.label.startswith(label))
table = lambda at: next(df.value for df in at.dataframe if "Movie ID" in df.value.columns)

print(f"featured profiles by fixed rule: {demos}")
print(f"default view (users {A} & {B}, average) vs group functions")
at = run(AppTest.from_file(str(ROOT / "app.py"), default_timeout=120))
check(
    len(demos) == 8
    and widget(at.selectbox, "Viewer A").options
    and len(widget(at.selectbox, "Viewer A").options) == 8,
    "eight featured profiles offered",
)
exp = g.rank_candidates(g.build_candidates(model, ctx, A, B), "average").head(10)
check(
    table(at)["Movie ID"].tolist() == exp.movieId.tolist(),
    "ranked movie IDs equal src.group output",
)
check((table(at)["Predicted A"].values == exp.pred_a.round(2).values).all(), "predictions match")
check(
    len(at.tabs) == 4 and any("recorded ratings" in m_.value for m_ in at.markdown),
    "four tabs and profile summaries rendered",
)
page = " ".join(m_.value for m_ in at.markdown)
check(
    all(w not in page for w in ("tough", "generous", "not real people")),
    "no judgmental rater labels or the old profile wording",
)
check(
    "Explore two anonymized MovieLens profiles. Scores are model estimates." in page,
    "required intro sentence present",
)
import html as _h, re as _re

_m = _re.match(r"^(.*?)\s*\(\d{4}\)$", exp.title.iloc[0])
first_title = _m.group(1) if _m else exp.title.iloc[0]
check(
    _h.escape(first_title) in page,
    "top recommendation title appears in the rendered recommendation rows",
)

print("switch strategy and thresholds")
widget(at.radio, "Choose how to balance").set_value("lower")
run(at)
exp = g.rank_candidates(g.build_candidates(model, ctx, A, B), "lower").head(10)
check(table(at)["Movie ID"].tolist() == exp.movieId.tolist(), "lower-score ranking matches")
widget(at.radio, "Choose how to balance").set_value("threshold")
run(at)
widget(at.slider, "Minimum predicted rating, viewer A").set_value(4.5)
widget(at.slider, "Minimum predicted rating, viewer B").set_value(4.5)
run(at)
check(
    any("No movie has a predicted rating" in w.value for w in at.warning),
    "empty threshold result shows an honest empty state",
)
widget(at.slider, "Minimum predicted rating, viewer A").set_value(3.5)
widget(at.slider, "Minimum predicted rating, viewer B").set_value(3.5)
run(at)
exp = g.rank_candidates(g.build_candidates(model, ctx, A, B), "threshold", 3.5, 3.5).head(10)
check(table(at)["Movie ID"].tolist() == exp.movieId.tolist(), "threshold ranking matches")

print("identical users")
widget(at.radio, "Choose how to balance").set_value("average")
widget(at.selectbox, "Viewer B").set_value(A)
run(at)
check(any("same profile" in i.value for i in at.info), "identical-user notice shown")
widget(at.selectbox, "Viewer B").set_value(B)

print("filters and exclusion mode")
at.multiselect[0].set_value(["Film-Noir", "Western"])
widget(at.slider, "Minimum training ratings").set_value(100)
run(at)
n = len(
    g.build_candidates(model, ctx, A, B, min_item_train=100, genre_filter=["Film-Noir", "Western"])
)
check(
    (
        len(table(at)) == min(10, n)
        if n
        else any("No candidate movies remain" in w.value for w in at.warning)
    ),
    f"genre filter handled ({n} candidates)",
)
widget(at.radio, "Hide movies").set_value("All recorded ratings (discovery mode)")
run(at)
n2 = len(
    g.build_candidates(
        model,
        ctx,
        A,
        B,
        exclude="all_recorded",
        min_item_train=100,
        genre_filter=["Film-Noir", "Western"],
    )
)
check(
    (
        len(table(at)) == min(10, n2)
        if n2
        else any("No candidate movies remain" in w.value for w in at.warning)
    ),
    f"discovery exclusion mode handled ({n2} candidates)",
)
widget(at.radio, "Hide movies").set_value("Training histories only (modelling default)")
at.multiselect[0].set_value(["IMAX"])
widget(at.slider, "Minimum training ratings").set_value(100)
run(at)
n3 = len(g.build_candidates(model, ctx, A, B, min_item_train=100, genre_filter=["IMAX"]))
if n3 == 0:
    check(
        any("No candidate movies remain" in w.value for w in at.warning),
        "empty candidate pool shows a message",
    )
else:
    print(
        f"  (IMAX with >=100 training ratings leaves {n3} candidates; empty-pool message not triggered in this check)"
    )

print("browse all profiles")
at.multiselect[0].set_value([])
widget(at.slider, "Minimum training ratings").set_value(10)
at.checkbox[0].set_value(True)
run(at)
check(len(widget(at.selectbox, "Viewer A").options) == 5000, "all 5,000 profiles selectable")
print("done")
