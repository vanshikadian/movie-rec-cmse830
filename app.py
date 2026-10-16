"""Both Pick: movie picks for two people. Local Streamlit app that uses saved models only.

    streamlit run app.py
"""

import html
import json
import re

import altair as alt
import pandas as pd
import streamlit as st

from src import data as d
from src import group as g
from src import models as m
from src import plots as pl
from src import profiles as pr

LABELS = {
    "average": "Highest average",
    "lower": "Prioritize the lower score",
    "threshold": "Both meet a minimum",
}
EXPLAIN = {
    "average": "Ranks movies by the mean of your two predicted ratings.",
    "lower": "Ranks movies by whichever of you is predicted to like them less.",
    "threshold": "Keeps movies each of you is predicted to rate at or above your minimum, then ranks them by the mean.",
}

st.set_page_config(page_title="Both Pick", layout="wide")
st.html(d.PROJECT_ROOT / "assets" / "both-pick.css")
st.markdown(
    """
<header class="bp-header">
<p class="bp-kicker">A double feature of your tastes</p>
<h1 class="bp-mark">BOTH <b>PICK</b><i>!</i></h1>
<div class="bp-billing"><p>Two tastes.<br><em>One movie.</em></p>
<p class="bp-credit">THE NEXT GREAT PICK<br>STARRING VIEWER A &amp; VIEWER B</p></div>
</header>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_everything():
    needed = [
        d.PROCESSED_DIR / f"{name}.parquet"
        for name in ("ratings", "movies", "movie_genres", "split")
    ]
    needed += [m.MODELS_DIR / "matrix_factorization.npz", m.MODELS_DIR / "metrics.json"]
    if any(not p.exists() for p in needed):
        return None
    ratings, movies, genres = d.load_ratings(), d.load_movies(), d.load_movie_genres()
    split = pd.read_parquet(d.PROCESSED_DIR / "split.parquet")
    ctx = g.build_context(ratings, split, movies, genres)
    model = m.load_model(m.MODELS_DIR / "matrix_factorization.npz")
    metrics = json.loads((m.MODELS_DIR / "metrics.json").read_text())
    return ctx, model, metrics, pr.demo_profile_ids(ctx)


@st.cache_data
def candidates(user_a, user_b, exclude, min_item_train, genre_filter):
    return g.build_candidates(
        MODEL, CTX, user_a, user_b, exclude, min_item_train, list(genre_filter) or None
    )


@st.cache_data
def describe(user_id):
    return pr.describe_user(CTX, user_id)


def split_title(title):
    found = re.match(r"^(.*?)\s*\((\d{4})\)$", str(title))
    return (found.group(1), found.group(2)) if found else (str(title), "")


loaded = load_everything()
if loaded is None:
    st.error(
        "Data or model files are missing. Run `python3 scripts/prepare_data.py` and `python3 scripts/train_models.py` first (see README)."
    )
    st.stop()
CTX, MODEL, METRICS, DEMOS = loaded
st.markdown(
    '<div class="bp-note">Explore two anonymized MovieLens profiles. Scores are model estimates.</div><div class="bp-note">Profiles represent anonymized real MovieLens users. Genre frequency describes recorded history, not someone’s complete taste.</div>',
    unsafe_allow_html=True,
)

# Viewer controls
users = [int(u) for u in MODEL["user_ids"]]
featured = {u: genre for genre, u in DEMOS}
options = users if st.session_state.get("browse_all", False) else [u for _, u in DEMOS]
label = lambda u: (
    f"Profile {u} (often rates {featured[u].lower()})" if u in featured else f"Profile {u}"
)
genre_order = [genre for genre, _ in DEMOS]
default_a = genre_order.index("Action") if "Action" in genre_order else 0
default_b = genre_order.index("Romance") if "Romance" in genre_order else 1


def viewer(col, name, color, default_index):
    with col:
        user = st.selectbox(
            f"Viewer {name}", options, index=min(default_index, len(options) - 1), format_func=label
        )
        info = describe(user)
        genres_txt = " · ".join(f"{k} {v:.0%}" for k, v in info["top_genres"])
        st.markdown(
            f'<div class="bp-facts" style="--c:{color}"><p>{info["n_train"]} recorded ratings · mean recorded rating {info["train_mean"]:.1f} / 5</p>'
            f"<p>Frequently rates {html.escape(genres_txt)}</p></div>",
            unsafe_allow_html=True,
        )
        if info["n_train"] < g.SHORT_HISTORY:
            st.caption(
                f"Short history (under {g.SHORT_HISTORY} ratings): these estimates have limited history to learn from."
            )
        with st.expander("Highest-rated movies in this history"):
            for t in info["favourites"]:
                st.write(t)
    return user


c1, c2 = st.columns(2, gap="medium")
user_a = viewer(c1, "A", pl.VIEWER_A, default_a)
user_b = viewer(c2, "B", pl.VIEWER_B, default_b)
if user_a == user_b:
    st.info(
        "Both viewers are the same profile, so both predictions are identical. Minimum thresholds can still remove movies."
    )

# Ranking controls
s1, s2 = st.columns([5, 2], gap="large")
with s1:
    strategy = st.radio(
        "Choose how to balance your tastes.", g.STRATEGIES, format_func=LABELS.get, horizontal=True
    )
    st.caption(EXPLAIN[strategy])
    min_a = min_b = 3.5
    if strategy == "threshold":
        t1, t2 = st.columns(2)
        min_a = t1.slider("Minimum predicted rating, viewer A", 0.5, 5.0, 3.5, 0.25)
        min_b = t2.slider("Minimum predicted rating, viewer B", 0.5, 5.0, 3.5, 0.25)
with s2:
    genre_pick = st.multiselect(
        "Only these genres (optional)", sorted(CTX["genres"].genre.unique())
    )
    with st.popover("Advanced settings"):
        st.checkbox("Browse all 5,000 profiles instead of the 8 featured ones", key="browse_all")
        exclude_label = st.radio(
            "Hide movies either of you has already rated",
            [
                "Training histories only (modelling default)",
                "All recorded ratings (discovery mode)",
            ],
            help="The second option also hides movies from the ratings held out for testing, which suits a real 'what should we watch' use.",
        )
        min_item = st.slider(
            "Minimum training ratings per movie",
            1,
            100,
            10,
            help="Movies with very few ratings have unreliable estimates.",
        )
        top_n = st.slider("Number of picks", 5, 25, 10)

exclude = "training" if exclude_label.startswith("Training") else "all_recorded"

try:
    cands = candidates(user_a, user_b, exclude, min_item, tuple(genre_pick))
except ValueError as e:
    st.error(str(e))
    st.stop()
ranked = g.rank_candidates(cands, strategy, min_a, min_b)

# Recommendation views
tab_picks, tab_scatter, tab_compare, tab_details = st.tabs(
    ["Recommendations", "Preference scatter", "Strategy comparison", "Details"]
)

with tab_picks:
    if len(cands) == 0:
        st.warning(
            "No candidate movies remain after the exclusions and filters. Loosen the genre filter or lower the minimum training ratings."
        )
    elif len(ranked) == 0:
        st.warning(
            f"No movie has a predicted rating of at least {min_a} for viewer A and {min_b} for viewer B. Lower a minimum to see results; nothing is shown as a fallback."
        )
    else:
        st.markdown('<h2 class="bp-section">Your next watch.</h2>', unsafe_allow_html=True)
        rows = []
        for _, r in ranked.head(top_n).iterrows():
            title, year = split_title(r["title"])
            rows.append(
                f'<tr><td class="rk">{int(r["rank"])}</td>'
                f'<td><span class="ttl">{html.escape(title)}</span><span class="yr">{year}</span><div class="gn">{html.escape(str(r["genres"]))}</div></td>'
                f'<td class="num ca">{r["pred_a"]:.1f} <small>/ 5</small></td><td class="num cb">{r["pred_b"]:.1f} <small>/ 5</small></td>'
                f'<td class="num">{r["score"]:.1f} <small>/ 5</small></td></tr>'
            )
        st.markdown(
            f'<table class="bp-picks"><caption style="text-align:left;color:#62584D;font-size:.85rem;caption-side:top">'
            f"{len(cands):,} candidate movies, the same pool for every strategy. Predicted ratings are model estimates.</caption>"
            '<thead><tr><th scope="col">#</th><th scope="col">Movie</th><th scope="col" class="num">Viewer A</th><th scope="col" class="num">Viewer B</th>'
            f'<th scope="col" class="num" title="{html.escape(LABELS[strategy])}: the value the ranking sorts by">Score</th></tr></thead><tbody>'
            + "".join(rows)
            + "</tbody></table>",
            unsafe_allow_html=True,
        )
        if len(ranked) < top_n:
            st.caption(f"Only {len(ranked)} movies qualify.")

with tab_scatter:
    if len(cands) == 0:
        st.info("Nothing to plot: there are no candidate movies.")
    else:
        both_high = int(((cands.pred_a >= 4) & (cands.pred_b >= 4)).sum())
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.altair_chart(
                pl.theme(
                    pl.pair_scatter(
                        cands,
                        ranked.head(top_n),
                        user_a,
                        user_b,
                        f"{LABELS[strategy]}: top {min(top_n, len(ranked))}",
                        min_a if strategy == "threshold" else None,
                        min_b if strategy == "threshold" else None,
                    )
                ),
                theme=None,
                width="content",
            )
        with right:
            st.markdown("**Reading the plot**")
            st.write(
                "Each dot is a candidate movie. Upper right: both of you are predicted to like it. Dots near the dashed diagonal would be rated alike; dots far from it favour one of you."
            )
            st.markdown(
                f"- {len(cands):,} candidate movies\n- {both_high:,} predicted 4 or higher by both\n- Mean gap between your predictions: {cands.gap.mean():.2f}"
            )
            if strategy == "threshold":
                st.caption("Dotted lines mark each viewer's minimum.")

with tab_compare:
    if len(cands) == 0:
        st.info("Nothing to compare: there are no candidate movies.")
    else:
        summary, overlap, tops = g.compare_strategies(cands, k=top_n, min_a=min_a, min_b=min_b)
        bars = pd.DataFrame(
            [
                {"strategy": LABELS[s], "viewer": v, "mean_pred": t[c].mean()}
                for s, t in tops.items()
                if len(t)
                for v, c in (("Viewer A", "pred_a"), ("Viewer B", "pred_b"))
            ]
        )
        if len(bars):
            base = alt.Chart(bars).encode(
                x=alt.X(
                    "strategy:N",
                    sort=[LABELS[s] for s in g.STRATEGIES],
                    axis=alt.Axis(
                        title=None,
                        labelAngle=0,
                        labelColor=pl.INK,
                        labelFontSize=12,
                        domainColor=pl.RULE,
                        tickColor=pl.RULE,
                    ),
                ),
                xOffset="viewer:N",
                y=alt.Y(
                    "mean_pred:Q",
                    scale=alt.Scale(domain=[0, 5]),
                    axis=pl.axis("Mean predicted rating (of 5)"),
                ),
                color=alt.Color(
                    "viewer:N",
                    scale=alt.Scale(
                        domain=["Viewer A", "Viewer B"], range=[pl.VIEWER_A, pl.VIEWER_B]
                    ),
                    legend=alt.Legend(title=None, orient="top"),
                ),
            )
            chart = base.mark_bar() + base.mark_text(dy=-6, fontSize=12, color=pl.INK).encode(
                text=alt.Text("mean_pred:Q", format=".2f")
            )
            st.altair_chart(
                pl.theme(
                    chart.properties(
                        width=600,
                        height=240,
                        title=f"What each of you gets from the top {top_n} of each strategy",
                    )
                ),
                theme=None,
                width="content",
            )
        pairs = [("average", "lower"), ("average", "threshold"), ("lower", "threshold")]
        st.write(
            " ".join(
                f"{LABELS[a]} and {LABELS[b]} share {overlap.loc[a, b]} of their top {top_n}."
                for a, b in pairs
            )
        )
        st.caption(
            "Computed from the same predictions that produce the rankings, so this describes how the strategies differ; it does not show that one gives better recommendations."
        )
        with st.expander("Numbers"):
            s = summary.rename(
                index=LABELS,
                columns={
                    "n_selected": "Movies selected",
                    "mean_avg_pred": "Mean predicted average",
                    "mean_lower_pred": "Mean predicted lower",
                    "mean_abs_gap": "Mean |A - B| gap",
                },
            )
            st.dataframe(s.round(3), width="stretch")
            st.dataframe(overlap.rename(index=LABELS, columns=LABELS), width="stretch")

with tab_details:
    if len(ranked):
        table = (
            g.display_table(ranked.head(top_n))
            .rename(
                columns={
                    "rank": "Rank",
                    "movieId": "Movie ID",
                    "title": "Title",
                    "genres": "Genres",
                    "pred_a": "Predicted A",
                    "pred_b": "Predicted B",
                    "avg_pred": "Average",
                    "min_pred": "Lower",
                    "score": "Strategy score",
                    "item_train_n": "Training ratings (movie)",
                }
            )
            .round(2)
        )
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Ties are broken by average, then lower score, then movie ID.")
        st.download_button(
            "Download these picks (CSV)",
            table.to_csv(index=False),
            file_name="both_pick_picks.csv",
            mime="text/csv",
        )
    else:
        st.info("No ranked movies to list.")
    t, b = METRICS["test"]["matrix_factorization"]["all"], METRICS["test"]["baseline_bias"]["all"]
    st.markdown("**About the estimates**")
    st.markdown(
        f"- A matrix-factorization model learned from {METRICS['split']['rows']['train']:,} ratings by 5,000 random MovieLens users. "
        f"On held-out ratings of the same users (random hold-out) its mean absolute error was {t['mae']:.2f} stars (RMSE {t['rmse']:.2f}); a simpler user-and-movie-average model scored {b['mae']:.2f} (RMSE {b['rmse']:.2f}).\n"
        "- Ratings are proxies for individual preference. Combining them is a decision rule, not a measure of how much two people would enjoy a film together.\n"
        "- Genre shares describe what a profile has frequently rated, not their whole taste; genres overlap, so shares do not add to 100%. Mean recorded rating also reflects which movies a person chose to rate.\n"
        "- The model's hidden factors are not interpretable, so the app does not explain why a movie is predicted high.\n"
        "- Only these profiles are supported; entering your own ratings is not available yet.\n"
        "- Unrated movies are unobserved, not disliked. Movies need enough training ratings to appear."
    )
