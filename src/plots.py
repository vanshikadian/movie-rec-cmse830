"""Altair charts shared by the notebooks and the Streamlit app."""

import altair as alt
import pandas as pd

INK, PAPER, GREEN, RULE = "#22251F", "#F6F3EC", "#24483C", "#D8D7CF"
VIEWER_A, VIEWER_B = "#287D8E", "#B65B36"  # dark teal, burnt orange
GRID, MUTED_DOT = "#E6E3D9", "#B4B3A8"
FONT = "Source Sans Pro, Helvetica Neue, Arial, sans-serif"


def axis(title, color=INK):
    return alt.Axis(
        title=title,
        titleColor=color,
        labelColor=INK,
        gridColor=GRID,
        domainColor=RULE,
        tickColor=RULE,
        labelFontSize=12,
        titleFontSize=12,
        titleFontWeight="normal",
        labelFont=FONT,
        titleFont=FONT,
    )


def theme(chart):
    """Final-chart styling for the app (call once on the finished chart, never on sub-charts)."""
    return (
        chart.configure(background=PAPER)
        .configure_view(strokeWidth=0)
        .configure_title(color=INK, font=FONT, fontWeight="normal", anchor="start")
        .configure_legend(labelColor=INK, titleColor=INK, labelFont=FONT, labelFontSize=12)
    )


def pair_scatter(
    cands, picks, user_a, user_b, title, min_a=None, min_b=None, sample_n=2500, seed=830
):
    """Candidate movies by predicted rating of viewer A (x) and viewer B (y).

    `picks` is the ranked table of selected movies (all drawn, highlighted). The grey background is a
    fixed-seed random sample of at most `sample_n` of the OTHER candidates (display only); every summary
    elsewhere uses the full pool.
    """
    rest = cands[~cands.movieId.isin(picks.movieId)]
    shown = rest if len(rest) <= sample_n else rest.sample(sample_n, random_state=seed)
    lim = [0.5, 5.0]
    sc = alt.Scale(domain=lim, nice=False)
    ax = alt.X(
        "pred_a:Q",
        scale=sc,
        axis=axis(f"Viewer A (profile {user_a}): predicted rating, out of 5", VIEWER_A),
    )
    ay = alt.Y(
        "pred_b:Q",
        scale=sc,
        axis=axis(f"Viewer B (profile {user_b}): predicted rating, out of 5", VIEWER_B),
    )
    pt = [
        alt.Tooltip("title:N", title="Movie"),
        alt.Tooltip("pred_a:Q", title="Viewer A", format=".2f"),
        alt.Tooltip("pred_b:Q", title="Viewer B", format=".2f"),
    ]
    layers = [
        alt.Chart(shown)
        .mark_circle(size=16, opacity=0.4, color=MUTED_DOT)
        .encode(x=ax, y=ay, tooltip=pt)
    ]
    if len(picks):
        layers.append(
            alt.Chart(picks)
            .mark_point(
                size=90, filled=True, color=GREEN, stroke=INK, strokeWidth=0.8, opacity=0.95
            )
            .encode(x=ax, y=ay, tooltip=[alt.Tooltip("rank:Q", title="Rank")] + pt)
        )
    layers.append(
        alt.Chart(pd.DataFrame({"x": lim, "y": lim}))
        .mark_line(color=INK, strokeDash=[4, 4], opacity=0.45)
        .encode(x="x:Q", y="y:Q")
    )
    if min_a is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"v": [min_a]}))
            .mark_rule(color=VIEWER_A, strokeDash=[2, 3])
            .encode(x="v:Q")
        )
    if min_b is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"v": [min_b]}))
            .mark_rule(color=VIEWER_B, strokeDash=[2, 3])
            .encode(y="v:Q")
        )
    note = [
        f"Grey: random display sample ({len(shown):,} of {len(rest):,} other candidates).",
        "Green: selected picks. Dashed line: equal predictions.",
    ]
    return alt.layer(*layers).properties(
        title=alt.TitleParams(
            title, subtitle=note, fontSize=14, subtitleFontSize=11, subtitleColor="#5E6157"
        ),
        width=460,
        height=440,
    )
