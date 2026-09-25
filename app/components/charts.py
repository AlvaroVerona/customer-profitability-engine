"""Shared Plotly chart builders.

Palette: Okabe-Ito, a widely-used colorblind-safe categorical set --
applied in a fixed order (never re-cycled per filter) for segments/actions/
scenarios, so a series keeps its color across pages and across filter
changes. Polarity (profit vs. loss, positive vs. negative delta) uses the
same two poles everywhere: orange for positive, blue for negative -- an
orange/blue diverging pair reads correctly for red-green color vision
deficiency, unlike the conventional red/green.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

CATEGORICAL_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # gray
]
POSITIVE_COLOR = "#E69F00"
NEGATIVE_COLOR = "#0072B2"
SEQUENTIAL_SCALE = "Blues"
TEMPLATE = "plotly_white"


def _style(fig: go.Figure, title: str, x_title: str | None = None, y_title: str | None = None) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "xanchor": "left", "y": 0.97, "yanchor": "top"},
        template=TEMPLATE,
        margin={"l": 10, "r": 10, "t": 80, "b": 10},
        legend={"orientation": "h", "yanchor": "top", "y": 0.88, "xanchor": "left", "x": 0},
    )
    if x_title is not None:
        fig.update_xaxes(title=x_title)
    if y_title is not None:
        fig.update_yaxes(title=y_title)
    return fig


def polarity_histogram(df: pd.DataFrame, column: str, title: str, nbins: int = 50) -> go.Figure:
    """Histogram of a profit/value-like column, split by sign so
    loss-making vs. profitable is visible at a glance."""
    data = df[[column]].dropna().copy()
    data["Polarity"] = data[column].apply(lambda v: "Profitable" if v >= 0 else "Loss-making")
    fig = px.histogram(
        data,
        x=column,
        color="Polarity",
        nbins=nbins,
        color_discrete_map={"Profitable": POSITIVE_COLOR, "Loss-making": NEGATIVE_COLOR},
    )
    return _style(fig, title, x_title=column.replace("_", " ").title(), y_title="Customers")


def bar_by_category(
    df: pd.DataFrame, x: str, y: str, title: str, category_order: list[str] | None = None
) -> go.Figure:
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=x,
        color_discrete_sequence=CATEGORICAL_PALETTE,
        category_orders={x: category_order} if category_order else None,
    )
    fig.update_layout(showlegend=False)
    return _style(fig, title, x_title=x.replace("_", " ").title(), y_title=y.replace("_", " ").title())


def scatter_2d(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    size: str | None = None,
    color: str | None = None,
    text: str | None = None,
) -> go.Figure:
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        text=text,
        color_discrete_sequence=CATEGORICAL_PALETTE,
    )
    if text:
        fig.update_traces(textposition="top center")
    return _style(fig, title, x_title=x.replace("_", " ").title(), y_title=y.replace("_", " ").title())


def revenue_vs_cost_scatter(df: pd.DataFrame, title: str = "Revenue vs. Cost") -> go.Figure:
    data = df[["total_revenue", "total_costs", "economic_profit"]].dropna()
    fig = px.scatter(
        data,
        x="total_revenue",
        y="total_costs",
        color="economic_profit",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
    )
    max_val = float(max(data["total_revenue"].max(), data["total_costs"].max()))
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val, line={"color": "#999999", "dash": "dot"})
    return _style(fig, title, x_title="Total Revenue", y_title="Total Costs")


def stacked_bar(df: pd.DataFrame, x: str, y_columns: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for i, col in enumerate(y_columns):
        fig.add_trace(
            go.Bar(x=df[x], y=df[col], name=col.replace("_", " ").title(), marker_color=CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)])
        )
    fig.update_layout(barmode="relative")
    return _style(fig, title, x_title=x.replace("_", " ").title(), y_title="Amount")


def line_series(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, markers=True, color_discrete_sequence=CATEGORICAL_PALETTE)
    return _style(fig, title, x_title=x.replace("_", " ").title(), y_title=y.replace("_", " ").title())


def horizontal_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str = POSITIVE_COLOR) -> go.Figure:
    fig = px.bar(df, x=x, y=y, orientation="h", color_discrete_sequence=[color])
    fig.update_layout(showlegend=False)
    return _style(fig, title, x_title=x.replace("_", " ").title(), y_title="")
