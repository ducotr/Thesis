# dashboard/callbacks_stats.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output
from plotly.subplots import make_subplots

from .data_store import KW_YEAR, PUBS_DAILY, get_net_year_metrics


def register(app: Dash) -> None:
    @app.callback(
        Output("pubs-ts-fig", "figure"),
        Input("pubs-smooth-window", "value"),
    )
    def update_pubs_ts(window_days: int):
        w = max(int(window_days or 90), 1)
        df = PUBS_DAILY.copy()

        # daily new publications (already gap-filled in PUBS_DAILY)
        df["new_smooth"] = (
            df["count"].rolling(window=w, min_periods=1, center=True).mean()
        )

        # cumulative actual
        df["Cumulative"] = df["count"].cumsum()

        # cumulative smoothed (smooth the *increments*, then cum-sum -> stays monotone)
        df["Smoothed cumulative"] = df["new_smooth"].cumsum()

        # scale smoothed curve so it ends at the same total as the true cumulative
        if len(df) and df["Smoothed cumulative"].iloc[-1] > 0:
            df["Smoothed cumulative"] *= (
                df["Cumulative"].iloc[-1] / df["Smoothed cumulative"].iloc[-1]
            )

        fig = px.line(df, x="date", y=["Cumulative", "Smoothed cumulative"])
        fig.update_layout(
            legend_title_text="",
            yaxis_title="Cumulative publications",
            xaxis_title="Date",
        )
        return fig

    @app.callback(
        Output("kw-emerging-table", "data"),
        Output("kw-fading-table", "data"),
        Output("kw-top-table", "data"),
        Input("kw-lookback", "value"),
        Input("kw-min-total", "value"),
    )
    def update_keyword_tables(lookback: int, min_total: int):
        lookback = int(lookback or 3)
        min_total = int(min_total or 5)

        if KW_YEAR.empty:
            return [], [], []

        years = sorted(KW_YEAR["year"].unique().tolist())
        end_year = years[-1]
        start_year = end_year - lookback

        window = KW_YEAR[
            (KW_YEAR["year"] >= start_year) & (KW_YEAR["year"] <= end_year)
        ].copy()
        pivot = window.pivot_table(
            index="keyword", columns="year", values="count", fill_value=0
        )
        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot[pivot["total"] >= min_total]

        year_cols = [c for c in pivot.columns if c != "total"]
        xs = np.array(sorted(year_cols), dtype=float)

        def slope(row) -> float:
            ys = row[year_cols].values.astype(float)
            return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0

        pivot["slope"] = pivot.apply(slope, axis=1)
        pivot["last"] = pivot.get(end_year, 0)
        pivot["prev"] = pivot.get(end_year - 1, 0)

        emerging = pivot.sort_values("slope", ascending=False).head(15).reset_index()
        fading = pivot.sort_values("slope", ascending=True).head(15).reset_index()

        top = (
            KW_YEAR.groupby("keyword")["count"]
            .sum()
            .sort_values(ascending=False)
            .head(20)
            .reset_index()
        )

        keep_em = ["keyword", "total", "slope", "last", "prev"]
        return (
            emerging[keep_em].to_dict("records"),
            fading[keep_em].to_dict("records"),
            top.to_dict("records"),
        )

    @app.callback(
        Output("net-similarity-fig", "figure"),
        Output("net-size-fig", "figure"),
        Output("net-structure-fig", "figure"),
        Input("stats-min-value", "value"),
        Input("net-smooth-years", "value"),
    )
    def update_network_evolution(min_value: int, smooth_years: int):
        df = get_net_year_metrics(int(min_value or 2)).copy().sort_values("year")
        w = max(int(smooth_years or 1), 1)

        def smooth(col: str) -> str:
            if w <= 1 or col not in df.columns:
                return col
            out = f"{col}_smooth"
            df[out] = df[col].rolling(window=w, min_periods=1, center=True).mean()
            return out

        # --- Similarity (0..1) ---
        sim_cols = []
        if "edge_jaccard_prev" in df.columns:
            sim_cols.append(smooth("edge_jaccard_prev"))
        elif "jaccard_prev" in df.columns:
            sim_cols.append(smooth("jaccard_prev"))
        if "node_jaccard_prev" in df.columns:
            sim_cols.append(smooth("node_jaccard_prev"))

        fig_sim = px.line(df, x="year", y=sim_cols, markers=True)
        fig_sim.update_layout(
            title="Similarity to previous year (Jaccard)",
            yaxis_title="Jaccard (0–1)",
            legend_title_text="",
        )
        fig_sim.update_yaxes(range=[0, 1])

        # --- Size & churn ---
        size_cols = [
            c for c in ["nodes", "edges", "births", "deaths"] if c in df.columns
        ]
        size_cols = [smooth(c) for c in size_cols]
        fig_size = px.line(df, x="year", y=size_cols, markers=True)
        fig_size.update_layout(
            title="Network size & churn",
            yaxis_title="Count",
            legend_title_text="",
        )

        # --- Structure (secondary axis for communities) ---
        left_metrics = [
            c for c in ["density", "avg_clustering", "modularity"] if c in df.columns
        ]
        left_metrics = [smooth(c) for c in left_metrics]

        comm_col = "n_communities" if "n_communities" in df.columns else None
        comm_col = smooth(comm_col) if comm_col else None

        fig_struct = make_subplots(specs=[[{"secondary_y": True}]])

        # left axis: [0..1-ish] metrics
        for col in left_metrics:
            fig_struct.add_trace(
                go.Scatter(
                    x=df["year"],
                    y=df[col],
                    mode="lines+markers",
                    name=col.replace("_smooth", ""),
                ),
                secondary_y=False,
            )

        # right axis: number of communities
        if comm_col:
            fig_struct.add_trace(
                go.Scatter(
                    x=df["year"],
                    y=df[comm_col],
                    mode="lines+markers",
                    name="n_communities",
                ),
                secondary_y=True,
            )

        fig_struct.update_layout(
            title="Network structure",
            legend_title_text="",
        )
        fig_struct.update_yaxes(
            title_text="Density / clustering / modularity", secondary_y=False
        )
        fig_struct.update_yaxes(title_text="# communities", secondary_y=True)

        return fig_sim, fig_size, fig_struct
