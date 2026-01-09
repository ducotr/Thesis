# export_analytics_tab_csv.py
"""
Export the *data behind the Dashboard Analytics tab* to CSV files for use in a thesis.

This script is intentionally "boring":
- No Dash
- No Plotly
- Just reproducible CSV outputs

Outputs are written to OUTPUT_DIR (default: data/analytics_tab/).

It mirrors the computations used in the analytics tab:
- Publications per day (filled gaps) + rolling means + cumulative
- Publications per year
- Keyword counts by year
- Top-20 keywords overall (by document frequency)
- Emerging/fading keywords (linear trend over last LOOKBACK_YEARS, filtered by MIN_TOTAL_IN_WINDOW)
- Co-occurrence network evolution metrics per year (Jaccard, births/deaths, density, clustering, modularity, ...)

Run:
    python3 export_analytics_tab_csv.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.publication_corpus import PublicationCorpus

# Reuse the same helpers as the dashboard so results match the UI
from dashboard.stats_store import (
    publications_to_df,
    compute_kpis,
    pubs_timeseries_daily,
    keyword_counts_by_year,
    build_temporal_cooccurrence,
    compute_network_metrics_over_time,
)

# ----------------------------
# Global configuration
# ----------------------------

# Where to write CSVs (LaTeX-friendly)
OUTPUT_DIR: Path = Path("overleaf_files/network_metrics")

# Keyword trend settings (match UI defaults)
LOOKBACK_YEARS: int = 3
MIN_TOTAL_IN_WINDOW: int = 5

# Rolling windows for the daily publications plot
PUB_SMOOTH_WINDOWS_DAYS: tuple[int, ...] = (7, 30, 90)

# Network evolution settings (match UI default)
# You can include multiple min-values to export multiple metric files.
NETWORK_MIN_VALUES: tuple[int, ...] = ()  # if empty, defaults to dashboard DEFAULT_MIN_VALUE

# ----------------------------
# Utilities
# ----------------------------


def _load_dashboard_defaults() -> tuple[str, bool, int]:
    """
    Best-effort import of dashboard config. Falls back to root config.py if present.
    Returns: (concept, use_cache, default_min_value)
    """
    concept = "Urban Digital Twin"
    use_cache = False
    default_min_value = 5

    return concept, use_cache, default_min_value


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_df(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path.parent)
    df.to_csv(path, index=False)


# ----------------------------
# Exports
# ----------------------------


def export_kpis(df: pd.DataFrame, concept: str, out_dir: Path) -> None:
    k = compute_kpis(df)

    if not df.empty:
        first_date = df["date"].min().date().isoformat()
        last_date = df["date"].max().date().isoformat()
        span_days = int((df["date"].max() - df["date"].min()).days)
    else:
        first_date = ""
        last_date = ""
        span_days = 0

    row = {
        "concept": concept,
        "total_pubs": int(k.get("total_pubs", 0)),
        "n_keywords": int(k.get("n_keywords", 0)),
        "n_authors": k.get("n_authors", "N/A"),
        "first_publication_date": first_date,
        "last_publication_date": last_date,
        "timespan_days": span_days,
    }

    _write_df(pd.DataFrame([row]), out_dir / "kpis.csv")


def export_publications_timeseries(df: pd.DataFrame, out_dir: Path) -> None:
    daily = pubs_timeseries_daily(df).copy()
    if daily.empty:
        _write_df(daily, out_dir / "publications_daily.csv")
        return

    daily = daily.rename(columns={"count": "publications"})
    daily["cumulative_publications"] = daily["publications"].cumsum()

    # rolling means
    for w in PUB_SMOOTH_WINDOWS_DAYS:
        col = f"rolling_{int(w)}d"
        daily[col] = (
            daily["publications"].rolling(window=int(w), min_periods=1, center=True).mean()
        )

    # ensure ISO date for LaTeX-friendly parsing
    daily["date"] = pd.to_datetime(daily["date"]).dt.date.astype(str)

    _write_df(daily, out_dir / "publications_daily.csv")

    # yearly
    yearly = (
        df.groupby("year")
        .size()
        .rename("publications")
        .reset_index()
        .sort_values("year")
    )
    _write_df(yearly, out_dir / "publications_yearly.csv")


def export_keyword_tables(df: pd.DataFrame, out_dir: Path) -> None:
    kw_year = keyword_counts_by_year(df).copy()
    _write_df(kw_year, out_dir / "keyword_counts_by_year.csv")

    if kw_year.empty:
        _write_df(pd.DataFrame({"keyword": [], "count": [], "rank": []}), out_dir / "keyword_frequency_top20.csv")
        _write_df(pd.DataFrame(), out_dir / f"keyword_trends_window{LOOKBACK_YEARS}_min{MIN_TOTAL_IN_WINDOW}.csv")
        _write_df(pd.DataFrame(), out_dir / f"keyword_emerging_top15_window{LOOKBACK_YEARS}_min{MIN_TOTAL_IN_WINDOW}.csv")
        _write_df(pd.DataFrame(), out_dir / f"keyword_fading_top15_window{LOOKBACK_YEARS}_min{MIN_TOTAL_IN_WINDOW}.csv")
        return

    # Top 20 overall
    top = (
        kw_year.groupby("keyword")["count"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    _write_df(top.rename(columns={"count": "frequency"}), out_dir / "keyword_frequency_top20.csv")

    # Emerging / fading (trend over last LOOKBACK_YEARS)
    years = sorted(kw_year["year"].unique().tolist())
    end_year = int(max(years))
    start_year = int(end_year - int(LOOKBACK_YEARS) + 1)
    window = kw_year[(kw_year["year"] >= start_year) & (kw_year["year"] <= end_year)].copy()

    pivot = window.pivot_table(index="keyword", columns="year", values="count", fill_value=0)
    pivot["total"] = pivot.sum(axis=1)

    # filter out very rare keywords (within the window)
    pivot = pivot[pivot["total"] >= int(MIN_TOTAL_IN_WINDOW)]

    year_cols = [c for c in pivot.columns if c != "total"]
    xs = np.array(sorted(year_cols), dtype=float)

    def slope(row: pd.Series) -> float:
        ys = row[sorted(year_cols)].values.astype(float)
        return float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 2 else 0.0

    if len(year_cols) >= 1:
        pivot["slope"] = pivot.apply(slope, axis=1)
        pivot["last"] = pivot.get(end_year, 0)
        pivot["prev"] = pivot.get(end_year - 1, 0)
    else:
        pivot["slope"] = 0.0
        pivot["last"] = 0
        pivot["prev"] = 0

    trends = pivot.reset_index()
    # Keep a consistent column order: keyword, years..., total, slope, last, prev
    ordered_cols = ["keyword"] + sorted([c for c in trends.columns if isinstance(c, (int, np.integer))]) + [
        "total",
        "slope",
        "last",
        "prev",
    ]
    trends = trends[ordered_cols]
    _write_df(trends, out_dir / f"keyword_trends_window{LOOKBACK_YEARS}_min{MIN_TOTAL_IN_WINDOW}.csv")

    emerging = trends.sort_values("slope", ascending=False).head(15)
    fading = trends.sort_values("slope", ascending=True).head(15)

    keep = ["keyword", "total", "slope", "last", "prev"]
    _write_df(emerging[keep], out_dir / f"keyword_emerging_top15_window{LOOKBACK_YEARS}_min{MIN_TOTAL_IN_WINDOW}.csv")
    _write_df(fading[keep], out_dir / f"keyword_fading_top15_window{LOOKBACK_YEARS}_min{MIN_TOTAL_IN_WINDOW}.csv")


def export_network_metrics(df: pd.DataFrame, min_values: list[int], out_dir: Path) -> None:
    if df.empty:
        return

    for mv in min_values:
        temporal = build_temporal_cooccurrence(df, min_weight=int(mv))
        metrics = compute_network_metrics_over_time(temporal).copy()
        metrics.insert(0, "min_value", int(mv))
        _write_df(metrics, out_dir / f"network_metrics_by_year_min{int(mv)}.csv")


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    concept, use_cache, default_min_value = _load_dashboard_defaults()

    min_values = list(NETWORK_MIN_VALUES) if NETWORK_MIN_VALUES else [default_min_value]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = PublicationCorpus(concept, use_cache=use_cache)
    df = publications_to_df(corpus.data)

    export_kpis(df, concept, OUTPUT_DIR)
    export_publications_timeseries(df, OUTPUT_DIR)
    export_keyword_tables(df, OUTPUT_DIR)
    export_network_metrics(df, min_values, OUTPUT_DIR)

    print(f"Wrote analytics-tab CSVs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
