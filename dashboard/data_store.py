from __future__ import annotations

"""
Data loading and lightweight derived statistics.

Side effects: loads the PublicationCorpus on import (same as the original single-file app).
"""

from functools import lru_cache

import pandas as pd
import plotly.express as px

from data.publication_corpus import PublicationCorpus

from .config import CONCEPT, DEFAULT_MIN_VALUE, USE_CACHE
from .stats_store import (
    build_temporal_cooccurrence,
    compute_kpis,
    compute_network_metrics_over_time,
    keyword_counts_by_year,
    publications_to_df,
    pubs_timeseries_daily,
)
from .vis_elements import publications_index

corpus = PublicationCorpus(CONCEPT, use_cache=USE_CACHE)
kw_index = publications_index(corpus)

years_count = corpus.years()
years_sorted = sorted(years_count.keys())
min_year = years_sorted[0] if years_sorted else 2000
max_year = years_sorted[-1] if years_sorted else 2000

DEFAULT_TEMP_YEAR = (
    min_year  # change to e.g. 2019 if you want the dashboard to start there
)

ts_df = [{"year": y, "publications": years_count[y]} for y in years_sorted]
ts_fig = px.line(ts_df, x="year", y="publications", markers=True)

# quick stats
total_pubs = len(corpus.data)
unique_keywords = len({kw for p in corpus.data for kw in p.keywords})


PUBLICATIONS_DF = publications_to_df(corpus.data)
KPIS = compute_kpis(PUBLICATIONS_DF)

PUBS_DAILY = pubs_timeseries_daily(PUBLICATIONS_DF)
KW_YEAR = keyword_counts_by_year(PUBLICATIONS_DF)


@lru_cache(maxsize=16)
def get_net_year_metrics(min_value: int) -> pd.DataFrame:
    temporal = build_temporal_cooccurrence(PUBLICATIONS_DF, min_weight=int(min_value))
    return compute_network_metrics_over_time(temporal)
