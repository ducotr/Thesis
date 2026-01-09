from __future__ import annotations

from analysis.embeddings import Word2VecConfig

"""
Configuration knobs for the dashboard.

This file intentionally contains *only* configuration constants (no Dash callbacks).
"""


CONCEPT = "Urban Digital Twin"
USE_CACHE = False

# Make edges readable (Cytoscape width units)
EDGE_WIDTH_MIN = 0.5
EDGE_WIDTH_MAX = 5.0

# Precompute some graphs at startup (fast UI). Semantic can be heavy.
PRECOMPUTE_LIGHT_NETWORKS = True
PRECOMPUTE_SEMANTIC = True

DEFAULT_MIN_VALUE = 5
DEFAULT_SIM_THRESHOLD = 0.5
DEFAULT_MIN_KW_FREQ = 5
DEFAULT_TOP_K = None

W2V_CONFIG = Word2VecConfig(
    vector_size=100, window=5, min_count=2, workers=4, sg=1, seed=42, epochs=50
)
