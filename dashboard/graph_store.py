from __future__ import annotations

"""Graph retrieval and caching.

Everything related to: PRECOMP, get_graph, get_elements, and element-level caching.
"""

from typing import Any, Dict, List

import networkx as nx

from .config import (
    DEFAULT_MIN_KW_FREQ,
    DEFAULT_MIN_VALUE,
    DEFAULT_SIM_THRESHOLD,
    DEFAULT_TOP_K,
    PRECOMPUTE_LIGHT_NETWORKS,
    PRECOMPUTE_SEMANTIC,
    W2V_CONFIG,
)
from .data_store import corpus
from .vis_elements import nx_to_cytoscape_elements

# ---------------------------------------------------------------------

PRECOMP: Dict[str, Any] = {}

if PRECOMPUTE_LIGHT_NETWORKS:
    PRECOMP["coocc_overall"] = corpus.cooccurrence_network(min_value=DEFAULT_MIN_VALUE)
    PRECOMP["coocc_yearly"] = corpus.temporal_network(min_value=DEFAULT_MIN_VALUE)
    PRECOMP["cm_overall"] = corpus.concept_method_network(min_value=DEFAULT_MIN_VALUE)
    PRECOMP["cm_yearly"] = corpus.temporal_concept_method_network(
        min_value=DEFAULT_MIN_VALUE
    )

if PRECOMPUTE_SEMANTIC:
    PRECOMP["sem_overall"] = corpus.semantic_similarity_network(
        w2v_config=W2V_CONFIG,
        min_keyword_freq=DEFAULT_MIN_KW_FREQ,
        similarity_threshold=DEFAULT_SIM_THRESHOLD,
        top_k=DEFAULT_TOP_K,
    )
    PRECOMP["sem_yearly"] = corpus.temporal_semantic_similarity_network(
        w2v_config=W2V_CONFIG,
        min_keyword_freq=DEFAULT_MIN_KW_FREQ,
        similarity_threshold=DEFAULT_SIM_THRESHOLD,
        top_k=DEFAULT_TOP_K,
    )


# ---------------------------------------------------------------------

_ELEMENTS_CACHE: Dict[str, List[dict]] = {}


def _key(*parts: Any) -> str:
    return "|".join(map(str, parts))


def get_graph(
    network_kind: str, year: int | None, *, min_value: int, sim_threshold: float
) -> nx.Graph:
    # Overall
    if network_kind == "cooccurrence_overall":
        if PRECOMPUTE_LIGHT_NETWORKS and min_value == DEFAULT_MIN_VALUE:
            return PRECOMP["coocc_overall"]
        return corpus.cooccurrence_network(min_value=min_value)

    if network_kind == "concept_method_overall":
        if PRECOMPUTE_LIGHT_NETWORKS and min_value == DEFAULT_MIN_VALUE:
            return PRECOMP["cm_overall"]
        return corpus.concept_method_network(min_value=min_value)

    if network_kind == "semantic_overall":
        if PRECOMPUTE_SEMANTIC and sim_threshold == DEFAULT_SIM_THRESHOLD:
            return PRECOMP["sem_overall"]
        return corpus.semantic_similarity_network(
            w2v_config=W2V_CONFIG,
            min_keyword_freq=DEFAULT_MIN_KW_FREQ,
            similarity_threshold=sim_threshold,
            top_k=DEFAULT_TOP_K,
        )

    # Temporal
    if network_kind == "cooccurrence_yearly":
        temporal = (
            PRECOMP.get("coocc_yearly")
            if (PRECOMPUTE_LIGHT_NETWORKS and min_value == DEFAULT_MIN_VALUE)
            else corpus.temporal_network(min_value=min_value)
        )
        year = int(year) if year is not None else max(temporal.keys())
        return temporal.get(year, nx.Graph())

    if network_kind == "concept_method_yearly":
        temporal = (
            PRECOMP.get("cm_yearly")
            if (PRECOMPUTE_LIGHT_NETWORKS and min_value == DEFAULT_MIN_VALUE)
            else corpus.temporal_concept_method_network(min_value=min_value)
        )
        year = int(year) if year is not None else max(temporal.keys())
        return temporal.get(year, nx.Graph())

    if network_kind == "semantic_yearly":
        temporal = (
            PRECOMP.get("sem_yearly")
            if (PRECOMPUTE_SEMANTIC and sim_threshold == DEFAULT_SIM_THRESHOLD)
            else corpus.temporal_semantic_similarity_network(
                w2v_config=W2V_CONFIG,
                min_keyword_freq=DEFAULT_MIN_KW_FREQ,
                similarity_threshold=sim_threshold,
                top_k=DEFAULT_TOP_K,
            )
        )
        year = int(year) if year is not None else max(temporal.keys())
        return temporal.get(year, nx.Graph())

    return nx.Graph()


def get_elements(
    network_kind: str,
    year: int | None,
    *,
    min_value: int,
    sim_threshold: float,
    node_scaling: str,
    edge_scaling: str,
    community: str,
) -> List[dict]:
    cache_key = _key(
        network_kind,
        year,
        min_value,
        sim_threshold,
        node_scaling,
        edge_scaling,
        community,
    )
    if cache_key in _ELEMENTS_CACHE:
        return _ELEMENTS_CACHE[cache_key]

    G = get_graph(network_kind, year, min_value=min_value, sim_threshold=sim_threshold)
    elements = nx_to_cytoscape_elements(
        G,
        node_scaling=node_scaling,
        edge_scaling=edge_scaling,
        community_attr=community,
    )
    _ELEMENTS_CACHE[cache_key] = elements
    return elements
