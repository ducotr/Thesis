import math
from enum import StrEnum
from typing import Any

import networkx as nx


def scale_to_range(values: dict, minimum: int, maximum: int) -> dict:
    """Linearly scale dict of node -> value into [minimum, maximum]."""
    if not values:
        return {}

    v_min = min(values.values())
    v_max = max(values.values())

    if v_max == v_min:
        # All nodes have same value so give them all the mid-size
        mid = (minimum + maximum) / 2
        return {node: mid for node in values}

    scaled: dict = {}
    for node, v in values.items():
        norm = (v - v_min) / (v_max - v_min)  # 0-1
        scaled[node] = minimum + norm * (maximum - minimum)
    return scaled


# Node scaling functions
def degree(G: nx.Graph) -> dict[Any, float]:
    return nx.degree_centrality(G)


def weighted_degree(G: nx.Graph) -> dict[Any, float]:
    return dict(G.degree(weight="raw_weight"))


def betweenness(G: nx.Graph) -> dict[Any, float]:
    return nx.betweenness_centrality(G, weight="raw_weight", normalized=True)


def closeness(G: nx.Graph) -> dict[Any, float]:
    return nx.closeness_centrality(G)


def eigenvector(G: nx.Graph) -> dict[Any, float]:
    return nx.eigenvector_centrality(G, max_iter=1000, weight="raw_weight")


class NodeScaling(StrEnum):
    NONE = "NODESCALING.NONE"  # No node scaling (constant node size)
    DEGREE = "NODESCALING.DEGREE"  # Degree centrality node scaling
    WEIGHTED_DEGREE = (
        "NODESCALING.WEIGHTED_DEGREE"  # Weighted degree centrality node scaling
    )
    BETWEENNESS = "NODESCALING.BETWEENNESS"  # Betweenness centrality node scaling
    CLOSENESS = "NODESCALING.CLOSENESS"  # Closeness centrality node scaling
    EIGENVECTOR = "NODESCALING.EIGENVECTOR"  # Eigenvector centrality node scaling


def add_node_sizes(G: nx.Graph, minimum: int = 10, maximum: int = 50) -> None:
    """
    Scales nodes using desired node scaling method and minimum and maximum value.
    """

    mid = (minimum + maximum) / 2
    values = {node: mid for node in G.nodes}
    for node, value in values.items():
        G.nodes[node][NodeScaling.NONE] = value

    size_metrics = [
        (degree, NodeScaling.DEGREE),
        (weighted_degree, NodeScaling.WEIGHTED_DEGREE),
        (betweenness, NodeScaling.BETWEENNESS),
        (closeness, NodeScaling.CLOSENESS),
        (eigenvector, NodeScaling.EIGENVECTOR),
    ]

    for fun, key in size_metrics:
        values = fun(G)
        values = scale_to_range(values, minimum, maximum)
        for node, value in values.items():
            G.nodes[node][key] = value


# Edge scaling functions
def linear(values: dict[Any, int]) -> dict[Any, float]:
    return {edge: float(w) for edge, w in values.items()}


def sqrt(values: dict[Any, int]) -> dict[Any, float]:
    return {edge: math.sqrt(float(w)) for edge, w in values.items()}


def log(values: dict[Any, int]) -> dict[Any, float]:
    return {edge: math.log(1.0 + float(w)) for edge, w in values.items()}


class EdgeScaling(StrEnum):
    NONE = "EDGESCALING.NONE"  # No edge scaling (constant edge width)
    LINEAR = "EDGESCALING.LINEAR"  # Linear edge scaling
    SQRT = "EDGESCALING.SQRT"  # Square root edge scaling
    LOG = "EDGESCALING.LOG"  # Logarithmic edge scaling


def add_edge_widths(
    G: nx.Graph,
    minimum: int = 10,
    maximum: int = 50,
) -> None:
    """
    Precomputes multiple edge width variants (none / linear / sqrt / log)
    and stores them as separate attributes on each edge:
        - none_width
        - linear_width
        - sqrt_width
        - log_width
    """

    base_values = {}
    for u, v, raw_weight in G.edges(data="raw_weight"):
        if raw_weight is not None:
            base_values[(u, v)] = raw_weight

    mid = (minimum + maximum) / 2

    for u, v in base_values:
        G.edges[u, v][EdgeScaling.NONE] = mid

    width_metrics = [
        (linear, EdgeScaling.LINEAR),
        (sqrt, EdgeScaling.SQRT),
        (log, EdgeScaling.LOG),
    ]

    for fun, key in width_metrics:
        values = fun(base_values)
        values = scale_to_range(values, minimum, maximum)
        for (u, v), width in values.items():
            G.edges[u, v][key] = width


# Community/cluster detection functions
def k_means():
    pass


def louvain():
    pass


def leiden():
    pass


def girvan_newman():
    pass


class CommunityDetection(StrEnum):
    NONE = "COMMUNITYDETECTION.NONE"  # No community detection
    LOUVAIN = "COMMUNITYDETECTION.LOUVAIN"  # Louvain community detection
    # LEIDEN = "COMMUNITYDETECTION.LEIDEN"  # Leiden community detection


def constant_partition(G: nx.Graph, value: int = 0) -> dict[Any, int]:
    return {n: value for n in G.nodes}


def louvain_partition(G: nx.Graph, weight: str = "raw_weight") -> dict[Any, int]:
    comms = nx.community.louvain_communities(G, weight=weight)
    node2comm: dict[Any, int] = {}
    for i, c in enumerate(comms):
        for n in c:
            node2comm[n] = i
    return node2comm


def leiden_partition(G: nx.Graph, weight: str = "raw_weight") -> dict[Any, int]:
    raise NotImplementedError()


def add_communities(G: nx.Graph, weight: str = "raw_weight") -> None:
    """
    Precompute community partitions and store them on nodes under:
        - CommunityDetection.NONE
        - CommunityDetection.LOUVAIN
        - CommunityDetection.LEIDEN
    """

    methods = [
        (lambda g: constant_partition(g, 0), CommunityDetection.NONE),
        (lambda g: louvain_partition(g, weight), CommunityDetection.LOUVAIN),
        # (lambda g: leiden_partition(g, weight), CommunityDetection.LEIDEN)
    ]

    for func, key in methods:
        node2comm = func(G)
        for n, cid in node2comm.items():
            G.nodes[n][key] = int(cid)


# Temporal metrics
# - Jaccard similarity of top keywords per year
#     - Shows persistence of semantic cores.
# - Edge-weight drift
#     - Shows which keyword associations get stronger or weaker over time.
# - Birth and death of nodes
#     - A node appears when a keyword first shows up.
# - Evolution of community structure
#     - Track clusters over time—this is amazing in visual form.


def jaccard_similarity_temporal_network(
    temporal_network: dict[int, nx.Graph],
) -> dict[int, float]:
    """
    Returns dict[year, jaccard_with_previous_year].
    """
    if not temporal_network:
        return {}

    years = list(temporal_network.keys())
    similarity: dict[int, float] = {years[0]: 0.0}
    prev_kws = set([name for name in temporal_network[years[0]].nodes])
    for year in years[1:]:
        cur_kws = set([name for name in temporal_network[year].nodes])
        union = prev_kws | cur_kws
        intersection = prev_kws & cur_kws
        similarity[year] = (len(intersection) / len(union)) if union else 0.0
        prev_kws = cur_kws

    return similarity


def node_birth_death(
    temporal_network: dict[int, nx.Graph],
) -> dict[int, dict[str, list[str]]]:
    """
    Returns:
    {
        year: {
            "births": ["Computer science", "Software", ...], # Nodes that were born this year (nodes that weren't there last year)
            "deaths": ["Data mining", "User interface", ...] # Nodes that will die this year (nodes that won't be there next year)
        },
        ...
    }
    """
    if not temporal_network:
        return {}

    # Make sure years are in chronological order
    years = list(temporal_network.keys())
    births_deaths: dict[int, dict[str, list[str]]] = {}

    for i, year in enumerate(years):
        G_current = temporal_network[year]
        current_nodes = set(G_current.nodes())

        # Ensure dict entry exists
        births_deaths[year] = {}

        # Births
        if i == 0:
            # First year: all nodes are births
            births_deaths[year]["births"] = list(current_nodes)
        else:
            prev_year = years[i - 1]
            prev_nodes = set(temporal_network[prev_year].nodes())
            births_deaths[year]["births"] = list(current_nodes - prev_nodes)

        # Deaths
        if i == len(years) - 1:
            # Last year: all nodes are deaths
            births_deaths[year]["deaths"] = list(current_nodes)
        else:
            next_year = years[i + 1]
            next_nodes = set(temporal_network[next_year].nodes())
            births_deaths[year]["deaths"] = list(current_nodes - next_nodes)

    return births_deaths
