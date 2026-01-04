from __future__ import annotations

"""Cytoscape element construction + scaling helpers."""

import math
from typing import Any, Dict, List, Tuple

import networkx as nx

from data.publication_corpus import PublicationCorpus

from .config import EDGE_WIDTH_MAX, EDGE_WIDTH_MIN


def _hsv_color(i: int, n: int) -> str:
    import colorsys

    n = max(n, 1)
    h = (i % n) / n
    r, g, b = colorsys.hsv_to_rgb(h, 0.55, 0.95)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def _community_color_map(G: nx.Graph, community_attr: str) -> Dict[Any, str]:
    groups = sorted({data.get(community_attr, 0) for _, data in G.nodes(data=True)})
    return {g: _hsv_color(i, len(groups)) for i, g in enumerate(groups)}


def _scale_to_range(values: List[float], out_min: float, out_max: float) -> List[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if vmax == vmin:
        mid = (out_min + out_max) / 2.0
        return [mid for _ in values]
    return [out_min + (v - vmin) * (out_max - out_min) / (vmax - vmin) for v in values]


def edge_width_map(G: nx.Graph, scaling: str) -> Dict[Tuple[Any, Any], float]:
    """
    Compute edge widths from raw_weight, then rescale to [EDGE_WIDTH_MIN, EDGE_WIDTH_MAX].
    This avoids the very large default widths coming from add_edge_widths(min=10,max=50).
    """
    edges = []
    raw_vals = []
    for u, v, w in G.edges(data="raw_weight"):
        if w is None:
            continue
        edges.append((u, v))
        raw_vals.append(float(w))

    if not edges:
        return {}

    scaling_upper = scaling.upper()

    def transform(x: float) -> float:
        if scaling_upper.endswith("NONE"):
            return 1.0
        if scaling_upper.endswith("LINEAR"):
            return x
        if scaling_upper.endswith("SQRT"):
            return math.sqrt(max(x, 0.0))
        if scaling_upper.endswith("LOG"):
            return math.log(1.0 + max(x, 0.0))
        # fallback
        return x

    transformed = [transform(x) for x in raw_vals]
    widths = _scale_to_range(transformed, EDGE_WIDTH_MIN, EDGE_WIDTH_MAX)

    return {e: w for e, w in zip(edges, widths)}


def nx_to_cytoscape_elements(
    G: nx.Graph,
    *,
    node_scaling: str,
    edge_scaling: str,
    community_attr: str,
) -> List[dict]:
    color_map = _community_color_map(G, community_attr)
    ew = edge_width_map(G, edge_scaling)

    elements: List[dict] = []

    # Nodes
    for node, data in G.nodes(data=True):
        size = float(data.get(node_scaling, 18.0))
        comm = data.get(community_attr, 0)
        node_type = data.get("node_type")

        elements.append(
            {
                "data": {
                    "id": str(node),
                    "label": str(node),
                    "size": size,
                    "community": comm,
                    "color": color_map.get(comm, "rgb(160,160,160)"),
                    "node_type": node_type or "",
                    "raw_degree": data.get("raw_degree", None),
                    "raw_weighted_degree": data.get("raw_weighted_degree", None),
                }
            }
        )

    # Edges
    for u, v, data in G.edges(data=True):
        width = ew.get((u, v), ew.get((v, u), 1.0))
        elements.append(
            {
                "data": {
                    "id": f"{u}__{v}",
                    "source": str(u),
                    "target": str(v),
                    "width": float(width),
                    "raw_weight": data.get("raw_weight", None),
                }
            }
        )

    return elements


def publications_index(corpus: PublicationCorpus) -> Dict[str, List[int]]:
    """
    Map keyword -> list of publication indices that contain it.
    (Later: switch to canonical keyword tokens for perfect alignment.)
    """
    index: Dict[str, List[int]] = {}
    for i, pub in enumerate(corpus.data):
        for kw in pub.keywords:
            index.setdefault(kw, []).append(i)
    return index


def bipartite_column_positions(
    left_ids: list[str],
    right_ids: list[str],
    *,
    left_x: float = 0.0,
    right_x: float = 650.0,
    y_gap: float = 42.0,
) -> dict[str, dict[str, float]]:
    """
    Returns {node_id: {"x": ..., "y": ...}} for a strict 2-column layout.
    Lists should already be sorted in desired top-to-bottom order.
    """
    pos: dict[str, dict[str, float]] = {}

    def _centered_y(ids: list[str], x: float):
        n = len(ids)
        y0 = -((n - 1) * y_gap) / 2.0 if n > 1 else 0.0
        for i, nid in enumerate(ids):
            pos[nid] = {"x": float(x), "y": float(y0 + i * y_gap)}

    _centered_y(left_ids, left_x)
    _centered_y(right_ids, right_x)
    return pos
