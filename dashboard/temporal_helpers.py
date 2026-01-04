from __future__ import annotations

"""
Temporal transition helpers (positions, alignment, spring layout).

This file contains NO Dash callbacks. It is safe to unit-test these functions in isolation.
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

# ---------------------------------------------------------------------

ANIM_INTERVAL_MS = 60

# Staging timing (in interval ticks)
FADE_STEPS = 10  # deaths fade out (no movement)
REVEAL_STEPS = 8  # births fade in after layout settles

# Position scale only used to seed *birth* spawn near the "middle"
POS_SCALE = 150.0
SPRING_SEED = 42
SPRING_ITER = 80

# (kept for backward compatibility; unused by cola controller)
MOVE_STEPS = 26
MAX_MOVE_PER_TRANSITION = 450.0


def _edge_key(u: str, v: str) -> Tuple[str, str]:
    return (u, v) if u <= v else (v, u)


def _pos_get(
    pos_store: Dict[str, List[float]], node_id: str
) -> Tuple[float, float] | None:
    v = pos_store.get(node_id)
    if not v or len(v) != 2:
        return None
    return float(v[0]), float(v[1])


def _pos_set(
    pos_store: Dict[str, List[float]], node_id: str, x: float, y: float
) -> None:
    pos_store[node_id] = [round(float(x), 3), round(float(y), 3)]


def _smoothstep(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _neighbors_centroid(
    G: nx.Graph, node_id: str, pos_store: Dict[str, List[float]]
) -> Tuple[float, float] | None:
    pts = []
    # node_id is string keyword; graph nodes are also keywords
    if node_id not in G:
        return None
    for nb in G.neighbors(node_id):
        p = _pos_get(pos_store, str(nb))
        if p is not None:
            pts.append(p)
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def align_target_to_start(
    start_pos: Dict[str, List[float]],
    target_pos: Dict[str, List[float]],
    align_nodes: List[str],
) -> Dict[str, List[float]]:
    """
    Translate/rotate/scale target positions onto start positions to prevent big global sweeps.
    """
    pts_s = []
    pts_t = []
    for n in align_nodes:
        if n in start_pos and n in target_pos:
            pts_s.append(start_pos[n])
            pts_t.append(target_pos[n])

    if len(pts_s) < 3:
        return target_pos

    S = np.array(pts_s, dtype=float)
    T = np.array(pts_t, dtype=float)

    S_mean = S.mean(axis=0)
    T_mean = T.mean(axis=0)
    S0 = S - S_mean
    T0 = T - T_mean

    sS = float(np.linalg.norm(S0)) + 1e-9
    sT = float(np.linalg.norm(T0)) + 1e-9
    S0 /= sS
    T0 /= sT

    U, _, Vt = np.linalg.svd(T0.T @ S0)
    R = U @ Vt

    out: Dict[str, List[float]] = {}
    for n, (x, y) in target_pos.items():
        v = np.array([x, y], dtype=float)
        v0 = (v - T_mean) / sT
        v1 = (v0 @ R) * sS + S_mean
        out[n] = [float(v1[0]), float(v1[1])]
    return out


def spring_layout_cyto(
    G: nx.Graph,
    *,
    init_pos_cyto: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[float, float]]:
    """
    Compute spring layout, but accept and return Cytoscape scaled coords.
    """
    if G.number_of_nodes() == 0:
        return {}

    init_norm: Dict[Any, Tuple[float, float]] = {}
    for n in G.nodes():
        nid = str(n)
        if nid in init_pos_cyto:
            x, y = init_pos_cyto[nid]
            init_norm[n] = (float(x) / POS_SCALE, float(y) / POS_SCALE)

    pos = nx.spring_layout(
        G,
        pos=init_norm,
        seed=SPRING_SEED,
        iterations=SPRING_ITER,
        scale=1.0,
    )
    return {
        str(n): (float(x) * POS_SCALE, float(y) * POS_SCALE)
        for n, (x, y) in pos.items()
    }


# Backward compatibility for older callback modules
BIRTH_STEPS = REVEAL_STEPS
