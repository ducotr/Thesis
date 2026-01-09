# export_dashboard_networks_latex.py
"""
Export dashboard networks to LaTeX/TikZ using NetworkX's nx.to_latex_raw().

This script writes one .tex file per network (a standalone tikzpicture snippet),
ready to be \\input{} into your thesis.

Run from project root:
    python3 export_dashboard_networks_latex.py

Output directory (default):
    data/latex_networks/

Notes:
- TikZ output can get huge for dense graphs. Use MAX_NODES / MAX_EDGES if needed.
- Node names in the graph may contain characters not valid for TikZ node identifiers.
  To avoid that, this script relabels nodes to n0, n1, ... and uses labels for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math

import networkx as nx
from networkx.drawing.nx_latex import to_latex_raw

from data.publication_corpus import PublicationCorpus
from analysis.embeddings import Word2VecConfig

# Dashboard helper (no Dash dependency)
from dashboard.vis_elements import bipartite_column_positions

# ----------------------------
# Global configuration
# ----------------------------

OUTPUT_DIR: Path = Path("overleaf_files/latex_networks")

EXPORT_COOCCURRENCE: bool = True
EXPORT_CONCEPT_METHOD: bool = True
EXPORT_SEMANTIC: bool = True  # can be slow if models aren't cached

MIN_VALUE: int | None = 10  # None -> use dashboard DEFAULT_MIN_VALUE

SIM_THRESHOLD: float | None = None  # None -> use dashboard DEFAULT_SIM_THRESHOLD
MIN_KEYWORD_FREQ: int | None = None  # None -> use dashboard DEFAULT_MIN_KW_FREQ
TOP_K: int | None = None  # None -> use dashboard DEFAULT_TOP_K
W2V_CONFIG: Word2VecConfig | None = None  # None -> use dashboard W2V_CONFIG

EXPORT_TEMPORAL: bool = True
YEARS: List[int] = []  # empty -> export all years present in corpus

# Layout
SPRING_SEED: int = 42
SPRING_ITER: int = 120
SPRING_K: float | None = None  # None -> size-based heuristic

TARGET_WIDTH_CM: float = 12.0
TARGET_HEIGHT_CM: float = 8.0

# Bipartite layout
BIPARTITE_LEFT_X: float = 0.0
BIPARTITE_RIGHT_X: float = 650.0
BIPARTITE_Y_GAP: float = 54.0

# Size controls (for LaTeX compilation sanity)
MAX_NODES: int | None = None   # e.g. 120
MAX_EDGES: int | None = None   # e.g. 250

# Labeling (avoid clutter)
MAX_LABELED_NODES: int = 80    # label only top nodes if graph is larger

# Styling knobs
NODE_FONT: str = r"\scriptsize"
NODE_INNER_SEP: str = "1.2pt"

EDGE_WIDTH_MIN_PT: float = 0.25
EDGE_WIDTH_MAX_PT: float = 1.25
EDGE_OPACITY_MIN: float = 0.15
EDGE_OPACITY_MAX: float = 0.65

# Node "importance" for pruning/labeling
LABEL_TOP_BY: str = "raw_weighted_degree"  # fallback to raw_degree / degree


# ----------------------------
# TeX escaping
# ----------------------------

_TEX_ESCAPE_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def tex_escape(s: str) -> str:
    return "".join(_TEX_ESCAPE_MAP.get(ch, ch) for ch in s)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Defaults from dashboard config (best-effort)
# ----------------------------

def _load_dashboard_defaults() -> dict[str, Any]:
    out: dict[str, Any] = {
        "CONCEPT": "Urban Digital Twin",
        "USE_CACHE": False,
        "DEFAULT_MIN_VALUE": 5,
        "DEFAULT_SIM_THRESHOLD": 0.5,
        "DEFAULT_MIN_KW_FREQ": 5,
        "DEFAULT_TOP_K": None,
        "W2V_CONFIG": Word2VecConfig(
            vector_size=100, window=5, min_count=2, workers=4, sg=1, seed=42, epochs=50
        ),
    }
    try:
        from dashboard import config as dcfg  # type: ignore

        for k in list(out.keys()):
            if hasattr(dcfg, k):
                out[k] = getattr(dcfg, k)
    except Exception:
        pass
    return out


# ----------------------------
# Graph selection + pruning
# ----------------------------

def _node_score(G: nx.Graph, n: Any) -> float:
    d = G.nodes[n]
    attr = LABEL_TOP_BY
    if attr in d and isinstance(d[attr], (int, float)):
        return float(d[attr])
    if "raw_weighted_degree" in d and isinstance(d["raw_weighted_degree"], (int, float)):
        return float(d["raw_weighted_degree"])
    if "raw_degree" in d and isinstance(d["raw_degree"], (int, float)):
        return float(d["raw_degree"])
    return float(G.degree(n))


def _rank_nodes(G: nx.Graph) -> List[Any]:
    return sorted(G.nodes(), key=lambda n: _node_score(G, n), reverse=True)


def restrict_graph(G: nx.Graph) -> nx.Graph:
    if MAX_NODES is None and MAX_EDGES is None:
        return G

    H = G.copy()

    if MAX_NODES is not None and H.number_of_nodes() > MAX_NODES:
        keep = set(_rank_nodes(H)[: int(MAX_NODES)])
        H = H.subgraph(keep).copy()

    if MAX_EDGES is not None and H.number_of_edges() > MAX_EDGES:
        edges = []
        for u, v, data in H.edges(data=True):
            w = data.get("raw_weight")
            w = float(w) if isinstance(w, (int, float)) else 1.0
            edges.append((w, u, v))
        edges.sort(reverse=True)
        keep_edges = {(u, v) for _, u, v in edges[: int(MAX_EDGES)]}

        H2 = nx.Graph()
        H2.add_nodes_from(H.nodes(data=True))
        for u, v, data in H.edges(data=True):
            if (u, v) in keep_edges or (v, u) in keep_edges:
                H2.add_edge(u, v, **data)
        H2.remove_nodes_from(list(nx.isolates(H2)))
        H = H2

    return H


# ----------------------------
# Layout
# ----------------------------

def spring_positions(G: nx.Graph) -> Dict[Any, Tuple[float, float]]:
    if G.number_of_nodes() == 0:
        return {}
    n = G.number_of_nodes()
    k = SPRING_K if SPRING_K is not None else (2.0 / math.sqrt(max(n, 1)))
    pos = nx.spring_layout(G, seed=SPRING_SEED, iterations=SPRING_ITER, k=k, scale=1.0)
    return {node: (float(x), float(y)) for node, (x, y) in pos.items()}


def bipartite_positions(G: nx.Graph) -> Dict[Any, Tuple[float, float]]:
    if G.number_of_nodes() == 0:
        return {}

    concepts: List[Any] = []
    methods: List[Any] = []

    for n, d in G.nodes(data=True):
        t = d.get("node_type") or d.get("kind") or ""
        if str(t).lower().startswith("concept") or d.get("bipartite") == 0:
            concepts.append(n)
        else:
            methods.append(n)

    concepts.sort(key=lambda n: _node_score(G, n), reverse=True)
    methods.sort(key=lambda n: _node_score(G, n), reverse=True)

    # helper expects lists of node IDs; nodes are typically strings already
    pos_cyto = bipartite_column_positions(
        [str(n) for n in concepts],
        [str(n) for n in methods],
        left_x=float(BIPARTITE_LEFT_X),
        right_x=float(BIPARTITE_RIGHT_X),
        y_gap=float(BIPARTITE_Y_GAP),
    )
    # Map back to node objects (by string match)
    # This is safe for your graphs (nodes are strings). If nodes ever become non-strings,
    # you may want a more explicit mapping.
    pos: Dict[Any, Tuple[float, float]] = {}
    for n in G.nodes():
        s = str(n)
        if s in pos_cyto:
            pos[n] = (float(pos_cyto[s]["x"]), float(pos_cyto[s]["y"]))
    return pos


def normalize_positions(
    pos: Dict[Any, Tuple[float, float]],
    *,
    target_w_cm: float,
    target_h_cm: float,
) -> Dict[Any, Tuple[float, float]]:
    if not pos:
        return {}
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    w = max(xmax - xmin, 1e-9)
    h = max(ymax - ymin, 1e-9)

    xmid = (xmax + xmin) / 2.0
    ymid = (ymax + ymin) / 2.0

    scale = min(target_w_cm / w, target_h_cm / h)
    return {k: ((x - xmid) * scale, (y - ymid) * scale) for k, (x, y) in pos.items()}


def choose_layout(G: nx.Graph, name: str) -> Dict[Any, Tuple[float, float]]:
    # Concept-method graphs are bipartite
    if name.startswith("cm_"):
        return bipartite_positions(G)
    # Semantic/cooccurrence are not
    return spring_positions(G)


# ----------------------------
# Styling for nx.to_latex_raw
# ----------------------------

def edge_options_by_weight(G: nx.Graph) -> Dict[Tuple[Any, Any], str]:
    weights: List[float] = []
    for _, _, d in G.edges(data=True):
        w = d.get("raw_weight")
        if isinstance(w, (int, float)):
            weights.append(float(w))

    if not weights:
        return {}

    vmin = min(weights)
    vmax = max(weights)
    if vmax == vmin:
        lw = (EDGE_WIDTH_MIN_PT + EDGE_WIDTH_MAX_PT) / 2.0
        op = (EDGE_OPACITY_MIN + EDGE_OPACITY_MAX) / 2.0
        return {e: f"line width={lw:.3f}pt,opacity={op:.3f}" for e in G.edges()}

    out: Dict[Tuple[Any, Any], str] = {}
    for u, v, d in G.edges(data=True):
        w = d.get("raw_weight")
        wv = float(w) if isinstance(w, (int, float)) else vmin
        t = (wv - vmin) / (vmax - vmin)
        lw = EDGE_WIDTH_MIN_PT + t * (EDGE_WIDTH_MAX_PT - EDGE_WIDTH_MIN_PT)
        op = EDGE_OPACITY_MIN + t * (EDGE_OPACITY_MAX - EDGE_OPACITY_MIN)
        out[(u, v)] = f"line width={lw:.3f}pt,opacity={op:.3f}"
    return out


def node_style_for(d: dict[str, Any], *, default: str = "kw") -> str:
    t = str(d.get("node_type") or d.get("kind") or "").lower()
    if t.startswith("concept"):
        return "concept"
    if t.startswith("method"):
        return "method"
    return default


def tikz_options_with_styles() -> str:
    # tikz_options is passed without brackets; NetworkX wraps it in [...]
    return (
        "x=1cm,y=1cm,"
        f"kw/.style={{draw,circle,inner sep={NODE_INNER_SEP},font={NODE_FONT},fill=black!5}},"
        f"concept/.style={{draw,circle,inner sep={NODE_INNER_SEP},font={NODE_FONT},fill=blue!10}},"
        f"method/.style={{draw,rounded corners=1pt,rectangle,inner sep={NODE_INNER_SEP},font={NODE_FONT},fill=orange!12}}"
    )


# ----------------------------
# Build graphs
# ----------------------------

@dataclass(frozen=True)
class SemanticParams:
    sim_threshold: float
    min_keyword_freq: int
    top_k: int | None


def build_graphs(
    corpus: PublicationCorpus,
    *,
    min_value: int,
    sem: SemanticParams,
    w2v_config: Word2VecConfig,
    years: List[int],
) -> Dict[str, nx.Graph]:
    graphs: Dict[str, nx.Graph] = {}

    if EXPORT_COOCCURRENCE:
        graphs[f"coocc_overall_min{min_value}"] = corpus.cooccurrence_network(min_value=min_value)

    if EXPORT_CONCEPT_METHOD:
        graphs[f"cm_overall_min{min_value}"] = corpus.concept_method_network(min_value=min_value)

    if EXPORT_SEMANTIC:
        graphs[f"sem_overall_th{sem.sim_threshold:.2f}_mkf{sem.min_keyword_freq}"] = corpus.semantic_similarity_network(
            w2v_config=w2v_config,
            min_keyword_freq=sem.min_keyword_freq,
            similarity_threshold=sem.sim_threshold,
            top_k=sem.top_k,
        )

    if EXPORT_TEMPORAL:
        if EXPORT_COOCCURRENCE:
            temp = corpus.temporal_network(min_value=min_value)
            for y in years:
                if y in temp:
                    graphs[f"coocc_yearly_{y}_min{min_value}"] = temp[y]

        if EXPORT_CONCEPT_METHOD:
            temp = corpus.temporal_concept_method_network(min_value=min_value)
            for y in years:
                if y in temp:
                    graphs[f"cm_yearly_{y}_min{min_value}"] = temp[y]

        if EXPORT_SEMANTIC:
            temp = corpus.temporal_semantic_similarity_network(
                w2v_config=w2v_config,
                min_keyword_freq=sem.min_keyword_freq,
                similarity_threshold=sem.sim_threshold,
                top_k=sem.top_k,
            )
            for y in years:
                if y in temp:
                    graphs[f"sem_yearly_{y}_th{sem.sim_threshold:.2f}_mkf{sem.min_keyword_freq}"] = temp[y]

    return graphs


# ----------------------------
# Export one graph to .tex via nx.to_latex_raw
# ----------------------------

def export_graph_tex(name: str, G0: nx.Graph, out_dir: Path) -> None:
    G = restrict_graph(G0)

    # Choose labels: only top nodes if large
    ranked = _rank_nodes(G)
    if G.number_of_nodes() <= MAX_LABELED_NODES:
        labeled = set(G.nodes())
    else:
        labeled = set(ranked[: MAX_LABELED_NODES])

    # Compute layout on original graph
    pos_raw = choose_layout(G, name)
    pos_cm = normalize_positions(pos_raw, target_w_cm=TARGET_WIDTH_CM, target_h_cm=TARGET_HEIGHT_CM)

    # Relabel nodes to safe TikZ identifiers
    mapping = {n: f"n{idx}" for idx, n in enumerate(G.nodes())}
    H = nx.relabel_nodes(G, mapping, copy=True)

    # Positions dict keyed by new node ids
    pos_new: Dict[str, Tuple[float, float]] = {}
    for old, new in mapping.items():
        if old in pos_cm:
            pos_new[new] = pos_cm[old]

    # Node options + labels dicts keyed by new node ids
    node_opts: Dict[str, str] = {}
    node_lbls: Dict[str, str] = {}
    for old, new in mapping.items():
        d = G.nodes[old]
        node_opts[new] = node_style_for(d, default="kw")
        node_lbls[new] = tex_escape(str(old)) if old in labeled else ""

    # Edge options keyed by new edges
    edge_opts_old = edge_options_by_weight(G)
    edge_opts: Dict[Tuple[str, str], str] = {}
    for (u, v), opt in edge_opts_old.items():
        edge_opts[(mapping[u], mapping[v])] = opt

    # Create TikZ via NetworkX
    tikz = to_latex_raw(
        H,
        pos=pos_new,
        tikz_options=tikz_options_with_styles(),
        default_node_options="",               # node path options
        node_options=node_opts,                # per-node (e.g. kw/concept/method)
        node_label=node_lbls,                  # dict: avoid fallback to node id
        default_edge_options="line cap=round", # becomes [-,line cap=round]
        edge_options=edge_opts,
        edge_label={},                         # no edge labels
        edge_label_options={},
    )

    header = (
        "% Auto-generated by export_dashboard_networks_latex.py (nx.to_latex_raw)\n"
        f"% {name} | nodes={G.number_of_nodes()} edges={G.number_of_edges()}\n"
    )

    out_path = out_dir / f"{name}.tex"
    out_path.write_text(header + tikz, encoding="utf-8")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    defaults = _load_dashboard_defaults()

    concept = defaults["CONCEPT"]
    use_cache = bool(defaults["USE_CACHE"])
    min_value = int(MIN_VALUE if MIN_VALUE is not None else defaults["DEFAULT_MIN_VALUE"])

    sem = SemanticParams(
        sim_threshold=float(SIM_THRESHOLD if SIM_THRESHOLD is not None else defaults["DEFAULT_SIM_THRESHOLD"]),
        min_keyword_freq=int(MIN_KEYWORD_FREQ if MIN_KEYWORD_FREQ is not None else defaults["DEFAULT_MIN_KW_FREQ"]),
        top_k=TOP_K if TOP_K is not None else defaults["DEFAULT_TOP_K"],
    )
    w2v_config = W2V_CONFIG if W2V_CONFIG is not None else defaults["W2V_CONFIG"]

    corpus = PublicationCorpus(concept, use_cache=use_cache)
    years = YEARS[:] if YEARS else sorted(corpus.years().keys())

    graphs = build_graphs(
        corpus,
        min_value=min_value,
        sem=sem,
        w2v_config=w2v_config,
        years=years,
    )

    ensure_dir(OUTPUT_DIR)

    for name, G in graphs.items():
        export_graph_tex(name, G, OUTPUT_DIR)

    print(f"Wrote {len(graphs)} TikZ files to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
