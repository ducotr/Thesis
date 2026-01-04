# analysis/networks.py

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal

import networkx as nx
from gensim.models import Word2Vec

from analysis.embeddings import Word2VecConfig, keyword_frequencies, train_word2vec
from analysis.metrics import (
    add_communities,
    add_edge_widths,
    add_node_sizes,
    jaccard_similarity_temporal_network,
    node_birth_death,
)
from data.publication import Publication

KeywordKind = Literal["concept", "method"]

# ---------------------------------------------------------------------------
# Keyword classification (concept vs method)
# ---------------------------------------------------------------------------

# Load LLM-generated labels once (from keyword_labels.json in project root)
try:
    _LABELS: dict[str, str] = json.loads(
        Path("classification/keyword_labels.json").read_text(encoding="utf-8")
    )
except FileNotFoundError:
    _LABELS = {}


def llm_classify_keyword(keyword: str) -> KeywordKind | None:
    """
    Map a keyword to 'concept', 'method', or None using LLM labels.

    Supports two formats for _LABELS[keyword]:

    1. Legacy single-label form:
       { "Urban resilience": "concept", "Agent-based model": "method", ... }

    2. Ensemble form (multiple models):
       {
         "3D city models": {
           "llama3": "method",
           "mistral": "method",
           "gemma2": "method",
           "phi3": "concept",
           "qwen2.5": "method"
         },
         ...
       }

    In the ensemble case, a strict majority vote is used:
    - if 'concept' has a strict majority -> returns 'concept'
    - if 'method' has a strict majority -> returns 'method'
    - otherwise -> treated as 'other' -> returns None
    """
    if not _LABELS:
        raise RuntimeError(
            "classification/keyword_labels.json not found, run keyword_classification.py first."
        )

    k = keyword.strip()
    if not k:
        return None

    raw = _LABELS.get(k)
    if raw is None:
        return None

    # --- Case 1: legacy single-label string ---
    if isinstance(raw, str):
        label = raw.strip().lower()
        if label == "concept":
            return "concept"
        if label == "method":
            return "method"
        # "other" or anything else
        return None

    # --- Case 2: ensemble dict: model_name -> label ---
    if isinstance(raw, dict):
        # collect model labels, normalised
        labels = [str(v).strip().lower() for v in raw.values() if v is not None]
        if not labels:
            return None

        counts = Counter(labels)
        winner, votes = counts.most_common(1)[0]
        total = len(labels)

        # strict majority
        if votes > total / 2:
            if winner == "concept":
                return "concept"
            if winner == "method":
                return "method"
            # majority "other" (or something else) -> treat as unlabelled
            return None

        # no strict majority -> treat as "other"
        return None

    # Unexpected type in _LABELS: be safe and treat as no label
    return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def bucket_publications_by_year(
    publications: List[Publication],
) -> Dict[int, List[Publication]]:
    """
    Group publications into {year -> [Publication, ...]}.
    """
    buckets: Dict[int, List[Publication]] = defaultdict(list)
    for pub in publications:
        year = pub.publication_date.year
        buckets[year].append(pub)

    return dict(sorted(buckets.items()))


def keywords_cooccurrence(
    publications: list[Publication],
) -> dict[tuple[str, str], int]:
    """
    Compute co-occurrence counts of keyword pairs across publications.

    Co-occurrence is defined at the publication level:
    each pair of keywords that appear together in a publication
    gets its count increased by 1 for that publication.
    """
    counts: dict[tuple[str, str], int] = {}

    for pub in publications:
        kws = list(pub.keywords)
        n = len(kws)
        for i in range(n):
            for j in range(i + 1, n):
                k1 = kws[i]
                k2 = kws[j]
                if k1 == k2:
                    continue
                # sort to keep (a, b) and (b, a) the same key
                pair = (k1, k2) if k1 < k2 else (k2, k1)
                counts[pair] = counts.get(pair, 0) + 1

    return counts


# ---------------------------------------------------------------------------
# Co-occurrence networks
# ---------------------------------------------------------------------------


def build_cooccurrence_network(
    publications: list[Publication],
    min_value: int = 0,
) -> nx.Graph:
    """
    Build a keyword co-occurrence network.

    - Nodes: keywords
    - Edges: connect keywords that co-occur in at least one publication
      with edge attribute 'raw_weight' = number of co-occurring publications.

    Edges with raw_weight < min_value are dropped.
    Node sizes, edge widths and community labels are precomputed.
    """
    if min_value < 0:
        min_value = 0

    G = nx.Graph()

    if min_value == 0:
        all_keywords: set[str] = set()
        for pub in publications:
            all_keywords.update(pub.keywords)
        G.add_nodes_from(all_keywords)

    counts = keywords_cooccurrence(publications)
    for (k1, k2), v in counts.items():
        if v >= min_value:
            G.add_edge(k1, k2, raw_weight=v)

    if not G.nodes:
        return G

    add_node_sizes(G, minimum=10, maximum=50)
    add_edge_widths(G, minimum=1, maximum=20)
    add_communities(G)

    return G


def build_temporal_network(
    publications: list[Publication],
    min_value: int = 0,
) -> dict[int, nx.Graph]:
    """
    Build a co-occurrence network per publication year.

    Annotates each yearly graph with:
      - G.graph["year"]
      - G.graph["jaccard_prev"]  (Jaccard similarity to previous year)
      - G.graph["births"]        (nodes appearing in this year)
      - G.graph["deaths"]        (nodes disappearing after this year)
    """
    buckets = bucket_publications_by_year(publications)

    temporal_network: dict[int, nx.Graph] = {}
    for year, bucket in buckets.items():
        temporal_network[year] = build_cooccurrence_network(
            publications=bucket,
            min_value=min_value,
        )

    # Ensure chronological order
    temporal_network = dict(sorted(temporal_network.items()))

    # --- attach temporal metrics ---
    jacc = jaccard_similarity_temporal_network(temporal_network)
    births_deaths = node_birth_death(temporal_network)

    for year, G in temporal_network.items():
        G.graph["year"] = year
        G.graph["jaccard_prev"] = jacc.get(year, 0.0)

        bd = births_deaths.get(year, {"births": [], "deaths": []})
        G.graph["births"] = bd.get("births", [])
        G.graph["deaths"] = bd.get("deaths", [])

    return temporal_network


# ---------------------------------------------------------------------------
# Concept-method bipartite networks
# ---------------------------------------------------------------------------


def build_concept_method_bipartite(
    publications: list[Publication],
    min_value: int = 0,
    classifier: Callable[[str], KeywordKind | None] = llm_classify_keyword,
) -> nx.Graph:
    """
    Build a bipartite network linking conceptual keywords to methodological/
    technological keywords.

    - Nodes:
        - one set of nodes for 'concept' keywords
        - one set of nodes for 'method' keywords
      Each node gets:
        - node_type = 'concept' or 'method'
        - bipartite = 0 (concept) or 1 (method)

    - Edges:
        - connect concept <-> method if they co-occur in at least one publication
        - raw_weight = number of publications where the pair co-occurs

    Edges with raw_weight < min_value are dropped.
    Isolated nodes are removed.
    Node sizes, edge widths and community labels are precomputed.
    """
    G = nx.Graph()

    for pub in publications:
        concepts: set[str] = set()
        methods: set[str] = set()

        for kw in pub.keywords:
            kind = classifier(kw)
            if kind == "concept":
                concepts.add(kw)
            elif kind == "method":
                methods.add(kw)

        # connect each concept with each method in this publication
        for c in concepts:
            if c not in G:
                G.add_node(c, node_type="concept", bipartite=0)
            for m in methods:
                if m not in G:
                    G.add_node(m, node_type="method", bipartite=1)

                if G.has_edge(c, m):
                    G[c][m]["raw_weight"] += 1
                else:
                    G.add_edge(c, m, raw_weight=1)

    # Optional thresholding on raw_weight
    if min_value > 0:
        to_remove = [
            (u, v)
            for u, v, data in G.edges(data=True)
            if data.get("raw_weight", 0) < min_value
        ]
        G.remove_edges_from(to_remove)

    # Remove isolated nodes (no edges left after thresholding)
    isolates = list(nx.isolates(G))
    if isolates:
        G.remove_nodes_from(isolates)

    if not G.nodes:
        return G

    add_node_sizes(G, minimum=10, maximum=50)
    add_edge_widths(G, minimum=1, maximum=20)
    add_communities(G)

    return G


def build_temporal_concept_method_bipartite(
    publications: list[Publication],
    min_value: int = 0,
    classifier: Callable[[str], KeywordKind | None] = llm_classify_keyword,
) -> dict[int, nx.Graph]:
    """
    Build a concept-method bipartite network per publication year,
    with the same temporal annotations as build_temporal_network.
    """
    buckets = bucket_publications_by_year(publications)

    temporal_network: dict[int, nx.Graph] = {}
    for year, bucket in buckets.items():
        temporal_network[year] = build_concept_method_bipartite(
            publications=bucket,
            min_value=min_value,
            classifier=classifier,
        )

    # Ensure chronological order
    temporal_network = dict(sorted(temporal_network.items()))

    # --- attach temporal metrics ---
    jacc = jaccard_similarity_temporal_network(temporal_network)
    births_deaths = node_birth_death(temporal_network)

    for year, G in temporal_network.items():
        G.graph["year"] = year
        G.graph["jaccard_prev"] = jacc.get(year, 0.0)

        bd = births_deaths.get(year, {"births": [], "deaths": []})
        G.graph["births"] = bd.get("births", [])
        G.graph["deaths"] = bd.get("deaths", [])

    return temporal_network


# ---------------------------------------------------------------------------
# Semantic similarity networks (Word2Vec-based)
# ---------------------------------------------------------------------------


def build_semantic_similarity_network(
    publications: list[Publication],
    w2v_config: Word2VecConfig,
    min_keyword_freq: int = 5,
    similarity_threshold: float = 0.5,
    top_k: int | None = None,
    model: Word2Vec | None = None,
) -> nx.Graph:
    """ """
    # If no model is provided, train a fresh one (legacy behaviour)
    if model is None:
        model = train_word2vec(
            publications=publications,
            config=w2v_config,
        )

    freqs: dict[str, int] = keyword_frequencies(publications)

    # Candidate keywords: frequent enough and present in the model vocabulary
    candidates = [
        kw
        for kw, count in freqs.items()
        if count >= min_keyword_freq and kw in model.wv
    ]

    G = nx.Graph()
    for kw in candidates:
        G.add_node(kw, frequency=freqs[kw])

    if not candidates:
        return G

    # --- Edge construction ---
    if top_k is None:
        for i, kw1 in enumerate(candidates):
            for kw2 in candidates[i + 1 :]:
                sim = float(model.wv.similarity(kw1, kw2))
                if sim >= similarity_threshold:
                    G.add_edge(kw1, kw2, raw_weight=sim)
    else:
        edge_sims: dict[tuple[str, str], float] = {}

        for kw1 in candidates:
            sims: list[tuple[str, float]] = []

            for kw2 in candidates:
                if kw1 == kw2:
                    continue
                sim = float(model.wv.similarity(kw1, kw2))
                if sim >= similarity_threshold:
                    sims.append((kw2, sim))

            sims.sort(key=lambda x: x[1], reverse=True)
            for kw2, sim in sims[:top_k]:
                u, v = sorted((kw1, kw2))
                current = edge_sims.get((u, v))
                if current is None or sim > current:
                    edge_sims[(u, v)] = sim

        for (u, v), sim in edge_sims.items():
            G.add_edge(u, v, raw_weight=sim)

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    if not G.nodes:
        return G

    add_node_sizes(G, minimum=10, maximum=50)
    add_edge_widths(G, minimum=1, maximum=20)
    add_communities(G)

    return G


def build_temporal_semantic_similarity_network(
    publications: list[Publication],
    w2v_config: Word2VecConfig,
    min_keyword_freq: int = 5,
    similarity_threshold: float = 0.5,
    top_k: int | None = None,
    models_by_year: dict[int, Word2Vec] | None = None,
) -> dict[int, nx.Graph]:
    """ """

    buckets = bucket_publications_by_year(publications)

    temporal_network: dict[int, nx.Graph] = {}

    for year, bucket in buckets.items():
        G_year = build_semantic_similarity_network(
            publications=bucket,
            w2v_config=w2v_config,
            min_keyword_freq=min_keyword_freq,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            model=(models_by_year.get(year) if models_by_year else None),
        )

        temporal_network[year] = G_year

    temporal_network = dict(sorted(temporal_network.items()))

    jacc = jaccard_similarity_temporal_network(temporal_network)
    births_deaths = node_birth_death(temporal_network)

    for year, G in temporal_network.items():
        G.graph["year"] = year
        G.graph["jaccard_prev"] = jacc.get(year, 0.0)

        bd = births_deaths.get(year, {"births": [], "deaths": []})
        G.graph["births"] = bd.get("births", [])
        G.graph["deaths"] = bd.get("deaths", [])

    return temporal_network
