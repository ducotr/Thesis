from pathlib import Path
from time import sleep

import networkx as nx
from pyvis.network import Network

from analysis.embeddings import Word2VecConfig
from analysis.metrics import (
    CommunityDetection,
    EdgeScaling,
    NodeScaling,
)
from data.publication_corpus import PublicationCorpus


def generate_group_colors(n: int) -> list[str]:
    colors = []

    s = 0.6
    v = 0.9

    for i in range(n):
        h = i / n % 1.0
        i = int(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i % 6

        if i == 0:
            r, g, b = v, t, p
        elif i == 1:
            r, g, b = q, v, p
        elif i == 2:
            r, g, b = p, v, t
        elif i == 3:
            r, g, b = p, q, v
        elif i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q

        colors.append(
            "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        )

    return colors


def blend_hex(c1: str, c2: str, t: float = 0.5) -> str:
    """
    Blend two hex colors c1 and c2 with factor t in [0, 1].
    t = 0   -> all c1
    t = 1   -> all c2
    t = 0.5 -> 50/50 blend
    """

    c1 = c1.lstrip("#")
    c2 = c2.lstrip("#")

    if len(c1) != 6 or len(c2) != 6:
        raise ValueError("Only 6-digit hex colors like #RRGGBB are supported.")

    r1, g1, b1 = int(c1[0:2], 16), int(c1[2:4], 16), int(c1[4:6], 16)
    r2, g2, b2 = int(c2[0:2], 16), int(c2[2:4], 16), int(c2[4:6], 16)

    r = round((1 - t) * r1 + t * r2)
    g = round((1 - t) * g1 + t * g2)
    b = round((1 - t) * b1 + t * b2)

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    return f"#{r:02x}{g:02x}{b:02x}"


def visualize_network(
    G: nx.Graph,
    node_scaling: NodeScaling = NodeScaling.NONE,
    edge_scaling: EdgeScaling = EdgeScaling.NONE,
    community_detection: CommunityDetection = CommunityDetection.NONE,
) -> None:

    groups = sorted(set(data for _, data in G.nodes(data=community_detection)))
    colors = {
        group: color for group, color in zip(groups, generate_group_colors(len(groups)))
    }

    for _, data in G.nodes(data=True):
        data["size"] = data[node_scaling]
        data["color"] = colors[data[community_detection]]

    for u, v, data in G.edges(data=True):
        data["weight"] = data[edge_scaling]
        size_u = float(G.nodes[u].get("size", 1.0))
        size_v = float(G.nodes[v].get("size", 1.0))
        den = size_u + size_v or 1.0
        t = max(0.0, min(1.0, size_v / den))

        G.edges[u, v]["color"] = blend_hex(
            c1=G.nodes[u]["color"],
            c2=G.nodes[v]["color"],
            t=t,
        )

    nt = Network("938px", "100%")
    nt.from_nx(G, edge_scaling=True)
    nt.show("nx.html", notebook=False)


def visualize_bipartite_network(
    G: nx.Graph,
    node_scaling: NodeScaling = NodeScaling.NONE,
    edge_scaling: EdgeScaling = EdgeScaling.NONE,
    community_detection: CommunityDetection = CommunityDetection.NONE,
    html_name: str | None = None,
) -> None:
    """
    Visualise a bipartite network with the two partitions laid out in columns.

    Expects nodes to carry:
        - bipartite: 0/1 partition indicator
        - node_type: "concept" / "method" (used for shape)
    """
    left = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    right = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]
    if not left or not right:
        raise ValueError("Graph must contain both bipartite partitions")

    html_name = html_name or "nx_bipartite.html"

    # Fixed layout keeps the two partitions separated
    pos = nx.bipartite_layout(G, left, scale=400)

    for n, data in G.nodes(data=True):
        data["size"] = data[node_scaling]
        data["group"] = data.get(community_detection, 0)
        data["x"] = pos[n][0] * 2  # widen gap between partitions
        data["y"] = pos[n][1] * 2
        data["physics"] = False  # preserve bipartite layout

        ntype = data.get("node_type")
        if ntype == "method":
            data["shape"] = "square"
        elif ntype == "concept":
            data["shape"] = "dot"

    for _, _, data in G.edges(data=True):
        data["weight"] = data[edge_scaling]

    nt = Network("938px", "100%")
    nt.from_nx(G, edge_scaling=True)
    nt.toggle_physics(False)
    nt.show(html_name, notebook=False)


def visualize_temporal_year(
    temporal_network: dict[int, nx.Graph],
    year: int,
    node_scaling: NodeScaling = NodeScaling.EIGENVECTOR,
    edge_scaling: EdgeScaling = EdgeScaling.LINEAR,
    community_detection: CommunityDetection = CommunityDetection.LOUVAIN,
    html_name: str | None = None,
) -> None:
    """
    Visualise the co-occurrence network for a single year, highlighting
    birth and death nodes using different shapes.

    - births: nodes that appear this year but not in the previous year
    - deaths: nodes that appear this year but not in the next year
    (as precomputed in build_temporal_network).
    """
    if year not in temporal_network:
        raise ValueError(f"Year {year} not in temporal network")

    G = temporal_network[year]

    births = set(G.graph.get("births", []))
    deaths = set(G.graph.get("deaths", []))

    # Fallback HTML file name
    if html_name is None:
        html_name = f"nx_{year}.html"

    # --- Node styling ---
    for n, data in G.nodes(data=True):
        # size by chosen centrality
        data["size"] = data[node_scaling]
        # community -> group (color)
        data["group"] = data.get(community_detection, 0)

        # shape encodes temporal status
        if n in births:
            # "born" this year
            data["shape"] = "triangle"
        elif n in deaths:
            # "dies" after this year
            data["shape"] = "triangleDown"
        else:
            # regular node
            data["shape"] = "dot"

        # Optional: add tooltip text
        flags = []
        if n in births:
            flags.append("birth")
        if n in deaths:
            flags.append("death")
        if flags:
            data["title"] = f"{n} ({', '.join(flags)})"
        else:
            data.setdefault("title", n)

    # --- Edge styling ---
    for u, v, data in G.edges(data=True):
        data["weight"] = data[edge_scaling]

    # Build and show PyVis network
    nt = Network("938px", "100%")
    nt.from_nx(G, edge_scaling=True)
    nt.show(html_name, notebook=False)


def graph_stats(G: nx.Graph) -> dict:
    n = G.number_of_nodes()
    m = G.number_of_edges()

    stats: dict[str, float | int] = {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(G) if n > 1 else 0.0,
    }

    if n == 0:
        stats.update(
            {
                "avg_degree": 0.0,
                "max_degree": 0,
                "n_components": 0,
                "largest_cc_size": 0,
                "largest_cc_frac": 0.0,
                "avg_clustering": 0.0,
            }
        )
        return stats

    degrees = [deg for _, deg in G.degree()]
    stats["avg_degree"] = sum(degrees) / n
    stats["max_degree"] = max(degrees) if degrees else 0

    if m > 0:
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        stats["n_components"] = len(components)
        stats["largest_cc_size"] = len(largest)
        stats["largest_cc_frac"] = len(largest) / n

        stats["avg_clustering"] = nx.average_clustering(G)
    else:
        stats.update(
            {
                "n_components": n,
                "largest_cc_size": 1,
                "largest_cc_frac": 1.0 / n,
                "avg_clustering": 0.0,
            }
        )

    return stats


if __name__ == "__main__":
    pc = PublicationCorpus(concept="Urban Digital Twin", use_cache=False)

    # G_bipartite = pc.concept_method_network(min_value=5)

    # visualize_bipartite_network(
    #     G=G_bipartite,
    #     node_scaling=NodeScaling.EIGENVECTOR,
    #     edge_scaling=EdgeScaling.LINEAR,
    #     community_detection=CommunityDetection.NONE
    # )

    # Define Word2Vec configuration
    w2v_config = Word2VecConfig(
        vector_size=100, window=5, min_count=2, workers=4, sg=1, seed=None, epochs=50
    )

    # --- (All time) semantic network ---
    G_semantic = pc.semantic_similarity_network(
        w2v_config=w2v_config, min_keyword_freq=1, similarity_threshold=0.5, top_k=5
    )

    stats = graph_stats(G_semantic)
    print(f"\n=== Title ===")
    for key, value in stats.items():
        print(f"{key:20s}: {value}")

    visualize_network(
        G=G_semantic,
        node_scaling=NodeScaling.EIGENVECTOR,
        edge_scaling=EdgeScaling.LINEAR,
        community_detection=CommunityDetection.NONE,
    )

    # --- Temporal semantic networks ---
    # G_temporal_semantic = pc.temporal_semantic_similarity_network(
    #     w2v_config=w2v_config,
    #     min_keyword_freq=2,
    #     similarity_threshold=0.5,
    #     top_k=3
    # )

    # for year, G in G_temporal_semantic.items():
    #     stats = graph_stats(G)
    #     print(f"\n=== Year {year} ===")
    #     for k, v in stats.items():
    #         print(f"{k:20s}: {v}")
    #     print("jaccard_prev      :", G.graph.get("jaccard_prev"))
    #     print("births            :", len(G.graph.get("births", [])))
    #     print("deaths            :", len(G.graph.get("deaths", [])))
    #     print("sample nodes      :", list(G.nodes)[:10])

    #     visualize_network(
    #         G=G,
    #         node_scaling=NodeScaling.EIGENVECTOR,
    #         edge_scaling=EdgeScaling.LINEAR,
    #         community_detection=CommunityDetection.LOUVAIN,
    #     )
    #     sleep(0.5)
