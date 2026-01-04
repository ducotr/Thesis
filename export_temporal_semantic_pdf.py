# scripts/export_temporal_semantic_pdf.py

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_pdf import PdfPages

from analysis.embeddings import Word2VecConfig
from data.publication_corpus import PublicationCorpus


def build_temporal_semantic(pc: PublicationCorpus) -> dict[int, nx.Graph]:
    """
    Build temporal semantic similarity networks.

    Assumes your build_semantic_similarity_network supports top_k.
    If your PublicationCorpus.temporal_semantic_similarity_network already
    passes top_k through, you can call that instead.
    """
    from analysis.networks import (
        bucket_publications_by_year,
        build_semantic_similarity_network,
    )

    w2v_config = Word2VecConfig(
        vector_size=100,
        window=5,
        min_count=3,
        sg=1,
        epochs=50,
    )

    # you decided on k=3 and sim >= 0.5
    top_k = 3
    similarity_threshold = 0.5
    min_keyword_freq = 3

    buckets = bucket_publications_by_year(pc.data)
    temporal: dict[int, nx.Graph] = {}

    for year, pubs in buckets.items():
        G = build_semantic_similarity_network(
            publications=pubs,
            w2v_config=w2v_config,
            min_keyword_freq=min_keyword_freq,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
        )
        # skip empty years
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            continue

        # annotate for use in the plot
        G.graph["year"] = year
        temporal[year] = G

    return dict(sorted(temporal.items()))


def draw_network_on_ax(G: nx.Graph, ax: plt.Axes) -> None:
    """
    Draw a single semantic network on the given Axes.
    Uses the precomputed node sizes and communities from your metrics.py.
    """
    ax.set_axis_off()

    # Stable layout so comparisons across years are easier
    pos = nx.spring_layout(G, seed=42, k=0.4)

    # Node sizes: scale from the eigenvector centrality stored by add_node_sizes
    size_attr = "NODESCALING.EIGENVECTOR"
    node_sizes = [G.nodes[n].get(size_attr, 30) for n in G.nodes()]

    # Communities: Louvain community labels from add_communities
    comm_attr = "COMMUNITYDETECTION.LOUVAIN"
    communities = [G.nodes[n].get(comm_attr, 0) for n in G.nodes()]

    # Edges: use linear_width from add_edge_widths if present
    edge_widths = [G.edges[e].get("linear_width", 1.0) for e in G.edges()]

    # Draw
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=communities,  # coloured by community
        cmap=plt.cm.tab20,
        ax=ax,
    )
    nodes.set_edgecolor("black")
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,
        alpha=0.6,
        ax=ax,
    )

    # Keep labels only if you really want them (can be cluttered for 100+ nodes)
    # nx.draw_networkx_labels(G, pos, font_size=6, ax=ax)


def main() -> None:
    # 1) Load / build your Urban Digital Twin corpus (uses cache if present)
    concept = "Urban Digital Twin"
    pc = PublicationCorpus(concept, use_cache=True)

    # 2) Build temporal semantic networks
    #    (if you already added a .temporal_semantic_similarity_network(top_k=3,...),
    #     you can call that instead of build_temporal_semantic())
    temporal = build_temporal_semantic(pc)

    # 3) Export one figure per year into a multi-page PDF
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "temporal_semantic_networks_udt.pdf"

    with PdfPages(pdf_path) as pdf:
        for year, G in temporal.items():
            fig, ax = plt.subplots(figsize=(8.27, 11.69))  # A4 portrait in inches

            draw_network_on_ax(G, ax)

            # Title with year + some metadata if available
            jacc = G.graph.get("jaccard_prev", None)
            births = len(G.graph.get("births", []))
            deaths = len(G.graph.get("deaths", []))

            subtitle_parts = []
            if jacc is not None:
                subtitle_parts.append(f"Jaccard vs prev: {jacc:.2f}")
            subtitle_parts.append(f"n={G.number_of_nodes()}, m={G.number_of_edges()}")
            subtitle_parts.append(f"births={births}, deaths={deaths}")

            title = f"Urban Digital Twin semantic network - {year}"
            fig.suptitle(
                title + "\n" + " • ".join(subtitle_parts),
                fontsize=12,
            )

            fig.tight_layout(rect=[0, 0.02, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved PDF to: {pdf_path}")


if __name__ == "__main__":
    main()
