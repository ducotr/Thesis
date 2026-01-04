from pathlib import Path

import networkx as nx

from data.publication_corpus import PublicationCorpus


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
    pc = PublicationCorpus(concept="Urban Digital Twin", use_cache=True)

    G_temporal_semantic = pc.temporal_semantic_similarity_network(
        model_dir=Path("models/yearly_w2v"),
        min_keyword_freq=2,
        similarity_threshold=0.7,
    )

    for year, G in sorted(G_temporal_semantic.items()):
        stats = graph_stats(G)
        print(f"\n=== Year {year} ===")
        for k, v in stats.items():
            print(f"{k:20s}: {v}")
        print("jaccard_prev      :", G.graph.get("jaccard_prev"))
        print("births            :", len(G.graph.get("births", [])))
        print("deaths            :", len(G.graph.get("deaths", [])))
        print("sample nodes      :", list(G.nodes)[:10])
