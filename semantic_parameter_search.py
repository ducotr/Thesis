from pathlib import Path

import networkx as nx

from analysis.embeddings import train_word2vec
from analysis.networks import build_semantic_similarity_network
from data.publication_corpus import PublicationCorpus


def graph_stats(G: nx.Graph) -> dict:
    """Compute some basic structural statistics for a graph."""
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
                "modularity": 0.0,
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

        # Louvain modularity as a proxy for community structure quality
        comms = nx.community.louvain_communities(G, weight="raw_weight")
        stats["modularity"] = nx.community.modularity(G, comms, weight="raw_weight")
    else:
        stats.update(
            {
                "n_components": n,
                "largest_cc_size": 1,
                "largest_cc_frac": 1.0 / n,
                "avg_clustering": 0.0,
                "modularity": 0.0,
            }
        )

    return stats


def main() -> None:
    pc = PublicationCorpus(concept="Urban Digital Twin", use_cache=True)
    publications = pc.data  # or pc._data if you prefer

    # A small, targeted set of parameter configs to compare
    configs = [
        {
            "name": "baseline",
            "vector_size": 100,
            "window": 5,
            "min_count": 3,
            "sg": 1,
            "keywords_weight": 2,
            "min_keyword_freq": 5,
            "similarity_threshold": 0.7,
        },
        {
            "name": "small_window",
            "vector_size": 100,
            "window": 3,
            "min_count": 3,
            "sg": 1,
            "keywords_weight": 2,
            "min_keyword_freq": 5,
            "similarity_threshold": 0.7,
        },
        {
            "name": "large_window_looser_threshold",
            "vector_size": 100,
            "window": 8,
            "min_count": 3,
            "sg": 1,
            "keywords_weight": 2,
            "min_keyword_freq": 5,
            "similarity_threshold": 0.6,
        },
        {
            "name": "smaller_vectors_higher_min_count",
            "vector_size": 50,
            "window": 5,
            "min_count": 5,
            "sg": 1,
            "keywords_weight": 2,
            "min_keyword_freq": 5,
            "similarity_threshold": 0.7,
        },
        {
            "name": "larger_vectors_lower_min_count",
            "vector_size": 200,
            "window": 5,
            "min_count": 2,
            "sg": 1,
            "keywords_weight": 2,
            "min_keyword_freq": 5,
            "similarity_threshold": 0.7,
        },
        {
            "name": "CBOW_tighter_threshold",
            "vector_size": 100,
            "window": 5,
            "min_count": 3,
            "sg": 0,  # CBOW
            "keywords_weight": 2,
            "min_keyword_freq": 5,
            "similarity_threshold": 0.8,
        },
    ]

    models_dir = Path("models/param_search")
    models_dir.mkdir(parents=True, exist_ok=True)

    for cfg in configs:
        print("\n=== Config:", cfg["name"], "===")

        model_path = models_dir / f"{cfg['name']}.model"

        # Train a Word2Vec model for this config
        model = train_word2vec(
            publications=publications,
            model_path=model_path,
            vector_size=cfg["vector_size"],
            window=cfg["window"],
            min_count=cfg["min_count"],
            sg=cfg["sg"],
            keywords_weight=cfg["keywords_weight"],
            # you can expose remove_stopwords here if you want to test it too
        )

        # Build the semantic similarity graph
        G = build_semantic_similarity_network(
            publications=publications,
            model=model,
            min_keyword_freq=cfg["min_keyword_freq"],
            similarity_threshold=cfg["similarity_threshold"],
        )

        # Compute and print stats
        stats = graph_stats(G)
        for k, v in stats.items():
            print(f"{k:20s}: {v}")
        print("Sample nodes:", list(G.nodes)[:10])


if __name__ == "__main__":
    main()
