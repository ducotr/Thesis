from pathlib import Path

import numpy as np

from analysis.embeddings import keyword_frequencies, load_word2vec
from analysis.networks import build_semantic_similarity_network
from data.publication_corpus import PublicationCorpus


def inspect_similarity_distribution(model_path: Path) -> None:
    pc = PublicationCorpus(concept="Urban Digital Twin", use_cache=True)
    publications = pc.data  # or pc._data

    # Use the same candidate selection as build_semantic_similarity_network
    model = load_word2vec(model_path)
    freqs = keyword_frequencies(publications)

    min_keyword_freq = 5  # same as in your experiments
    candidates = [
        kw
        for kw, count in freqs.items()
        if count >= min_keyword_freq and kw in model.wv
    ]

    print(f"Number of candidate keywords: {len(candidates)}")

    sims = []
    for i, kw1 in enumerate(candidates):
        for kw2 in candidates[i + 1 :]:
            sims.append(float(model.wv.similarity(kw1, kw2)))

    sims = np.array(sims)
    print("Similarity stats:")
    print("  min    :", sims.min())
    print("  max    :", sims.max())
    print("  mean   :", sims.mean())
    for q in [50, 75, 90, 95, 99]:
        print(f"  p{q:>2}   :", np.percentile(sims, q))


if __name__ == "__main__":
    model_path = Path("models/param_search/baseline.model")
    inspect_similarity_distribution(model_path)
