#!/usr/bin/env python3
"""
wordnet_test.py

Quick-and-dirty exploration of whether there are very similar keywords
according to WordNet (ontology-based semantic similarity).

Usage:
    python wordnet_test.py
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from pathlib import Path

import nltk
from nltk.corpus import wordnet as wn

# Adjust imports to match your package layout
from data.publication_corpus import PublicationCorpus


def ensure_wordnet_downloaded() -> None:
    """Make sure the WordNet data is available."""
    try:
        wn.synsets("city")
    except LookupError:
        print("Downloading NLTK WordNet data...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def collect_keywords(
    concept: str,
    min_freq: int = 3,
    max_keywords: int | None = 200,
) -> list[str]:
    """
    Load the corpus for a given concept and return a list of keywords
    that are frequent enough and likely to be supported by WordNet.

    For a first test we:
      - lowercase
      - keep single-word keywords only (WordNet works best here)
      - require at least `min_freq` occurrences
      - optionally truncate to `max_keywords` most frequent
    """
    corpus = PublicationCorpus(concept=concept, use_cache=False)
    kw_counter: Counter[str] = Counter()

    for pub in corpus:
        for kw in pub.keywords:
            if not kw:
                continue
            k = kw.strip().lower()
            if not k:
                continue
            kw_counter[k] += 1

    # filter by frequency and single-word
    filtered = [
        (kw, freq)
        for kw, freq in kw_counter.items()
        if freq >= min_freq and " " not in kw and "-" not in kw
    ]

    # sort by frequency, descending
    filtered.sort(key=lambda x: x[1], reverse=True)

    if max_keywords is not None:
        filtered = filtered[:max_keywords]

    print(f"Total distinct keywords: {len(kw_counter)}")
    print(f"Keywords after filters: {len(filtered)}")
    print("Top 10 keywords:", filtered[:10])

    return [kw for kw, _ in filtered]


def best_wup_similarity(word1: str, word2: str) -> float:
    """
    Compute the maximum Wu-Palmer similarity between any pair of synsets
    of word1 and word2 in WordNet.

    Returns 0.0 if no synsets or no similarity is defined.
    """
    syns1 = wn.synsets(word1)
    syns2 = wn.synsets(word2)
    if not syns1 or not syns2:
        return 0.0

    best = 0.0
    for s1 in syns1:
        for s2 in syns2:
            sim = s1.wup_similarity(s2)
            if sim is None:
                continue
            if sim > best:
                best = sim
    return best


def find_similar_pairs(
    keywords: list[str],
    min_similarity: float = 0.9,
    top_n: int = 50,
) -> list[tuple[str, str, float]]:
    """
    Compute pairwise Wu-Palmer similarities between all keyword pairs,
    and return the top_n pairs with similarity >= min_similarity.
    """
    pairs: list[tuple[str, str, float]] = []

    total = len(keywords) * (len(keywords) - 1) // 2
    print(f"Computing similarities for {total} pairs...")

    checked = 0
    for w1, w2 in combinations(keywords, 2):
        sim = best_wup_similarity(w1, w2)
        if sim >= min_similarity:
            pairs.append((w1, w2, sim))

        checked += 1
        if checked % 5000 == 0:
            print(f"  processed {checked}/{total} pairs...")

    # sort by similarity descending
    pairs.sort(key=lambda x: x[2], reverse=True)
    if top_n is not None:
        pairs = pairs[:top_n]

    return pairs


def main() -> None:
    ensure_wordnet_downloaded()

    # 1) Collect candidate keywords from your UDT corpus
    concept = "Urban Digital Twin"  # adjust if needed
    keywords = collect_keywords(
        concept=concept,
        min_freq=3,  # require at least 3 occurrences
        max_keywords=200,  # limit for speed; adjust as you like
    )

    # 2) Compute WordNet-based similarity between keyword pairs
    similar_pairs = find_similar_pairs(
        keywords,
        min_similarity=0.9,  # very high: almost synonyms
        top_n=1000,
    )

    # 3) Print results
    if not similar_pairs:
        print("No keyword pairs above the similarity threshold.")
        return

    print("\nMost similar keyword pairs according to WordNet (Wu–Palmer):")
    for w1, w2, sim in similar_pairs:
        print(f"  {w1!r:25s}  ~  {w2!r:25s}   sim={sim:.3f}")


if __name__ == "__main__":
    main()
