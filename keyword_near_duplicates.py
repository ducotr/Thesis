#!/usr/bin/env python3
"""
keyword_near_duplicates.py

Find very similar (near-identical) keywords based on string similarity,
to help you spot things like:
    - "3D city model" vs "3D city models"
    - "visualization" vs "visualisation"
    - "digital_twin" vs "digital twins"

Usage:
    python keyword_near_duplicates.py
"""

from __future__ import annotations

from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations

from data.publication_corpus import PublicationCorpus


def normalize_for_merge(kw: str) -> str:
    """
    Normalise a keyword for duplicate detection.

    - lowercase
    - strip leading/trailing whitespace
    - replace hyphens/underscores with spaces
    - collapse multiple spaces
    """
    kw = kw.strip().lower()
    if not kw:
        return ""

    # unify separators
    for ch in ["_", "-"]:
        kw = kw.replace(ch, " ")

    # collapse whitespace
    kw = " ".join(kw.split())
    return kw


def string_similarity(a: str, b: str) -> float:
    """
    String similarity in [0,1], using SequenceMatcher ratio.
    1.0 = identical
    """
    return SequenceMatcher(None, a, b).ratio()


def collect_keywords(concept: str, min_freq: int = 1) -> dict[str, int]:
    """
    Collect keywords and their raw frequencies from the corpus.
    Keys are the *original* keyword strings (not normalised).
    """
    corpus = PublicationCorpus(concept=concept, use_cache=False)
    counts: Counter[str] = Counter()

    for pub in corpus:
        for kw in pub.keywords:
            if not kw:
                continue
            k = kw.strip()
            if not k:
                continue
            counts[k] += 1

    # apply a minimum frequency filter to avoid super-rare noise
    filtered = {kw: freq for kw, freq in counts.items() if freq >= min_freq}

    print(f"Total distinct keywords: {len(counts)}")
    print(f"Keywords with freq >= {min_freq}: {len(filtered)}")
    return filtered


def find_near_duplicate_pairs(
    keywords: dict[str, int],
    min_similarity: float = 0.9,
    max_len_diff: int = 5,
) -> list[tuple[str, str, float, int, int]]:
    """
    Find pairs of near-duplicate keywords.

    - `keywords`: dict of original_kw -> frequency
    - `min_similarity`: threshold on similarity score in [0, 1]
    - `max_len_diff`: optional guard; skip pairs whose length difference
      is larger than this (fast way to avoid silly pairs)
    """
    items = list(keywords.items())
    norm_cache = {kw: normalize_for_merge(kw) for kw, _ in items}

    pairs: list[tuple[str, str, float, int, int]] = []

    total = len(items) * (len(items) - 1) // 2
    print(f"Checking {total} pairs...")

    checked = 0
    for (kw1, f1), (kw2, f2) in combinations(items, 2):
        n1 = norm_cache[kw1]
        n2 = norm_cache[kw2]

        if not n1 or not n2:
            continue

        # quick length filter to skip obviously different words
        if abs(len(n1) - len(n2)) > max_len_diff:
            continue

        sim = string_similarity(n1, n2)
        if sim >= min_similarity and n1 != n2:
            pairs.append((kw1, kw2, sim, f1, f2))

        checked += 1
        if checked % 5000 == 0:
            print(f"  processed {checked}/{total} pairs...")

    # sort by similarity descending, then by total frequency
    pairs.sort(key=lambda x: (x[2], x[3] + x[4]), reverse=True)
    return pairs


def main() -> None:
    concept = "Urban Digital Twin"  # adjust if needed

    keywords = collect_keywords(concept, min_freq=1)

    near_dups = find_near_duplicate_pairs(
        keywords,
        min_similarity=0.80,  # 0.9–0.95 is a good "near identical" range
        max_len_diff=5,
    )

    if not near_dups:
        print("No near-duplicate keyword pairs found above threshold.")
        return

    print("\nCandidate near-duplicate keyword pairs:")
    print("(sim  freq1  freq2  kw1  ~  kw2)")
    for kw1, kw2, sim, f1, f2 in near_dups:
        print(f"{sim:0.3f}  {f1:5d}  {f2:5d}  {kw1!r}  ~  {kw2!r}")


if __name__ == "__main__":
    main()
