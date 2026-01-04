from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from threading import Lock

import networkx as nx
from gensim.models import Word2Vec

from analysis.embeddings import Word2VecConfig, train_word2vec
from analysis.networks import (
    build_concept_method_bipartite,
    build_cooccurrence_network,
    build_semantic_similarity_network,
    build_temporal_concept_method_bipartite,
    build_temporal_network,
    build_temporal_semantic_similarity_network,
)
from data.collectors import collect_openalex_data
from data.publication import Publication


class PublicationCorpus:
    """
    A collection of Publication objects associated with a specific concept.

    If use_cache is True and the cache file exists, data is loaded from that file.
    Otherwise, data is fetched live from OpenAlex and written to the cache.
    """

    def __init__(self, concept: str, use_cache: bool = True) -> None:
        self._concept: str = concept

        self._safe_concept = concept.lower().strip().replace(" ", "_")
        self._cache_path: Path = Path(f"cache/{self._safe_concept}.json")

        # Word2Vec model cache (memory + disk)
        self._w2v_models: dict[tuple[int | None, Word2VecConfig], Word2Vec] = {}
        self._w2v_lock: Lock = Lock()
        self._w2v_cache_dir: Path = Path(f"cache/w2v/{self._safe_concept}")

        if use_cache and self._cache_path.exists():
            # Load JSON cache
            raw = self._cache_path.read_text(encoding="utf-8")
            records = json.loads(raw)
            self._data: list[Publication] = [Publication.from_dict(r) for r in records]
        else:
            # Fetch live and write JSON cache
            publications = collect_openalex_data(concept)
            self._data = publications

            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            records = [p.to_dict() for p in publications]
            self._cache_path.write_text(
                json.dumps(records, indent=2),
                encoding="utf-8",
            )

        # --- metadata construction ---
        pubs_per_year = self.years()
        min_year = min(pubs_per_year) if pubs_per_year else None
        max_year = max(pubs_per_year) if pubs_per_year else None

        kws_occurrence = self.keywords_occurrence()
        top_keywords = sorted(
            kws_occurrence.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:20]

        self._meta_data = {
            "concept": concept,
            "description": (
                f"OpenAlex works where '{concept}' appears in title or abstract"
            ),
            "data_source": {
                "name": "OpenAlex",
                "endpoint": "https://api.openalex.org/works",
            },
            "query": {
                "filter": f'title_and_abstract.search:"{concept}"',
                "per-page": 200,
                "cursor_start": "*",
                "mailto": "duco@trompert.net",
            },
            "retrieval": {
                "date": date.today().isoformat(),  # metadata generation date
                "use_cache": use_cache,
                "cache_path": str(self._cache_path),
            },
            "preprocessing": {
                "record_filters": [
                    "must have non-empty title",
                    "must have non-empty abstract",
                    "must have valid publication date",
                    "must have at least one keyword",
                ],
                "keyword_normalisation": "lowercased, spaces replaced with underscores",
            },
            "corpus_stats": {
                "n_publications": len(self._data),
                "year_min": min_year,
                "year_max": max_year,
                "pubs_per_year": pubs_per_year,
                "top_keywords": top_keywords,
            },
            "version": {
                "schema_version": 1,
            },
        }

        meta_path = self._cache_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(self._meta_data, indent=2),
            encoding="utf-8",
        )

    # Dunder methods
    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, index: int) -> Publication:
        return self._data[index]

    @property
    def data(self) -> list[Publication]:
        """Return a copy of the data list."""
        return list(self._data)

    @property
    def concept(self) -> str:
        """Return the concept as a string."""
        return self._concept

    @property
    def meta(self) -> dict:
        return dict(self._meta_data)

    # Keyword methods
    def keywords_occurrence(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pub in self._data:
            for kw in pub.keywords:
                counts[kw] = counts.get(kw, 0) + 1
        return counts

    # Year method
    def years(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for pub in self._data:
            pub_year = pub.publication_date.year
            counts[pub_year] = counts.get(pub_year, 0) + 1
        return dict(sorted(counts.items()))

    # Summary
    def summary(self) -> str:
        years = self.years().keys()
        year_range = (min(years), max(years)) if years else (None, None)
        return (
            f"PublicationCorpus(concept={self._concept!r}, "
            f"n_pubs={len(self)}, "
            f"year_range={year_range}, "
            f"n_unique_keywords={len(self.keywords_occurrence())})"
        )

    # Network graph methods
    def cooccurrence_network(self, min_value: int = 0) -> nx.Graph:
        return build_cooccurrence_network(publications=self._data, min_value=min_value)

    def temporal_network(self, min_value: int = 0) -> dict[int, nx.Graph]:
        return build_temporal_network(publications=self._data, min_value=min_value)

    def concept_method_network(self, min_value: int = 0) -> nx.Graph:
        return build_concept_method_bipartite(
            publications=self._data,
            min_value=min_value,
        )

    def temporal_concept_method_network(self, min_value: int = 0) -> nx.Graph:
        return build_temporal_concept_method_bipartite(
            publications=self._data,
            min_value=min_value,
        )

    def _w2v_model_path(self, w2v_config: Word2VecConfig, year: int | None) -> Path:
        """
        Disk location for a cached Word2Vec model.
        If you change tokenization / sentence construction, bump tokv.
        """
        tokv = 1
        scope = "overall" if year is None else f"year{year}"
        cfg = (
            f"vs{w2v_config.vector_size}_"
            f"win{w2v_config.window}_"
            f"mc{w2v_config.min_count}_"
            f"sg{w2v_config.sg}_"
            f"ep{w2v_config.epochs}_"
            f"seed{w2v_config.seed}"
        )
        return self._w2v_cache_dir / f"{scope}__tokv{tokv}__{cfg}.model"

    def get_or_train_word2vec_model(
        self,
        w2v_config: Word2VecConfig,
        *,
        year: int | None = None,
        persist: bool = False,
    ) -> Word2Vec:
        """
        Return a Word2Vec model trained on:
          - the full corpus (year=None), or
          - only publications from a given year.

        Cached in memory; optionally persisted to disk.
        """
        key = (year, w2v_config)
        cached = self._w2v_models.get(key)
        if cached is not None:
            return cached

        with self._w2v_lock:
            # check again after acquiring lock
            cached = self._w2v_models.get(key)
            if cached is not None:
                return cached

            model_path = self._w2v_model_path(w2v_config, year)

            # Try disk cache first
            if persist and model_path.exists():
                model = Word2Vec.load(str(model_path))
                self._w2v_models[key] = model
                return model

            # Train
            if year is None:
                pubs = self._data
            else:
                pubs = [p for p in self._data if p.publication_date.year == year]

            model = train_word2vec(publications=pubs, config=w2v_config)

            # Save to disk
            if persist:
                self._w2v_cache_dir.mkdir(parents=True, exist_ok=True)
                model.save(str(model_path))

            self._w2v_models[key] = model
            return model

    def semantic_similarity_network(
        self,
        w2v_config: Word2VecConfig,
        min_keyword_freq: int = 5,
        similarity_threshold: float = 0.5,
        top_k: int | None = None,
    ) -> nx.Graph:
        model = self.get_or_train_word2vec_model(w2v_config, year=None)
        return build_semantic_similarity_network(
            publications=self._data,
            w2v_config=w2v_config,
            min_keyword_freq=min_keyword_freq,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            model=model,
        )

    def temporal_semantic_similarity_network(
        self,
        w2v_config: Word2VecConfig,
        min_keyword_freq: int = 5,
        similarity_threshold: float = 0.5,
        top_k: int | None = None,
    ) -> dict[int, nx.Graph]:
        # Train/load one model per year, once per config
        years = list(self.years().keys())
        models_by_year = {
            y: self.get_or_train_word2vec_model(w2v_config, year=y) for y in years
        }
        return build_temporal_semantic_similarity_network(
            publications=self._data,
            w2v_config=w2v_config,
            min_keyword_freq=min_keyword_freq,
            similarity_threshold=similarity_threshold,
            top_k=top_k,
            models_by_year=models_by_year,
        )
