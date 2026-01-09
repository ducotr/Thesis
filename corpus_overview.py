"""
corpus_overview.py

Generate a small set of thesis-ready CSVs that describe your OpenAlex-based corpus.

This version is intentionally minimal: it ONLY writes the CSVs/fields used in your LaTeX:
- data/filtering_outcomes.csv
- data/metadata_coverage.csv
- data/publications_yearly.csv
- data/keyword_frequency_distribution.csv
- data/keyword_frequency_top20.csv
- data/keyword_novelty_by_year.csv

Edit the GLOBAL VARIABLES below, then run:
    python3 corpus_overview.py
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date
import json
from pathlib import Path
from typing import Any

import requests

# ---------------------------------------------------------------------
# GLOBAL VARIABLES (edit these)
# ---------------------------------------------------------------------

# If None, we'll try to import CONCEPT from dashboard/config.py
CONCEPT: str | None = "Urban Digital Twin"

# OpenAlex "mailto" parameter (recommended by OpenAlex etiquette)
OPENALEX_MAILTO: str = "duco@trompert.net"

# Where to write CSVs
OUTPUT_DIR: Path = Path("corpus_overview_outputs")

# Cache raw OpenAlex records to avoid refetching (recommended for reproducibility)
CACHE_DIR: Path = Path("cache")
REFETCH_RAW_OPENALEX: bool = True  # set True to force a new API fetch

# Request settings
OPENALEX_TIMEOUT_S: int = 60
OPENALEX_PER_PAGE: int = 200
OPENALEX_BASE_URL: str = "https://api.openalex.org/works"


# ---------------------------------------------------------------------
# Helpers: text/date/keyword parsing
# ---------------------------------------------------------------------

def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = "".join(ch if ch.isalnum() else "_" for ch in s)
    s = "_".join([p for p in s.split("_") if p])
    return s or "concept"


def _keyword_token(kw: str) -> str | None:
    kw = (kw or "").strip()
    if not kw:
        return None
    return kw.lower().replace(" ", "_")


def reconstruct_abstract(abstract_inverted_index: Any) -> str:
    """Rebuild abstract text from OpenAlex inverted index dict, else empty string."""
    if not isinstance(abstract_inverted_index, dict):
        return ""
    position_map: dict[int, str] = {}
    for word, positions in abstract_inverted_index.items():
        if not isinstance(word, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                position_map[pos] = word
    if not position_map:
        return ""
    return " ".join(position_map[i] for i in sorted(position_map))


def _parse_date(d: Any) -> date | None:
    if not isinstance(d, str) or not d.strip():
        return None
    try:
        return date.fromisoformat(d.strip())
    except ValueError:
        return None


def _extract_keywords(record: dict) -> list[str]:
    """
    Return a list of normalized keyword tokens from an OpenAlex record.
    Handles keywords as:
      - list[str]
      - list[dict] with 'display_name'
    """
    kws = record.get("keywords")
    if not isinstance(kws, list):
        return []
    out: list[str] = []
    for item in kws:
        if isinstance(item, str):
            tok = _keyword_token(item)
            if tok:
                out.append(tok)
        elif isinstance(item, dict):
            name = item.get("display_name")
            if isinstance(name, str):
                tok = _keyword_token(name)
                if tok:
                    out.append(tok)
    return out


# ---------------------------------------------------------------------
# Inclusion/exclusion logic
# ---------------------------------------------------------------------

def exclusion_reasons_openalex(record: dict) -> list[str]:
    """
    Reasons are NOT mutually exclusive; a record can fail multiple criteria.

    Reasons returned (keys used for counts):
      - missing_title
      - missing_abstract
      - missing_date
      - invalid_date
      - no_keywords
    """
    reasons: list[str] = []

    title = record.get("title")
    if not isinstance(title, str) or not title.strip():
        reasons.append("missing_title")

    abstract = reconstruct_abstract(record.get("abstract_inverted_index"))
    if not abstract.strip():
        reasons.append("missing_abstract")

    pub_date_raw = record.get("publication_date")
    if pub_date_raw is None or (isinstance(pub_date_raw, str) and not pub_date_raw.strip()):
        reasons.append("missing_date")
    else:
        if _parse_date(pub_date_raw) is None:
            reasons.append("invalid_date")

    kws = _extract_keywords(record)
    if len(kws) == 0:
        reasons.append("no_keywords")

    return reasons


def is_included(record: dict) -> bool:
    return len(exclusion_reasons_openalex(record)) == 0


# ---------------------------------------------------------------------
# OpenAlex retrieval + caching
# ---------------------------------------------------------------------

def _raw_cache_path(concept: str) -> Path:
    slug = _slugify(concept)
    return CACHE_DIR / f"raw_openalex_{slug}.json"


def fetch_openalex_raw_records(concept: str) -> list[dict]:
    """Fetch raw OpenAlex works via title+abstract search (cursor pagination)."""
    params: dict[str, Any] = {
        "filter": f'title_and_abstract.search:"{concept}"',
        "per-page": OPENALEX_PER_PAGE,
        "cursor": "*",
    }
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    records: list[dict] = []
    while True:
        r = requests.get(OPENALEX_BASE_URL, params=params, timeout=OPENALEX_TIMEOUT_S)
        r.raise_for_status()
        payload = r.json()

        results = payload.get("results") or []
        if not isinstance(results, list) or len(results) == 0:
            break

        for rec in results:
            if isinstance(rec, dict):
                records.append(rec)

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        params["cursor"] = next_cursor

    return records


def load_or_fetch_raw_records(concept: str) -> list[dict]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _raw_cache_path(concept)

    if (not REFETCH_RAW_OPENALEX) and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass  # fall back to refetch if cache is corrupted

    records = fetch_openalex_raw_records(concept)
    cache_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


# ---------------------------------------------------------------------
# CSV exports (ONLY the ones used in your LaTeX)
# ---------------------------------------------------------------------

def export_filtering_outcomes(raw_records: list[dict], included_records: list[dict], out_dir: Path) -> None:
    """
    Writes data/filtering_outcomes.csv

    IMPORTANT: This file is row-indexed in your LaTeX. Keep this exact order:
      0 retrieved
      1 included_valid
      2 excluded_total
      3 missing_title
      4 missing_abstract
      5 missing_date
      6 invalid_date
      7 no_keywords
      8 date_problem_total (missing OR invalid)
    """
    import pandas as pd

    n_raw = len(raw_records)
    n_included = len(included_records)
    n_excluded = n_raw - n_included

    counts = Counter()
    date_problem_total = 0
    for rec in raw_records:
        rs = set(exclusion_reasons_openalex(rec))
        for reason in rs:
            counts[reason] += 1
        if ("missing_date" in rs) or ("invalid_date" in rs):
            date_problem_total += 1

    rows = [
        {"label": "retrieved", "count": n_raw},
        {"label": "included_valid", "count": n_included},
        {"label": "excluded_total", "count": n_excluded},
        {"label": "missing_title", "count": int(counts["missing_title"])},
        {"label": "missing_abstract", "count": int(counts["missing_abstract"])},
        {"label": "missing_date", "count": int(counts["missing_date"])},
        {"label": "invalid_date", "count": int(counts["invalid_date"])},
        {"label": "no_keywords", "count": int(counts["no_keywords"])},
        {"label": "date_problem_total", "count": int(date_problem_total)},
    ]

    pd.DataFrame(rows).to_csv(out_dir / "filtering_outcomes.csv", index=False)


def _is_present(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    if isinstance(val, (list, tuple, set, dict)):
        return len(val) > 0
    return True


def export_metadata_coverage(raw_records: list[dict], included_records: list[dict], out_dir: Path) -> None:
    """
    Writes data/metadata_coverage.csv

    Only the fields used in your LaTeX plot are included, in the same order as the axis:
      source_id, doi, title, abstract, publication_date, keywords, authors, total_cited_by

    Row order:
      raw_all block: rows 0..7
      included_valid block: rows 8..15
    """
    import pandas as pd

    # (output_field_name, openalex_field_key)
    fields: list[tuple[str, str]] = [
        ("source_id", "id"),
        ("doi", "doi"),
        ("title", "title"),
        ("abstract", "abstract_inverted_index"),
        ("publication_date", "publication_date"),
        ("keywords", "keywords"),
        ("authors", "authorships"),
        ("total_cited_by", "cited_by_count"),
    ]

    groups: list[tuple[str, list[dict]]] = [
        ("raw_all", raw_records),
        ("included_valid", included_records),
    ]

    rows: list[dict[str, Any]] = []
    for group_name, recs in groups:
        total = len(recs)
        for out_field, key in fields:
            present = 0
            for r in recs:
                if out_field == "abstract":
                    abstract = reconstruct_abstract(r.get("abstract_inverted_index"))
                    present += 1 if abstract.strip() else 0
                elif out_field == "keywords":
                    present += 1 if len(_extract_keywords(r)) > 0 else 0
                else:
                    present += 1 if _is_present(r.get(key)) else 0

            pct = (100.0 * present / total) if total else float("nan")
            rows.append(
                {
                    "group": group_name,
                    "field": out_field,
                    "count_present": int(present),
                    "total": int(total),
                    "pct_present": float(pct),
                }
            )

    pd.DataFrame(rows).to_csv(out_dir / "metadata_coverage.csv", index=False)


def export_publications_yearly(included_records: list[dict], out_dir: Path) -> None:
    """Writes data/publications_yearly.csv with columns: year, publications"""
    import pandas as pd

    by_year = Counter()
    for rec in included_records:
        d = _parse_date(rec.get("publication_date"))
        if d:
            by_year[d.year] += 1

    rows = [{"year": y, "publications": int(by_year[y])} for y in sorted(by_year)]
    pd.DataFrame(rows).to_csv(out_dir / "publications_yearly.csv", index=False)




def export_corpus_summary(raw_records: list[dict], included_records: list[dict], out_dir: Path) -> None:
    """
    Write a single-row CSV with a few headline corpus stats for thesis reporting.
    Columns:
      - candidate_records
      - included_records
      - first_publication_date
      - last_publication_date
      - unique_keywords
    """
    n_candidate = len(raw_records)
    n_included = len(included_records)

    dates = [_parse_date(r.get("publication_date")) for r in included_records]
    dates = [d for d in dates if d is not None]
    first_date = min(dates).isoformat() if dates else ""
    last_date = max(dates).isoformat() if dates else ""

    vocab: set[str] = set()
    for r in included_records:
        vocab.update(_extract_keywords(r))

    rows = [
        {
            "candidate_records": n_candidate,
            "included_records": n_included,
            "first_publication_date": first_date,
            "last_publication_date": last_date,
            "unique_keywords": len(vocab),
        }
    ]
    import pandas as pd

    pd.DataFrame(rows).to_csv(out_dir / "corpus_summary.csv", index=False)


def export_keyword_frequency_distribution(included_records: list[dict], out_dir: Path) -> None:
    """
    Writes data/keyword_frequency_distribution.csv with columns used in your LaTeX:
      - frequency
      - n_keywords
      - cum_share_keywords_pct
    (Document frequency = #publications a keyword appears in.)
    """
    import pandas as pd

    df_counts: Counter[str] = Counter()
    for rec in included_records:
        for kw in set(_extract_keywords(rec)):
            df_counts[kw] += 1

    freq_hist: Counter[int] = Counter(df_counts.values())
    total_unique = len(df_counts)

    rows = []
    cum = 0
    for freq in sorted(freq_hist):
        n_kw = int(freq_hist[freq])
        cum += n_kw
        cum_share = (100.0 * cum / total_unique) if total_unique else float("nan")
        rows.append(
            {
                "frequency": int(freq),
                "n_keywords": n_kw,
                "cum_share_keywords_pct": float(cum_share),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "keyword_frequency_distribution.csv", index=False)




def export_keyword_frequency_top20(included_records: list[dict], out_dir: Path, top_n: int = 20) -> None:
    """
    Writes data/keyword_frequency_top20.csv with the top-N keywords by document frequency.

    Columns:
      - rank (1..N)
      - keyword
      - frequency  (document frequency = number of publications the keyword appears in)
    """
    import pandas as pd

    df_counts: Counter[str] = Counter()
    for rec in included_records:
        for kw in set(_extract_keywords(rec)):
            df_counts[kw] += 1

    top = sorted(df_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]
    rows = [
        {"rank": i + 1, "keyword": kw, "frequency": freq}
        for i, (kw, freq) in enumerate(top)
    ]
    pd.DataFrame(rows).to_csv(out_dir / "keyword_frequency_top20.csv", index=False)

def export_keyword_novelty_by_year(included_records: list[dict], out_dir: Path) -> None:
    """
    Writes data/keyword_novelty_by_year.csv with columns used in your LaTeX:
      - year
      - unique_keywords
      - new_keywords
      - cumulative_unique_keywords
    """
    import pandas as pd

    year_to_keywords: dict[int, set[str]] = defaultdict(set)

    for rec in included_records:
        d = _parse_date(rec.get("publication_date"))
        if not d:
            continue
        year_to_keywords[d.year].update(_extract_keywords(rec))

    years = sorted(year_to_keywords)
    seen: set[str] = set()
    rows = []
    for y in years:
        kws = year_to_keywords.get(y, set())
        new_kws = kws - seen
        seen |= kws
        rows.append(
            {
                "year": int(y),
                "unique_keywords": int(len(kws)),
                "new_keywords": int(len(new_kws)),
                "cumulative_unique_keywords": int(len(seen)),
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "keyword_novelty_by_year.csv", index=False)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    global CONCEPT
    if CONCEPT is None:
        try:
            from dashboard.config import CONCEPT as DASH_CONCEPT  # type: ignore
            CONCEPT = str(DASH_CONCEPT)
        except Exception:
            raise RuntimeError(
                "CONCEPT is None and could not import dashboard.config.CONCEPT. "
                "Set CONCEPT at the top of corpus_overview.py."
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_records = load_or_fetch_raw_records(CONCEPT)
    included_records = [r for r in raw_records if is_included(r)]

    export_filtering_outcomes(raw_records, included_records, OUTPUT_DIR)
    export_corpus_summary(raw_records, included_records, OUTPUT_DIR)
    export_metadata_coverage(raw_records, included_records, OUTPUT_DIR)
    export_publications_yearly(included_records, OUTPUT_DIR)
    export_keyword_frequency_distribution(included_records, OUTPUT_DIR)
    export_keyword_frequency_top20(included_records, OUTPUT_DIR)
    export_keyword_novelty_by_year(included_records, OUTPUT_DIR)

    print(f"[corpus_overview] Wrote CSVs to: {OUTPUT_DIR.resolve()}")
    print(f"[corpus_overview] Retrieved: {len(raw_records)} | Included: {len(included_records)} | Excluded: {len(raw_records) - len(included_records)}")


if __name__ == "__main__":
    main()
