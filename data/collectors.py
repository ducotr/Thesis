from datetime import date

import requests

from data.publication import Publication


def reconstruct_abstract(abstract_inverted_index: dict[str, list[int]]) -> str:
    """
    Rebuild abstract text from inverted index dictionary, else empty string.
    """
    if not isinstance(abstract_inverted_index, dict):
        return ""
    position_map = {
        pos: word
        for word, positions in abstract_inverted_index.items()
        for pos in positions
    }
    if not position_map:
        return ""
    return " ".join(position_map[i] for i in sorted(position_map))


def is_valid_record(record: dict) -> bool:
    """
    Check if OpenAlex record contains necessary fields:
    - title
    - abstract (after reconstruction)
    - publication_date (valid YYYY-MM-DD)
    - keywords list
    """
    # Title
    title = record.get("title")
    if not title or not title.strip():
        return False

    # Abstract
    abstract_raw = record.get("abstract_inverted_index")
    abstract = reconstruct_abstract(abstract_raw)
    if not abstract.strip():
        return False

    # Publication date
    pub_date = record.get("publication_date")
    if not pub_date:
        return False
    try:
        y, m, d = [int(x) for x in pub_date.split("-")]
        _ = date(y, m, d)
    except Exception:
        return False

    # Keywords
    keywords = record.get("keywords")
    if not isinstance(keywords, list) or len(keywords) == 0:
        return False

    return True


def collect_openalex_data(concept: str) -> list[Publication]:
    """
    Collect OpenAlex data and return list of valid Publications.
    """
    BASE_URL = "https://api.openalex.org/works"
    parameters = {
        "filter": f'title_and_abstract.search:"{concept}"',
        "per-page": 200,
        "cursor": "*",
        "mailto": "duco@trompert.net",
    }

    publications: list[Publication] = []

    while True:
        r = requests.get(url=BASE_URL, params=parameters, timeout=60)
        r.raise_for_status()
        payload = r.json()

        results = payload.get("results", [])
        if not results:
            break

        for result in results:
            if is_valid_record(result):
                y, m, d = [int(x) for x in result["publication_date"].split("-")]
                authorships = result.get("authorships", []) or []
                authors = []
                for a in authorships:
                    author = (a.get("author") or {}).get("display_name")
                    if author:
                        authors.append(author.lower())

                publications.append(
                    Publication(
                        source_id=result["id"],
                        doi=result.get("doi", ""),
                        title=result["title"].lower(),
                        abstract=reconstruct_abstract(
                            result["abstract_inverted_index"]
                        ).lower(),
                        publication_date=date(y, m, d),
                        keywords=[
                            kw["display_name"].lower() for kw in result["keywords"]
                        ],
                        authors=authors,
                        total_cited_by=int(result.get("cited_by_count", 0)),
                        source="OpenAlex",
                    )
                )

        cursor = payload.get("meta", {}).get("next_cursor")
        if not cursor:
            break
        parameters["cursor"] = cursor

    return publications

