import json
import time
from pathlib import Path

import requests

from data.publication_corpus import PublicationCorpus

MODELS = [
    "llama3",
    "mistral",
    "gemma2",
    "phi3",
    "qwen2.5",
]

CONCEPT = "Urban Digital Twin"
OUT_PATH = Path("classification/keyword_labels.json")

# How many keywords to send per request per model
BATCH_SIZE = 1

# How often to save intermediate results (in "batches processed")
SAVE_EVERY = 1

# ---------- PROMPTS ----------

# Single-keyword template (kept for reference / debugging, but not used in batching)
PROMPT_TEMPLATE = """
You are an expert in Urban Digital Twin (UDT), smart cities, and urban analytics.

Your task is to classify a given keyword into exactly one of three categories:

- "concept": a domain idea, phenomenon, or theoretical notion
  (e.g. sustainability, urban resilience, energy transition, smart city,
  urban regeneration, interoperability, citizen engagement).

- "method": a technique, tool, algorithm, model, data source, technology,
  or methodological approach
  (e.g. remote sensing, agent-based model, machine learning, photogrammetry,
  GIS, 3D city model, BIM, sensor network, simulation platform).

- "other": a discipline, background field, or term that is too broad or
  ambiguous to be meaningfully treated as a concept or method in the Urban
  Digital Twin context
  (e.g. mathematics, economics, sociology, geology, engineering).

Decision rules:
- Use "concept" for abstract ideas, properties, or domain constructs.
- Use "method" for concrete procedures, tools, algorithms, models,
  data types, infrastructures, or technical implementations.
- Use "other" ONLY if the keyword cannot reasonably be classified as
  "concept" or "method" within Urban Digital Twin and related fields.
- If a keyword could be seen as both a concept and a method, choose the
  interpretation that is more common in Urban Digital Twin research.
- If you are still unsure, prefer "concept" over "method" over "other".

Here are some examples:

Keyword: "Urban resilience"
Label: concept

Keyword: "Agent-based model"
Label: method

Keyword: "Remote sensing"
Label: method

Keyword: "Smart city"
Label: concept

Keyword: "Economics"
Label: other

Now classify the following keyword. Answer in strict JSON.
Return ONLY a single JSON object, with no extra text:

{"keyword": "<KEYWORD>", "label": "<concept|method|other>"}
"""

# New: batch template that takes a JSON array of keywords
BATCH_PROMPT_TEMPLATE = """
You are an expert in Urban Digital Twin (UDT), smart cities, and urban analytics.

Your task is to classify EACH keyword in the given JSON array into exactly one of three categories:

- "concept": a domain idea, phenomenon, or theoretical notion.
- "method": a technique, tool, algorithm, model, data source, technology,
  or methodological approach.
- "other": a discipline, background field, or term that is too broad or
  ambiguous to be meaningfully treated as a concept or method in the Urban
  Digital Twin context.

Decision rules:
- Use "concept" for abstract ideas, properties, or domain constructs.
- Use "method" for concrete procedures, tools, algorithms, models,
  data types, infrastructures, or technical implementations.
- Use "other" ONLY if the keyword cannot reasonably be classified as
  "concept" or "method" within Urban Digital Twin and related fields.
- If a keyword could be seen as both a concept and a method, choose the
  interpretation that is more common in Urban Digital Twin research.
- If you are still unsure, prefer "concept" over "method" over "other".

Return ONLY a single JSON object that maps each keyword string (exactly as
given) to its label. For example:

{{
  "Smart city": "concept",
  "Remote sensing": "method",
  "Economics": "other"
}}

Keywords (JSON array):
{keywords_json}
"""


def build_prompt(keyword: str) -> str:
    return PROMPT_TEMPLATE.replace("<KEYWORD>", keyword)


def build_batch_prompt(keywords: list[str]) -> str:
    """
    Build the batch prompt with a JSON array of keywords.
    """
    keywords_json = json.dumps(keywords, ensure_ascii=False)
    return BATCH_PROMPT_TEMPLATE.format(keywords_json=keywords_json)


def call_llama(keyword: str, model: str) -> str:
    """
    Single-keyword call (kept for debugging; batching uses call_llama_batch).
    """
    payload = {
        "model": model,
        "prompt": build_prompt(keyword),
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 32,
        },
    }
    r = requests.post(
        url="http://localhost:11434/api/generate",
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    content = data["response"].strip()

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        content = content[start : end + 1]

    parsed = json.loads(content)
    label = parsed.get("label")
    if label not in ("concept", "method", "other"):
        raise ValueError(f"Unexpected label for {keyword!r} from {model!r}: {label!r}")
    return label


def call_llama_batch(keywords: list[str], model: str) -> dict[str, str]:
    """
    Ask one model (via Ollama) to classify multiple keywords at once.

    Returns a mapping: {keyword: label}
    """
    payload = {
        "model": model,
        "prompt": build_batch_prompt(keywords),
        "stream": False,
        "options": {
            "temperature": 0.0,
            # Need more tokens than single-call, because we return labels for many keywords.
            "num_predict": 256,
        },
    }

    r = requests.post(
        url="http://localhost:11434/api/generate",
        json=payload,
        timeout=90,
    )
    r.raise_for_status()
    data = r.json()
    content = data["response"].strip()

    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1:
        content = content[start : end + 1]

    parsed = json.loads(content)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Batch output from {model!r} is not a JSON object: {parsed!r}"
        )

    # Basic sanity checks
    for kw in keywords:
        if kw not in parsed:
            raise ValueError(
                f"Keyword {kw!r} missing in batch output from {model!r}: keys={list(parsed.keys())}"
            )
        label = parsed[kw]
        if label not in ("concept", "method", "other"):
            raise ValueError(
                f"Invalid label {label!r} for keyword {kw!r} from {model!r}"
            )

    return parsed  # {keyword: label}


def get_unique_keywords(concept: str) -> list[str]:
    """
    Use PublicationCorpus to get all unique keywords for the concept.
    """
    corpus = PublicationCorpus(
        concept=concept,
        use_cache=False,
    )
    counts = corpus.keywords_occurrence()
    return sorted(counts.keys())


def chunks(seq, n):
    """
    Yield successive chunks of length n from seq.
    """
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def normalize_label(raw) -> str:
    # allow either a string or {"label": "..."} just in case
    if isinstance(raw, dict) and "label" in raw:
        raw = raw["label"]

    s = str(raw).strip().lower()

    # common harmless variations
    aliases = {
        "concept": "concept",
        "concepts": "concept",
        "method": "method",
        "methods": "method",
        "other": "other",
        "misc": "other",
        "unknown": "other",
    }

    if s not in aliases:
        raise ValueError(f"Unrecognized label: {raw!r}")

    return aliases[s]


def classify_keywords(concept: str, out_path: Path) -> None:
    """
    Classify all unique keywords with different models and save them to a JSON file.

    Output format:
    {
      "Smart city": {
        "llama3": "concept",
        "mistral": "concept",
        ...
      },
      ...
    }
    """
    keywords = get_unique_keywords(concept)
    print(f"Found {len(keywords)} unique keywords for concept '{concept}'.")

    # keyword -> {model_name -> label}
    labels: dict[str, dict[str, str]] = {}

    if out_path.exists():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            labels.update(existing)
        print(f"Loaded labels for {len(labels)} keywords from {out_path}.")

    calls_since_save = 0

    for model in MODELS:
        print(f"\n=== Running model {model!r} ===")

        remaining = [
            kw for kw in keywords if kw not in labels or model not in labels[kw]
        ]
        print(f"{len(remaining)} keywords remaining for model {model!r}.")

        # ---------- SPECIAL CASE: phi3 → NO BATCHING ----------
        if model == "phi3":
            for kw in remaining:
                try:
                    label = call_llama(kw, model)  # single-keyword call
                except Exception as e:
                    print(f"Single-call failed for {kw!r} with {model!r}: {e}")
                    continue

                kw_labels = labels.get(kw) or {}
                kw_labels[model] = label
                labels[kw] = kw_labels
                print(f"{kw!r} [{model}] -> {label} (single)")

                calls_since_save += 1
                if calls_since_save >= SAVE_EVERY:
                    out_path.write_text(
                        json.dumps(labels, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    calls_since_save = 0

            # Done with phi3, move on to next model
            continue
        # ---------- END SPECIAL CASE ----------

        # Existing batching logic for all other models
        for batch in chunks(remaining, BATCH_SIZE):
            unresolved = []

            try:
                batch_raw = call_llama_batch(batch, model)
            except Exception as e:
                print(f"Batch classification failed for {batch!r} with {model!r}: {e}")
                unresolved = batch[:]  # fallback to single-call for entire batch
            else:
                for kw in batch:
                    raw_label = batch_raw.get(kw)
                    if raw_label is None:
                        print(
                            f"Keyword {kw!r} missing in batch output from {model!r}; "
                            "will fall back to single-call."
                        )
                        unresolved.append(kw)
                        continue

                    try:
                        label = normalize_label(raw_label)
                    except Exception as e:
                        print(
                            f"Problem normalising label for {kw!r} from {model!r}: "
                            f"{raw_label!r} ({e}); will fall back to single-call."
                        )
                        unresolved.append(kw)
                        continue

                    kw_labels = labels.get(kw) or {}
                    kw_labels[model] = label
                    labels[kw] = kw_labels
                    print(f"{kw!r} [{model}] -> {label} (batch)")

            # Fallback for unresolved keywords in this batch
            for kw in unresolved:
                try:
                    label = call_llama(kw, model)
                except Exception as e:
                    print(f"Single-call fallback failed for {kw!r} with {model!r}: {e}")
                    continue

                kw_labels = labels.get(kw) or {}
                kw_labels[model] = label
                labels[kw] = kw_labels
                print(f"{kw!r} [{model}] -> {label} (single)")

            calls_since_save += 1
            if calls_since_save >= SAVE_EVERY:
                out_path.write_text(
                    json.dumps(labels, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                calls_since_save = 0

    # Final save
    out_path.write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDone. Wrote labels for {len(labels)} keywords to {out_path}.")


if __name__ == "__main__":
    classify_keywords(
        concept=CONCEPT,
        out_path=OUT_PATH,
    )
