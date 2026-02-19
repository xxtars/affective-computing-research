import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

from config import CANDIDATES_PATH, EMOTION_KEYWORDS, PIPELINE_STATE_PATH, RESEARCHERS_ENRICHED_PATH, S2_RPS

S2_BASE = "https://api.semanticscholar.org/graph/v1"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def http_get_json(url: str, headers: dict | None = None):
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def s2_headers():
    key = os.getenv("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def is_candidate(paper: dict) -> bool:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    return any(k in text for k in EMOTION_KEYWORDS)


def fetch_author_papers(author_id: str):
    query = urllib.parse.urlencode(
        {
            "fields": "papers.paperId,papers.externalIds,papers.title,papers.abstract,papers.year,papers.venue,papers.url,papers.authors",
            "limit": 100,
        }
    )
    data = http_get_json(f"{S2_BASE}/author/{author_id}/papers?{query}", headers=s2_headers())
    return data.get("data", [])


def main():
    researchers = load_json(RESEARCHERS_ENRICHED_PATH, [])
    state = load_json(PIPELINE_STATE_PATH, {"known_ids": [], "rejected_ids": [], "last_updated": None})
    seen = set(state.get("known_ids", []) + state.get("rejected_ids", []))

    candidates = []
    for researcher in researchers:
        author_id = researcher.get("semantic_scholar_id")
        if not author_id:
            continue
        try:
            papers = fetch_author_papers(author_id)
        except Exception as exc:
            print(f"skip author={author_id}: {exc}")
            continue

        for p in papers:
            pid = p.get("paperId")
            if not pid or pid in seen:
                continue
            normalized = {
                "paperId": pid,
                "doi": (p.get("externalIds") or {}).get("DOI"),
                "title": p.get("title"),
                "abstract": p.get("abstract"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "url": p.get("url"),
                "authors": [a.get("name") for a in p.get("authors", []) if a.get("name")],
                "researcher": researcher.get("name"),
            }
            if is_candidate(normalized):
                candidates.append(normalized)
        time.sleep(1 / S2_RPS)

    save_json(CANDIDATES_PATH, candidates)
    print(f"Fetched {len(candidates)} candidates -> {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()
