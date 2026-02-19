import json
from datetime import datetime, timezone
from pathlib import Path

from config import (
    CLASSIFIED_PATH,
    NEWLY_REJECTED_PATH,
    PAPERS_PATH,
    PIPELINE_STATE_PATH,
    RESEARCHERS_ENRICHED_PATH,
)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    classified = load_json(CLASSIFIED_PATH, [])
    rejected = load_json(NEWLY_REJECTED_PATH, [])
    papers = load_json(PAPERS_PATH, [])
    state = load_json(PIPELINE_STATE_PATH, {"known_ids": [], "rejected_ids": [], "last_updated": None})
    researchers = {
        r.get("name"): r for r in load_json(RESEARCHERS_ENRICHED_PATH, [])
    }

    existing = {p.get("s2_id") for p in papers if p.get("s2_id")}

    for p in classified:
        if p["paperId"] in existing:
            continue
        researcher = researchers.get(p.get("researcher"), {})
        papers.append(
            {
                "s2_id": p.get("paperId"),
                "doi": p.get("doi"),
                "title": p.get("title"),
                "authors": p.get("authors", []),
                "researcher": p.get("researcher"),
                "institution": researcher.get("institution"),
                "country": researcher.get("country"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "direction": p.get("direction"),
                "tags": p.get("tags", []),
                "abstract": p.get("abstract"),
                "url": p.get("url"),
                "source": "auto",
            }
        )

    state["known_ids"] = sorted(set(state.get("known_ids", []) + [p.get("paperId") for p in classified]))
    state["rejected_ids"] = sorted(set(state.get("rejected_ids", []) + rejected))
    state["last_updated"] = datetime.now(timezone.utc).isoformat()

    save_json(PAPERS_PATH, papers)
    save_json(PIPELINE_STATE_PATH, state)
    print(f"Merged {len(classified)} papers, rejected+{len(rejected)}")


if __name__ == "__main__":
    main()
