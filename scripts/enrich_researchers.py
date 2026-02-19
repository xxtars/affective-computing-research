import json
import os
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from config import RESEARCHERS_ENRICHED_PATH, RESEARCHERS_PATH

S2_BASE = "https://api.semanticscholar.org/graph/v1"


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def http_get(url: str, headers: dict | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.getcode(), resp.read().decode("utf-8", errors="ignore")


def parse_scholar_name(google_scholar_url: str) -> str | None:
    try:
        _, body = http_get(google_scholar_url)
        m = re.search(r"<title>(.*?) - Google Scholar</title>", body, flags=re.I | re.S)
        if m:
            return m.group(1).strip()
    except Exception:
        return None
    return None


def s2_headers():
    key = os.getenv("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def lookup_semantic_author(name: str):
    query = urllib.parse.urlencode({"query": name, "limit": 1, "fields": "authorId,name,affiliations,homepage"})
    _, body = http_get(f"{S2_BASE}/author/search?{query}", headers=s2_headers())
    data = json.loads(body).get("data", [])
    if not data:
        return None
    top = data[0]
    return {
        "semantic_scholar_id": top.get("authorId"),
        "resolved_name": top.get("name") or name,
        "institution": (top.get("affiliations") or [None])[0],
        "homepage": top.get("homepage"),
    }


def main():
    researchers = load_json(RESEARCHERS_PATH, [])
    old = {r.get("google_scholar"): r for r in load_json(RESEARCHERS_ENRICHED_PATH, [])}
    enriched = []

    for r in researchers:
        scholar = r.get("google_scholar")
        base = old.get(scholar, {})
        input_name = r.get("name", "").strip()
        scholar_name = parse_scholar_name(scholar) if scholar else None
        query_name = scholar_name or input_name

        row = {"name": input_name or query_name, "google_scholar": scholar}

        try:
            s2 = lookup_semantic_author(query_name) if query_name else None
        except Exception:
            s2 = None

        if s2:
            row.update(
                {
                    "name": s2.get("resolved_name") or row["name"],
                    "semantic_scholar_id": s2.get("semantic_scholar_id"),
                    "institution": s2.get("institution") or base.get("institution"),
                    "homepage": s2.get("homepage") or base.get("homepage"),
                    "country": base.get("country"),
                    "directions": base.get("directions", []),
                    "needs_review": False,
                }
            )
        else:
            row.update(
                {
                    "semantic_scholar_id": base.get("semantic_scholar_id"),
                    "institution": base.get("institution"),
                    "homepage": base.get("homepage"),
                    "country": base.get("country"),
                    "directions": base.get("directions", []),
                    "needs_review": True,
                }
            )

        row["last_verified"] = datetime.now(timezone.utc).isoformat()
        enriched.append(row)

    save_json(RESEARCHERS_ENRICHED_PATH, enriched)
    print(f"Enriched {len(enriched)} researchers -> {RESEARCHERS_ENRICHED_PATH}")


if __name__ == "__main__":
    main()
