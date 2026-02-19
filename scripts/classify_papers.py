import json
import os
from pathlib import Path

from config import (
    CANDIDATES_PATH,
    CLASSIFIED_PATH,
    DIRECTIONS,
    NEWLY_REJECTED_PATH,
    TAGS,
)


def load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def heuristic_classify(paper: dict):
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    relevant = any(k in text for k in ["emotion", "affect", "sentiment", "empathy"])
    if not relevant:
        return {"is_relevant": False}

    if "speech" in text:
        direction = "Speech Emotion Recognition"
    elif "micro" in text:
        direction = "Micro-expression"
    elif "llm" in text or "language model" in text:
        direction = "Emotion-aware LLM"
    elif "multimodal" in text:
        direction = "Multimodal Emotion Analysis"
    else:
        direction = "Affect Recognition"

    tags = [t for t in TAGS if t in text][:5]
    return {
        "is_relevant": True,
        "direction": direction if direction in DIRECTIONS else "Affect Recognition",
        "tags": tags or ["benchmark"],
        "suggested_new_tags": [],
    }


def main():
    candidates = load_json(CANDIDATES_PATH, [])
    classified = []
    rejected = []

    # Placeholder hook for future Claude integration.
    _has_claude = bool(os.getenv("ANTHROPIC_API_KEY"))

    for paper in candidates:
        result = heuristic_classify(paper)
        if result["is_relevant"]:
            paper["direction"] = result["direction"]
            paper["tags"] = result["tags"]
            classified.append(paper)
        else:
            rejected.append(paper["paperId"])

    save_json(CLASSIFIED_PATH, classified)
    save_json(NEWLY_REJECTED_PATH, rejected)
    print(f"Classified={len(classified)} Rejected={len(rejected)}")


if __name__ == "__main__":
    main()
