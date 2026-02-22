#!/usr/bin/env python3
"""
Build problem/method taxonomy from paper-level directions:
directions -> embeddings (Qwen) -> BERTopic -> Qwen L2 labels -> Qwen L1 merge.

Outputs are written to data-repo by default:
  data-repo/data/taxonomy/{problem|method}/...
Raw API responses are preserved in:
  data-repo/data/taxonomy/api_logs/{axis}/...
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
from bertopic import BERTopic
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_EMBED_MODEL = "text-embedding-v4"
DEFAULT_CHAT_MODEL = "qwen3.5-plus"
DEFAULT_EMBEDDING_MAX_BATCH = 10


@dataclass
class DirectionRecord:
    axis: str
    text: str
    paper_id: str
    paper_title: str
    publication_year: int | None
    researcher_id: str
    researcher_name: str
    context: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def dump_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    return text.strip("_")[:120] or "item"


def parse_json_from_text(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            raise
        return json.loads(match.group(0))


def parse_bool_env(value: str | None) -> bool | None:
    if value is None:
        return None
    v = value.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return None


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return load_json(path)


def make_record_key(record: DirectionRecord) -> str:
    # Stable id for resuming embeddings across runs.
    raw = f"{record.axis}|{record.paper_id}|{record.researcher_id}|{record.text}"
    return stable_hash(raw)


def sort_records_stable(records: List[DirectionRecord]) -> List[DirectionRecord]:
    return sorted(
        records,
        key=lambda r: (
            str(r.paper_id),
            str(r.publication_year) if r.publication_year is not None else "",
            str(r.researcher_id),
            str(r.text),
        ),
    )


def make_bertopic_fingerprint(
    records: List[DirectionRecord],
    emb_model: str,
    min_topic_size: int,
    random_seed: int,
) -> str:
    payload = {
        "emb_model": emb_model,
        "min_topic_size": int(min_topic_size),
        "random_seed": int(random_seed),
        "records": [make_record_key(r) for r in records],
        "contexts": [stable_hash(r.context) for r in records],
    }
    return stable_hash(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def make_topic_fingerprint(axis: str, keywords: List[str], examples: List[str]) -> str:
    normalized = {
        "axis": axis,
        "keywords": sorted([str(x).strip().lower() for x in keywords if str(x).strip()]),
        "examples": sorted([str(x).strip().lower() for x in examples if str(x).strip()]),
    }
    return stable_hash(json.dumps(normalized, ensure_ascii=False, sort_keys=True))


def make_l1_cluster_fingerprint(axis: str, members: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "l2_name": str(m.get("l2_name", "")).strip().lower(),
            "definition": str(m.get("definition", "")).strip().lower(),
        }
        for m in members
    ]
    payload = sorted(payload, key=lambda x: x["l2_name"])
    return stable_hash(f"{axis}|{json.dumps(payload, ensure_ascii=False, sort_keys=True)}")


def log_api(
    log_dir: Path,
    api_type: str,
    name: str,
    request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
) -> None:
    ensure_dir(log_dir)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = log_dir / f"{stamp}_{api_type}_{safe_slug(name)}.json"
    dump_json(
        path,
        {
            "timestamp": utc_now(),
            "api_type": api_type,
            "name": name,
            "request": request_payload,
            "response": response_payload,
        },
    )


def build_embedding_text(direction: str, title: str, axis: str) -> str:
    # Direction-only embedding for cleaner cross-paper taxonomy clustering.
    _ = title
    _ = axis
    return direction


def collect_records(researchers_root: Path, axis: str) -> List[DirectionRecord]:
    index_path = researchers_root / "researchers.index.json"
    index_data = load_json(index_path)
    records: List[DirectionRecord] = []
    for item in index_data.get("researchers", []):
        identity = item.get("identity", {}) or {}
        rid = str(identity.get("openalex_author_id", "")).strip()
        rname = str(identity.get("name", "")).strip()
        profile_path_raw = str(item.get("profile_path", "")).strip()
        if profile_path_raw:
            profile_path = researchers_root / "profiles" / Path(profile_path_raw).name
        elif rid:
            profile_path = researchers_root / "profiles" / f"{rid}.json"
        else:
            continue
        if not profile_path.exists():
            continue
        profile = load_json(profile_path)
        works = profile.get("works", []) or []
        for w in works:
            analysis = (w or {}).get("analysis", {}) or {}
            if not analysis.get("is_interesting", False):
                continue
            if axis == "problem":
                directions = analysis.get("problem_directions", []) or []
            elif axis == "method":
                directions = analysis.get("method_directions", []) or []
            else:
                raise ValueError(f"invalid axis: {axis}")

            for d in directions:
                direction = str(d or "").strip().lower()
                if not direction:
                    continue
                title = str((w or {}).get("title", "")).strip()
                pid = str((w or {}).get("id", "")).strip()
                year = (w or {}).get("publication_year", None)
                records.append(
                    DirectionRecord(
                        axis=axis,
                        text=direction,
                        paper_id=pid,
                        paper_title=title,
                        publication_year=year if isinstance(year, int) else None,
                        researcher_id=rid,
                        researcher_name=rname,
                        context=build_embedding_text(direction, title, axis),
                    )
                )
    return records


def call_embedding_api(
    api_key: str,
    base_url: str,
    model: str,
    texts: List[str],
    log_dir: Path,
    batch_name: str,
) -> List[List[float]]:
    url = f"{base_url.rstrip('/')}/embeddings"
    req = {"model": model, "input": texts}
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=req, timeout=120)
    body_text = resp.text
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw_text": body_text}
    log_api(log_dir, "embedding", batch_name, req, payload)
    resp.raise_for_status()
    data = payload.get("data", []) or []
    if len(data) != len(texts):
        raise RuntimeError(f"embedding count mismatch: expected {len(texts)} got {len(data)}")
    return [d.get("embedding", []) for d in data]


def embed_records(
    records: List[DirectionRecord],
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    embedding_concurrency: int,
    log_dir: Path,
    cache_path: Path,
) -> np.ndarray:
    texts = [r.context for r in records]
    keys = [make_record_key(r) for r in records]
    context_hashes = [stable_hash(t) for t in texts]

    cache_payload = load_json_if_exists(
        cache_path,
        {"version": 1, "model": model, "items": {}},
    )
    cache_items = cache_payload.get("items", {}) if isinstance(cache_payload, dict) else {}
    if not isinstance(cache_items, dict):
        cache_items = {}
    cache_model = str(cache_payload.get("model", "")).strip() if isinstance(cache_payload, dict) else ""
    if cache_model and cache_model != model:
        print(f"[taxonomy] embedding cache model changed: {cache_model} -> {model}, cache will be ignored")
        cache_items = {}

    vectors: List[List[float] | None] = [None] * len(records)
    missing_indexes: List[int] = []
    for i, key in enumerate(keys):
        hit = cache_items.get(key)
        if (
            isinstance(hit, dict)
            and hit.get("context_hash") == context_hashes[i]
            and isinstance(hit.get("vector"), list)
            and len(hit.get("vector")) > 0
        ):
            vectors[i] = hit["vector"]
        else:
            missing_indexes.append(i)

    print(
        f"[taxonomy] embedding cache: hits={len(records) - len(missing_indexes)} "
        f"misses={len(missing_indexes)} total={len(records)}"
    )
    if len(missing_indexes) == 0:
        return np.asarray(vectors, dtype=np.float32)

    embedding_max_batch = int(os.getenv("QWEN_EMBEDDING_MAX_BATCH", str(DEFAULT_EMBEDDING_MAX_BATCH)))
    effective_batch_size = max(1, min(batch_size, embedding_max_batch))
    if effective_batch_size != batch_size:
        print(
            f"[taxonomy] embedding batch size adjusted: requested={batch_size}, "
            f"effective={effective_batch_size}, max={embedding_max_batch}"
        )
    total_batches = (len(missing_indexes) + effective_batch_size - 1) // effective_batch_size
    effective_concurrency = max(1, min(embedding_concurrency, total_batches))
    print(
        f"[taxonomy] embedding workers={effective_concurrency} "
        f"(requested={embedding_concurrency}, batches={total_batches})"
    )

    batch_specs: List[Tuple[int, List[int], List[str], int]] = []
    for i in range(0, len(missing_indexes), effective_batch_size):
        batch_indexes = missing_indexes[i : i + effective_batch_size]
        batch = [texts[j] for j in batch_indexes]
        batch_idx = i // effective_batch_size + 1
        batch_specs.append((i, batch_indexes, batch, batch_idx))

    with ThreadPoolExecutor(max_workers=effective_concurrency) as ex:
        futures = {
            ex.submit(
                call_embedding_api,
                api_key=api_key,
                base_url=base_url,
                model=model,
                texts=batch,
                log_dir=log_dir,
                batch_name=f"batch_{i//effective_batch_size:04d}",
            ): (i, batch_indexes, batch_idx)
            for i, batch_indexes, batch, batch_idx in batch_specs
        }

        completed = 0
        for fut in as_completed(futures):
            i, batch_indexes, batch_idx = futures[fut]
            batch_vec = fut.result()
            completed += 1
            print(
                f"[taxonomy] embedding progress: {completed}/{total_batches} done "
                f"(batch {batch_idx}, missing items {i + 1}-{min(i + len(batch_indexes), len(missing_indexes))}/{len(missing_indexes)})"
            )
            for local_idx, vec in enumerate(batch_vec):
                record_idx = batch_indexes[local_idx]
                vectors[record_idx] = vec
                key = keys[record_idx]
                cache_items[key] = {
                    "context_hash": context_hashes[record_idx],
                    "vector": vec,
                    "updated_at": utc_now(),
                }
            dump_json(
                cache_path,
                {
                    "version": 1,
                    "model": model,
                    "updated_at": utc_now(),
                    "items": cache_items,
                },
            )
            time.sleep(0.05)
    if any(v is None for v in vectors):
        raise RuntimeError("embedding resume failed: some vectors are still missing")
    return np.asarray(vectors, dtype=np.float32)


def build_topic_model(
    docs: List[str],
    embeddings: np.ndarray,
    min_topic_size: int,
    random_seed: int,
) -> Tuple[BERTopic, List[int]]:
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=int(random_seed),
    )
    model = BERTopic(
        vectorizer_model=vectorizer,
        umap_model=umap_model,
        min_topic_size=min_topic_size,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(docs, embeddings=embeddings)
    return model, topics


def call_chat_json(
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    log_dir: Path,
    log_name: str,
    max_tokens: int = 320,
    enable_thinking: bool = False,
    max_retries: int = 2,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        strict_suffix = ""
        if attempt > 0:
            strict_suffix = (
                "\n\nIMPORTANT RETRY RULES:\n"
                "- Output must be a single valid JSON object.\n"
                "- No markdown fences, no comments, no trailing text.\n"
                "- Ensure all quotes/commas/brackets are valid JSON."
            )
        req = {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "enable_thinking": enable_thinking if attempt == 0 else False,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a precise taxonomy analyst. Return valid JSON only."},
                {"role": "user", "content": f"{prompt}{strict_suffix}"},
            ],
        }
        resp = requests.post(url, headers=headers, json=req, timeout=120)
        body_text = resp.text
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw_text": body_text}
        log_api(log_dir, "chat", f"{log_name}_try{attempt+1}", req, payload)
        resp.raise_for_status()
        content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
        try:
            return parse_json_from_text(content)
        except Exception as err:
            last_err = err
            if attempt < max_retries:
                print(f"[taxonomy] chat parse retry {attempt+1}/{max_retries} for {log_name}: {err}")
                time.sleep(0.4)
                continue
            break
    if last_err is None:
        raise RuntimeError("chat parse failed with unknown error")
    raise last_err


def build_l2_prompt(axis: str, topic_id: int, keywords: List[str], examples: List[str]) -> str:
    return (
        f"Axis: {axis}\n"
        f"Topic id: {topic_id}\n"
        f"Top keywords: {', '.join(keywords) or 'none'}\n"
        f"Example directions:\n- " + "\n- ".join(examples[:12]) + "\n\n"
        "Return strict JSON:\n"
        "{\n"
        '  "l2_name": string,\n'
        '  "definition": string,\n'
        '  "aliases": string[]\n'
        "}\n\n"
        "Rules:\n"
        "- l2_name should be short, stable, and canonical.\n"
        "- definition should be one sentence.\n"
        "- aliases can include paraphrases, abbreviations, and close variants.\n"
        "- Do not output markdown."
    )


def suggest_l1_count_range(l2_count: int) -> Tuple[int, int, int]:
    if l2_count <= 0:
        return (6, 8, 10)
    target = int(round(math.sqrt(l2_count) * 1.3))
    target = max(8, min(24, target))
    min_count = max(6, min(24, target - 3))
    max_count = max(min_count, min(24, target + 3))
    return (min_count, target, max_count)


def build_l2_canonical_items(l2_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_name: Dict[str, Dict[str, Any]] = {}
    for e in l2_entries:
        l2_name = str(e.get("l2_name", "")).strip()
        if not l2_name:
            continue
        key = l2_name.lower()
        topic_id = int(e.get("topic_id", -1))
        if key not in by_name:
            by_name[key] = {
                "l2_name": l2_name,
                "definition": str(e.get("definition", "")).strip(),
                "aliases": [str(x).strip() for x in (e.get("aliases") or []) if str(x).strip()],
                "topic_ids": [topic_id] if topic_id >= 0 else [],
            }
        else:
            if not by_name[key]["definition"] and str(e.get("definition", "")).strip():
                by_name[key]["definition"] = str(e.get("definition", "")).strip()
            aliases = set(by_name[key]["aliases"])
            for x in e.get("aliases") or []:
                sx = str(x).strip()
                if sx:
                    aliases.add(sx)
            by_name[key]["aliases"] = sorted(aliases)
            if topic_id >= 0 and topic_id not in by_name[key]["topic_ids"]:
                by_name[key]["topic_ids"].append(topic_id)
    return sorted(by_name.values(), key=lambda x: x["l2_name"].lower())


def build_l2_embedding_text(item: Dict[str, Any], axis: str) -> str:
    definition = str(item.get("definition", "")).strip()
    if definition:
        return f"{axis} L2: {item.get('l2_name', '')}. {definition}"
    return f"{axis} L2: {item.get('l2_name', '')}"


def embed_l2_items(
    axis: str,
    items: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    embedding_concurrency: int,
    log_dir: Path,
    cache_path: Path,
) -> np.ndarray:
    texts = [build_l2_embedding_text(item, axis) for item in items]
    keys = [stable_hash(f"{axis}|{str(item.get('l2_name', '')).strip().lower()}") for item in items]
    text_hashes = [stable_hash(t) for t in texts]
    cache_payload = load_json_if_exists(cache_path, {"version": 1, "model": model, "items": {}})
    cache_items = cache_payload.get("items", {}) if isinstance(cache_payload, dict) else {}
    if not isinstance(cache_items, dict):
        cache_items = {}
    cache_model = str(cache_payload.get("model", "")).strip() if isinstance(cache_payload, dict) else ""
    if cache_model and cache_model != model:
        cache_items = {}

    vectors: List[List[float] | None] = [None] * len(items)
    missing_indexes: List[int] = []
    for i, key in enumerate(keys):
        hit = cache_items.get(key)
        if (
            isinstance(hit, dict)
            and hit.get("text_hash") == text_hashes[i]
            and isinstance(hit.get("vector"), list)
            and len(hit.get("vector")) > 0
        ):
            vectors[i] = hit["vector"]
        else:
            missing_indexes.append(i)

    print(
        f"[taxonomy] axis={axis} L2 embedding cache: hits={len(items)-len(missing_indexes)} "
        f"misses={len(missing_indexes)} total={len(items)}"
    )
    if len(missing_indexes) == 0:
        return np.asarray(vectors, dtype=np.float32)

    embedding_max_batch = int(os.getenv("QWEN_EMBEDDING_MAX_BATCH", str(DEFAULT_EMBEDDING_MAX_BATCH)))
    effective_batch_size = max(1, min(batch_size, embedding_max_batch))
    total_batches = (len(missing_indexes) + effective_batch_size - 1) // effective_batch_size
    effective_concurrency = max(1, min(embedding_concurrency, total_batches))

    batch_specs: List[Tuple[int, List[int], List[str], int]] = []
    for i in range(0, len(missing_indexes), effective_batch_size):
        batch_indexes = missing_indexes[i : i + effective_batch_size]
        batch = [texts[j] for j in batch_indexes]
        batch_idx = i // effective_batch_size + 1
        batch_specs.append((i, batch_indexes, batch, batch_idx))

    with ThreadPoolExecutor(max_workers=effective_concurrency) as ex:
        futures = {
            ex.submit(
                call_embedding_api,
                api_key=api_key,
                base_url=base_url,
                model=model,
                texts=batch,
                log_dir=log_dir,
                batch_name=f"l2_batch_{i//effective_batch_size:04d}",
            ): (i, batch_indexes, batch_idx)
            for i, batch_indexes, batch, batch_idx in batch_specs
        }
        completed = 0
        for fut in as_completed(futures):
            i, batch_indexes, batch_idx = futures[fut]
            batch_vec = fut.result()
            completed += 1
            print(
                f"[taxonomy] axis={axis} L2 embedding progress: {completed}/{total_batches} "
                f"(batch {batch_idx})"
            )
            for local_idx, vec in enumerate(batch_vec):
                item_idx = batch_indexes[local_idx]
                vectors[item_idx] = vec
                cache_items[keys[item_idx]] = {
                    "text_hash": text_hashes[item_idx],
                    "vector": vec,
                    "updated_at": utc_now(),
                }
            dump_json(
                cache_path,
                {"version": 1, "model": model, "updated_at": utc_now(), "items": cache_items},
            )
    if any(v is None for v in vectors):
        raise RuntimeError("L2 embedding incomplete")
    return np.asarray(vectors, dtype=np.float32)


def cluster_l2_items(l2_embeddings: np.ndarray, target_clusters: int) -> List[int]:
    n_items = int(l2_embeddings.shape[0])
    n_clusters = max(1, min(target_clusters, n_items))
    if n_items == 1:
        return [0]
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="cosine", linkage="average")
    labels = model.fit_predict(l2_embeddings)
    return [int(x) for x in labels]


def build_l1_name_prompt(axis: str, cluster_id: int, members: List[Dict[str, Any]]) -> str:
    payload = [
        {
            "l2_name": str(m.get("l2_name", "")).strip(),
            "definition": str(m.get("definition", "")).strip(),
        }
        for m in members
    ]
    return (
        f"Axis: {axis}\n"
        f"Second-level cluster id: {cluster_id}\n"
        "Name this cluster as one L1 category.\n"
        f"Cluster members JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        "Return strict JSON:\n"
        "{\n"
        '  "l1_name": string,\n'
        '  "definition": string,\n'
        '  "aliases": string[]\n'
        "}\n\n"
        "Rules:\n"
        "- l1_name should be broad, stable, and concise.\n"
        "- definition should be one sentence.\n"
        "- aliases are optional close paraphrases.\n"
        "- Keep output factual and concise."
    )


def extract_topic_keywords(topic_model: BERTopic, topic_id: int, top_n: int = 10) -> List[str]:
    words = topic_model.get_topic(topic_id) or []
    return [str(w[0]).strip().lower() for w in words[:top_n] if w and w[0]]


def run_axis(
    axis: str,
    args: argparse.Namespace,
    api_key: str,
    base_url: str,
    emb_model: str,
    chat_model: str,
) -> Dict[str, Any]:
    axis_started = time.time()
    researchers_root = Path(args.researchers_root)
    out_axis_dir = Path(args.out_dir) / axis
    log_dir = Path(args.out_dir) / "api_logs" / axis
    ensure_dir(out_axis_dir)
    ensure_dir(log_dir)

    records = collect_records(researchers_root, axis)
    records = sort_records_stable(records)
    print(f"[taxonomy] axis={axis} collected records={len(records)}")
    if len(records) == 0:
        result = {"axis": axis, "generated_at": utc_now(), "records": 0, "topics": [], "l2": [], "l1": []}
        dump_json(out_axis_dir / "taxonomy.json", result)
        return result

    dump_json(out_axis_dir / "records.sample.json", [asdict(r) for r in records[:100]])
    dump_json(
        out_axis_dir / "records.json",
        [{"embedding_index": i, "record_key": make_record_key(r), **asdict(r)} for i, r in enumerate(records)],
    )

    embeddings = embed_records(
        records=records,
        api_key=api_key,
        base_url=base_url,
        model=emb_model,
        batch_size=args.embedding_batch_size,
        embedding_concurrency=args.embedding_concurrency,
        log_dir=log_dir,
        cache_path=out_axis_dir / "cache.embedding.json",
    )
    np.save(out_axis_dir / "embeddings.npy", embeddings)
    print(f"[taxonomy] axis={axis} embeddings ready shape={list(embeddings.shape)}")

    docs = [r.context for r in records]
    bertopic_cache_path = out_axis_dir / "cache.bertopic.json"
    bertopic_fingerprint = make_bertopic_fingerprint(
        records=records,
        emb_model=emb_model,
        min_topic_size=args.min_topic_size,
        random_seed=args.random_seed,
    )
    bertopic_cache = load_json_if_exists(bertopic_cache_path, {})
    topics: List[int]
    topic_candidates: List[Dict[str, Any]]
    if (
        isinstance(bertopic_cache, dict)
        and bertopic_cache.get("fingerprint") == bertopic_fingerprint
        and isinstance(bertopic_cache.get("topics"), list)
        and isinstance(bertopic_cache.get("topic_candidates"), list)
        and len(bertopic_cache.get("topics")) == len(records)
    ):
        topics = [int(x) for x in bertopic_cache.get("topics", [])]
        topic_candidates = bertopic_cache.get("topic_candidates", [])
        print(
            f"[taxonomy] axis={axis} BERTopic cache hit "
            f"(topics={len(topic_candidates)}, assignments={len(topics)})"
        )
    else:
        np.random.seed(int(args.random_seed))
        topic_model, topics = build_topic_model(
            docs,
            embeddings,
            min_topic_size=args.min_topic_size,
            random_seed=args.random_seed,
        )
        print(f"[taxonomy] axis={axis} BERTopic finished assignments={len(topics)}")
        topic_model.save(str(out_axis_dir / "bertopic_model"), serialization="safetensors", save_ctfidf=True)

        by_topic: Dict[int, List[int]] = {}
        for i, t in enumerate(topics):
            tid = int(t)
            by_topic.setdefault(tid, []).append(i)

        topic_candidates = []
        for tid, idxs in sorted(by_topic.items(), key=lambda x: x[0]):
            if tid == -1:
                continue
            keywords = extract_topic_keywords(topic_model, tid, top_n=12)
            examples = [records[i].text for i in idxs[:20]]
            topic_candidates.append(
                {
                    "topic_id": tid,
                    "size": len(idxs),
                    "keywords": keywords,
                    "examples": examples,
                }
            )
        dump_json(
            bertopic_cache_path,
            {
                "version": 1,
                "generated_at": utc_now(),
                "fingerprint": bertopic_fingerprint,
                "random_seed": int(args.random_seed),
                "min_topic_size": int(args.min_topic_size),
                "topics": [int(t) for t in topics],
                "topic_candidates": topic_candidates,
            },
        )

    dump_json(out_axis_dir / "topics.assignments.json", [{"i": i, "topic_id": int(t)} for i, t in enumerate(topics)])
    dump_json(out_axis_dir / "topics.candidates.json", topic_candidates)
    print(f"[taxonomy] axis={axis} topic candidates={len(topic_candidates)}")

    l2_cache_path = out_axis_dir / "cache.l2.json"
    l2_cache = load_json_if_exists(l2_cache_path, {"version": 1, "items": {}})
    l2_cache_items = l2_cache.get("items", {}) if isinstance(l2_cache, dict) else {}
    if not isinstance(l2_cache_items, dict):
        l2_cache_items = {}

    topic_fingerprints = [
        make_topic_fingerprint(axis, c["keywords"], c["examples"]) for c in topic_candidates
    ]
    l2_miss_count = sum(1 for fp in topic_fingerprints if not isinstance(l2_cache_items.get(fp), dict))
    planned_chat_calls = l2_miss_count + (1 if len(topic_candidates) > 0 else 0)
    thinking_override = parse_bool_env(os.getenv("QWEN_ENABLE_THINKING"))
    thinking_max_calls = int(os.getenv("QWEN_THINKING_MAX_CALLS", "8"))
    if thinking_override is None:
        enable_thinking = planned_chat_calls <= thinking_max_calls
    else:
        enable_thinking = thinking_override
    print(
        f"[taxonomy] axis={axis} chat_calls={planned_chat_calls} "
        f"enable_thinking={enable_thinking} (threshold={thinking_max_calls})"
    )

    l2_entries: List[Dict[str, Any] | None] = [None] * len(topic_candidates)
    total_l2 = len(topic_candidates)
    chat_workers = max(1, min(args.chat_concurrency, max(1, l2_miss_count)))
    print(
        f"[taxonomy] axis={axis} L2 workers={chat_workers} "
        f"(requested={args.chat_concurrency}, misses={l2_miss_count}, total={total_l2})"
    )

    def to_l2_entry(c: Dict[str, Any], topic_fingerprint: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        l2_name = str(raw.get("l2_name", "")).strip() or f"{axis}_topic_{c['topic_id']}"
        return {
            "topic_id": c["topic_id"],
            "size": c["size"],
            "keywords": c["keywords"],
            "examples": c["examples"][:8],
            "topic_fingerprint": topic_fingerprint,
            "l2_name": l2_name,
            "definition": str(raw.get("definition", "")).strip(),
            "aliases": [str(x).strip() for x in (raw.get("aliases") or []) if str(x).strip()],
        }

    done = 0
    pending = []
    for idx0, c in enumerate(topic_candidates):
        topic_fingerprint = topic_fingerprints[idx0]
        raw = l2_cache_items.get(topic_fingerprint)
        if isinstance(raw, dict):
            l2_entries[idx0] = to_l2_entry(c, topic_fingerprint, raw)
            done += 1
            print(
                f"[taxonomy] axis={axis} L2 progress: {done}/{total_l2} "
                f"(topic_id={c['topic_id']}, cache hit)"
            )
        else:
            pending.append((idx0, c, topic_fingerprint))

    if pending:
        with ThreadPoolExecutor(max_workers=chat_workers) as ex:
            futures = {}
            for idx0, c, topic_fingerprint in pending:
                prompt = build_l2_prompt(axis, c["topic_id"], c["keywords"], c["examples"])
                fut = ex.submit(
                    call_chat_json,
                    api_key=api_key,
                    base_url=base_url,
                    model=chat_model,
                    prompt=prompt,
                    log_dir=log_dir,
                    log_name=f"l2_topic_{c['topic_id']}",
                    max_tokens=260,
                    enable_thinking=enable_thinking,
                )
                futures[fut] = (idx0, c, topic_fingerprint)

            for fut in as_completed(futures):
                idx0, c, topic_fingerprint = futures[fut]
                raw = fut.result()
                l2_cache_items[topic_fingerprint] = raw
                l2_entries[idx0] = to_l2_entry(c, topic_fingerprint, raw)
                dump_json(
                    l2_cache_path,
                    {
                        "version": 1,
                        "updated_at": utc_now(),
                        "items": l2_cache_items,
                    },
                )
                done += 1
                print(
                    f"[taxonomy] axis={axis} L2 progress: {done}/{total_l2} "
                    f"(topic_id={c['topic_id']}, api)"
                )
                time.sleep(0.05)

    if any(x is None for x in l2_entries):
        raise RuntimeError("L2 labeling incomplete: some topics are missing labels")

    dump_json(out_axis_dir / "taxonomy.l2.json", l2_entries)

    l1_min, l1_target, l1_max = suggest_l1_count_range(len(l2_entries))
    l2_items = build_l2_canonical_items(l2_entries)
    dump_json(out_axis_dir / "l1.second_level.items.json", l2_items)
    print(f"[taxonomy] axis={axis} L1 second-level items={len(l2_items)} target={l1_target}")

    l2_emb = embed_l2_items(
        axis=axis,
        items=l2_items,
        api_key=api_key,
        base_url=base_url,
        model=emb_model,
        batch_size=args.embedding_batch_size,
        embedding_concurrency=args.embedding_concurrency,
        log_dir=log_dir,
        cache_path=out_axis_dir / "cache.l1.embedding.json",
    )
    np.save(out_axis_dir / "l1.second_level.embeddings.npy", l2_emb)
    second_labels = cluster_l2_items(l2_emb, target_clusters=l1_target)

    second_assignments = []
    by_second_cluster: Dict[int, List[Dict[str, Any]]] = {}
    for i, item in enumerate(l2_items):
        cid = int(second_labels[i])
        second_assignments.append(
            {
                "l2_index": i,
                "l2_name": item.get("l2_name"),
                "topic_ids": item.get("topic_ids", []),
                "second_cluster_id": cid,
            }
        )
        by_second_cluster.setdefault(cid, []).append(item)
    dump_json(out_axis_dir / "l1.second_level.assignments.json", second_assignments)
    dump_json(
        out_axis_dir / "l1.second_level.clusters.json",
        [
            {
                "second_cluster_id": cid,
                "size": len(members),
                "l2_names": [m.get("l2_name") for m in members],
            }
            for cid, members in sorted(by_second_cluster.items(), key=lambda x: x[0])
        ],
    )

    l1_cache_path = out_axis_dir / "cache.l1.json"
    l1_cache = load_json_if_exists(l1_cache_path, {"version": 1, "items": {}})
    l1_cache_items = l1_cache.get("items", {}) if isinstance(l1_cache, dict) else {}
    if not isinstance(l1_cache_items, dict):
        l1_cache_items = {}

    l1_names_payload = []
    l1_clean = []
    l2_to_l1: Dict[str, str] = {}
    l2_to_second_cluster: Dict[str, int] = {}
    cluster_ids = sorted(by_second_cluster.keys())
    print(f"[taxonomy] axis={axis} L1 naming start clusters={len(cluster_ids)}")
    l1_workers = max(1, min(args.chat_concurrency, max(1, len(cluster_ids))))
    print(
        f"[taxonomy] axis={axis} L1 workers={l1_workers} "
        f"(requested={args.chat_concurrency}, total={len(cluster_ids)})"
    )

    def to_l1_payload(cid: int, fp: str, members: List[Dict[str, Any]], raw: Dict[str, Any], source: str) -> Dict[str, Any]:
        l1_name = str(raw.get("l1_name", "")).strip() or f"{axis}_l1_cluster_{cid}"
        definition = str(raw.get("definition", "")).strip()
        aliases = [str(x).strip() for x in (raw.get("aliases") or []) if str(x).strip()]
        l2_names = [str(m.get("l2_name", "")).strip() for m in members if str(m.get("l2_name", "")).strip()]
        return {
            "second_cluster_id": cid,
            "cluster_fingerprint": fp,
            "source": source,
            "l1_name": l1_name,
            "definition": definition,
            "aliases": aliases,
            "l2_names": l2_names,
        }

    done_l1 = 0
    pending_l1: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    for cid in cluster_ids:
        members = by_second_cluster[cid]
        fp = make_l1_cluster_fingerprint(axis, members)
        raw = l1_cache_items.get(fp)
        if isinstance(raw, dict):
            payload = to_l1_payload(cid, fp, members, raw, "cache")
            l1_names_payload.append(payload)
            done_l1 += 1
            print(
                f"[taxonomy] axis={axis} L1 naming progress: {done_l1}/{len(cluster_ids)} "
                f"(cluster={cid}, cache)"
            )
        else:
            pending_l1.append((cid, fp, members))

    if pending_l1:
        with ThreadPoolExecutor(max_workers=l1_workers) as ex:
            futures = {}
            for cid, fp, members in pending_l1:
                fut = ex.submit(
                    call_chat_json,
                    api_key=api_key,
                    base_url=base_url,
                    model=chat_model,
                    prompt=build_l1_name_prompt(axis, cid, members),
                    log_dir=log_dir,
                    log_name=f"l1_name_cluster_{cid}",
                    max_tokens=260,
                    enable_thinking=enable_thinking,
                )
                futures[fut] = (cid, fp, members)

            for fut in as_completed(futures):
                cid, fp, members = futures[fut]
                raw = fut.result()
                l1_cache_items[fp] = raw
                dump_json(
                    l1_cache_path,
                    {"version": 2, "updated_at": utc_now(), "items": l1_cache_items},
                )
                payload = to_l1_payload(cid, fp, members, raw, "api")
                l1_names_payload.append(payload)
                done_l1 += 1
                print(
                    f"[taxonomy] axis={axis} L1 naming progress: {done_l1}/{len(cluster_ids)} "
                    f"(cluster={cid}, api)"
                )

    l1_names_payload = sorted(l1_names_payload, key=lambda x: int(x["second_cluster_id"]))
    for item in l1_names_payload:
        l1_clean.append(
            {
                "name": item["l1_name"],
                "definition": item["definition"],
                "l2_names": item["l2_names"],
            }
        )
        cid = int(item["second_cluster_id"])
        l1_name = str(item["l1_name"])
        for l2n in item["l2_names"]:
            if l2n and l2n not in l2_to_l1:
                l2_to_l1[l2n] = l1_name
            if l2n and l2n not in l2_to_second_cluster:
                l2_to_second_cluster[l2n] = cid

    dump_json(out_axis_dir / "l1.naming.json", l1_names_payload)
    dump_json(out_axis_dir / "taxonomy.l1.json", l1_clean)
    print(f"[taxonomy] axis={axis} L1 naming done categories={len(l1_clean)}")

    topic_id_to_l2 = {int(e["topic_id"]): e["l2_name"] for e in l2_entries}

    assignments = []
    for i, t in enumerate(topics):
        topic_id = int(t)
        l2_name = topic_id_to_l2.get(topic_id)
        assignments.append(
            {
                "record_index": i,
                "axis": axis,
                "direction_text": records[i].text,
                "paper_id": records[i].paper_id,
                "paper_title": records[i].paper_title,
                "publication_year": records[i].publication_year,
                "researcher_id": records[i].researcher_id,
                "researcher_name": records[i].researcher_name,
                "topic_id": topic_id,
                "l2_name": l2_name,
                "second_cluster_id": l2_to_second_cluster.get(l2_name) if l2_name else None,
                "l1_name": l2_to_l1.get(l2_name) if l2_name else None,
            }
        )

    final_payload = {
        "axis": axis,
        "generated_at": utc_now(),
        "meta": {
            "researchers_root": str(researchers_root),
            "embedding_model": emb_model,
            "chat_model": chat_model,
            "enable_thinking": enable_thinking,
            "planned_chat_calls": planned_chat_calls,
            "l1_target_min": l1_min,
            "l1_target": l1_target,
            "l1_target_max": l1_max,
            "min_topic_size": args.min_topic_size,
            "records": len(records),
            "clusters": len(topic_candidates),
            "l2_unique_count": len(l2_items),
            "l1_second_clusters": len(cluster_ids),
        },
        "l1_categories": l1_clean,
        "l2_categories": l2_entries,
        "assignments": assignments,
    }
    dump_json(out_axis_dir / "taxonomy.json", final_payload)
    elapsed_sec = int(time.time() - axis_started)
    print(
        f"[taxonomy] axis={axis} completed in {elapsed_sec}s "
        f"(records={len(records)}, topics={len(topic_candidates)}, l1={len(l1_clean)})"
    )
    return final_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build L1/L2 taxonomy from problem/method directions.")
    parser.add_argument(
        "--researchers-root",
        default="data-repo/data/researchers",
        help="Root path containing researchers.index.json + profiles/",
    )
    parser.add_argument(
        "--out-dir",
        default="data-repo/data/taxonomy",
        help="Output root for taxonomy files and api logs.",
    )
    parser.add_argument(
        "--axis",
        default="both",
        choices=["problem", "method", "both"],
        help="Which axis to process.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=10)
    parser.add_argument("--embedding-concurrency", type=int, default=4)
    parser.add_argument("--chat-concurrency", type=int, default=4)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("QWEN_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("QWEN_API_KEY is required.")
    base_url = os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL).strip()
    emb_model = os.getenv("QWEN_EMBEDDING_MODEL", DEFAULT_EMBED_MODEL).strip()
    chat_model = os.getenv("QWEN_MODEL", DEFAULT_CHAT_MODEL).strip()

    axes = ["problem", "method"] if args.axis == "both" else [args.axis]
    summary = []
    for axis in axes:
        print(f"[taxonomy] axis={axis} start")
        result = run_axis(axis, args, api_key, base_url, emb_model, chat_model)
        summary.append(
            {
                "axis": axis,
                "records": result.get("meta", {}).get("records", result.get("records", 0)),
                "clusters": result.get("meta", {}).get("clusters", 0),
            }
        )
        print(f"[taxonomy] axis={axis} done")

    dump_json(Path(args.out_dir) / "taxonomy.summary.json", {"generated_at": utc_now(), "axes": summary})
    print(json.dumps({"ok": True, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
