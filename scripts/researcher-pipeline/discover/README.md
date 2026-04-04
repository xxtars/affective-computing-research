# Discover Pipeline (Potential Authors)

This document records the current `researcher:discover` pipeline logic and filters.

Script:
- `scripts/researcher-pipeline/discover-potential-authors.mjs`

Command:
- `npm run researcher:discover -- [options]`

## Goal

Find **potential researchers** not yet in seed, based on:
- co-authorship with seed researchers, and
- affective/emotion relevance signals from OpenAlex metadata.

Output:
- `data/researchers/potential-authors.json`

Seen cache (for excluding already scanned candidates):
- `data/researchers/potential-authors.seen.json`

---

## Default Parameters

- `--seed data/researchers/researcher.seed.json`
- `--out data/researchers/potential-authors.json`
- `--years 5`
- `--per-page 200`
- `--max-works-per-seed 500`
- `--max-candidate-works 120`
- `--candidate-concurrency 6`
- `--min-score 0.35`
- `--min-shared-works 1`
- `--max-candidates 1000`
- `--delay-ms 0`
- `--mailto ""`
- `--keywords "emotion,affective,affect,sentiment,depression,anxiety,stress,mood,empathy,micro-expression,facial expression,emotion recognition,multimodal sentiment"`

Discovery gates (default enabled):
- `--require-affective-shared-works true`
- `--require-recent-additional-affective-works true`

Seen-cache behavior (default enabled):
- `--seen-cache data/researchers/potential-authors.seen.json`
- `--exclude-seen true`
- `--mark-seen true`

---

## Pipeline Steps

1. Load seed researchers from `researcher.seed.json`.
2. For each seed author, fetch recent OpenAlex works in the time window.
3. Collect co-authors not in seed as candidate pool.
4. If `require-affective-shared-works=true`, only count shared works that match affective keywords.
5. Build candidate linkage stats:
   - `shared_work_count`
   - `shared_with_seed_count`
   - `shared_with_seed_authors`
   - `shared_works_sample`
6. Apply early candidate filters:
   - `shared_work_count >= min_shared_works`
   - exclude IDs in seen cache if `exclude-seen=true`
   - keep top `max-candidates` by shared work count
7. For each candidate (concurrent):
   - fetch OpenAlex author profile
   - fetch candidate recent works
   - compute keyword stats
   - compute quality gates
   - compute rule-based score
8. Apply final shortlist filters:
   - `score >= min_score`
   - additional affective works gate
   - recent-active gate
9. Save shortlisted results to `potential-authors.json`.
10. Update seen cache (if `mark-seen=true`) for all scanned candidates.

---

## Scoring Formula

Rule-based score:

`score = 0.45*affective_ratio + 0.25*coauthor_strength + 0.20*keyword_density_norm + 0.10*recent_shared_ratio`

Where:
- `affective_ratio`: fraction of candidate works matching keywords
- `coauthor_strength`: normalized shared coauth works (`shared_work_count / 6`, clamped 0..1)
- `keyword_density_norm`: normalized keyword density per work
- `recent_shared_ratio`: fraction of shared works in last 2 years

Recommended threshold in practice:
- high precision: `--min-score 0.55`
- balanced: `--min-score 0.45~0.55`

---

## Quality Gates (Current)

1) Shared works must be affective-related (default on):
- shared linkage only counts works with keyword hits.

2) Candidate must have **recent additional affective works** (default on):
- `recent_3y_additional_affective_works_count > 0`
- meaning:
  - works must be affective-related by keyword hit,
  - must be **additional** (not the shared works with seed),
  - and must be published in the last 3 years.

---

## Duplicate / Merge Risk Signals

The pipeline does **not auto-merge** authors.

It surfaces signals for manual review:
- `duplicate_signals.orcid_matches_seed`
- `duplicate_signals.name_matches_seed`
- `duplicate_signals.possible_unmerged_author`

Primary identity key remains:
- `openalex_author_id`

---

## Useful Examples

High-precision run:

```bash
npm run researcher:discover -- \
  --seed data/researchers/researcher.seed.json \
  --out data/researchers/potential-authors.json \
  --years 5 \
  --min-score 0.55 \
  --min-shared-works 2
```

Rescan all candidates ignoring seen cache:

```bash
npm run researcher:discover -- \
  --exclude-seen false
```
