# Awesome Affective Computing

This is a personal, ongoing collection of resources in Affective Computing, including research teams, directions, and representative papers.

The repository aims to provide a structured perspective on the landscape of affective computing research.

⚠️ **Disclaimer**

- This list is not comprehensive and may not cover all relevant works.
- The organization reflects personal interpretation and research interests.
- No ranking or endorsement is implied.

## Researcher Pipeline (OpenAlex + Qwen)

Current website structure:

- `/researchers`: researcher overview (search/filter by country, university, keyword)
- `/researchers/detail?id=<openalex_author_id>`: per-researcher detail page
- `/papers`: aggregated affective-related papers from tracked researchers

### Seed file

Base researcher info is stored in:

- `data/researchers/researcher.seed.json`

Current example researcher:

- `Jufeng Yang`
- Scholar: `https://scholar.google.com/citations?user=c5vDJv0AAAAJ`
- OpenAlex Author ID: `a5089409678`

### Environment variables

- `QWEN_API_KEY`: required unless using `--skip-ai`
- `QWEN_BASE_URL`: optional, defaults to `https://dashscope.aliyuncs.com/compatible-mode/v1`
- `QWEN_MODEL`: optional, defaults to `qwen-plus`

### Run

```bash
npm run researcher:build
```

Default behavior is incremental:

- Only new papers are fetched/analyzed.
- Existing papers from `researcher.profile.json` are reused.
- Cached AI results are reused from `paper-analysis-cache.json`.
- Works are deduplicated by title before final output.
  - If both published and preprint versions exist, published version is preferred.
  - If only preprints exist, `ArXiv.org` is preferred.

Optional flags:

```bash
node scripts/researcher-pipeline/run.mjs --max-papers 20 --delay-ms 300
node scripts/researcher-pipeline/run.mjs --skip-ai --max-papers 5
node scripts/researcher-pipeline/run.mjs --full-refresh
```

### Output

- `data/researchers/researcher.profile.json`: enriched researcher profile
- `data/researchers/cache/paper-analysis-cache.json`: per-paper AI cache

### Pipeline flow

1. Read researchers from `data/researchers/researcher.seed.json`
2. Fetch author profile + works from OpenAlex
3. Deduplicate works by title with source preference rules
4. Run AI analysis per paper (affective-related judgment + directions + keywords)
5. Build researcher-level summaries
6. Export profile JSON for website pages

## Disclaimer

- Researchers are continuously being added.
- The current list is not a filtered shortlist, ranking, or complete coverage of the field.
- Parts of this project are AI-assisted. Metadata extraction, topic labeling, and summaries may contain errors or omissions.
- Please verify important details with official paper pages, publishers, and OpenAlex records.
