# Awesome Affective Computing

This is a personal, ongoing collection of resources in Affective Computing, including research teams, directions, and representative papers.

The repository aims to provide a structured perspective on the landscape of affective computing research.

⚠️ **Disclaimer**

- This list is not comprehensive and may not cover all relevant works.
- The organization reflects personal interpretation and research interests.
- No ranking or endorsement is implied.

---

## 自动论文采集 Pipeline（Auto Paper Harvesting Pipeline）

### 1) 目标概述

本项目将 Affective Computing 论文库从人工维护升级为自动增量更新。

系统以 `data/researchers.json`（仅维护学者最小信息）为入口，自动完成：

1. 学者信息补全与对齐（Google Scholar / Semantic Scholar）
2. 拉取学者论文（Semantic Scholar API）
3. 关键词粗筛（低成本过滤）
4. 精筛 + 方向分类 + 打标签（当前为启发式，预留 Claude 接口）
5. 增量合并至 `data/papers.json`，更新 `data/pipeline_state.json`
6. 提交变更并触发站点重新部署（GitHub Actions 定时任务）

整体流程每周自动跑一次，同时支持手动触发。

### 2) 推荐目录结构

```text
awesome-affective-computing/
├── data/
│   ├── researchers.json
│   ├── researchers_enriched.json
│   ├── papers.json
│   ├── pipeline_state.json
│   ├── candidates.json
│   ├── classified.json
│   └── newly_rejected_ids.json
├── scripts/
│   ├── enrich_researchers.py
│   ├── fetch_papers.py
│   ├── classify_papers.py
│   ├── merge_papers.py
│   └── config.py
└── .github/workflows/
    └── update_papers.yml
```

### 3) 数据文件格式

#### `researchers.json`（手动维护：最小入口）

```json
[
  {
    "name": "",
    "google_scholar": "https://scholar.google.com.hk/citations?user=TxKNCSoAAAAJ"
  }
]
```

#### `researchers_enriched.json`（自动维护：补全后的学者索引）

自动补全 `semantic_scholar_id / institution / homepage` 等字段，`last_verified` 记录最近校验时间。

#### `pipeline_state.json`（自动维护，禁止手动编辑）

维护 `known_ids`（已收录）和 `rejected_ids`（已拒绝）并避免重复处理。

#### `papers.json`（自动生成）

支持 `source: "auto" | "manual"`，自动流程只追加 `auto` 论文，不覆盖手工条目。

### 4) Pipeline 流程（0 → 3）

- Step 0: `python scripts/enrich_researchers.py`
- Step 1: `python scripts/fetch_papers.py`
- Step 2: `python scripts/classify_papers.py`
- Step 3: `python scripts/merge_papers.py`

### 5) 一键本地运行

```bash
python scripts/enrich_researchers.py
python scripts/fetch_papers.py
python scripts/classify_papers.py
python scripts/merge_papers.py
```

> 可选环境变量：`S2_API_KEY`（Semantic Scholar API），`ANTHROPIC_API_KEY`（后续用于 Claude 分类）。
