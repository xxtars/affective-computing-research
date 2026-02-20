import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";

const OPENALEX_BASE_URL = "https://api.openalex.org";
const DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1";

function parseArgs(argv) {
  const args = {
    seed: "data/researchers/researcher.seed.json",
    out: "data/researchers/researcher.profile.json",
    cache: "data/researchers/cache/paper-analysis-cache.json",
    model: process.env.QWEN_MODEL || "qwen-plus",
    skipAi: false,
    maxPapers: null,
    delayMs: 200,
    fullRefresh: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--seed") args.seed = argv[++i];
    else if (token === "--out") args.out = argv[++i];
    else if (token === "--cache") args.cache = argv[++i];
    else if (token === "--model") args.model = argv[++i];
    else if (token === "--max-papers") args.maxPapers = Number(argv[++i]);
    else if (token === "--delay-ms") args.delayMs = Number(argv[++i]);
    else if (token === "--skip-ai") args.skipAi = true;
    else if (token === "--full-refresh") args.fullRefresh = true;
  }

  return args;
}

function normalizeAuthorId(rawId) {
  if (!rawId) throw new Error("openalex_author_id is required");
  const clean = String(rawId).trim();
  if (clean.startsWith("https://openalex.org/")) {
    const id = clean.split("/").pop();
    return id.toUpperCase();
  }
  return clean.toUpperCase().startsWith("A") ? clean.toUpperCase() : `A${clean}`;
}

function invertedIndexToText(indexObj) {
  if (!indexObj || typeof indexObj !== "object") return "";
  let maxPos = -1;
  for (const positions of Object.values(indexObj)) {
    if (Array.isArray(positions)) {
      for (const pos of positions) {
        if (typeof pos === "number" && pos > maxPos) maxPos = pos;
      }
    }
  }
  if (maxPos < 0) return "";

  const tokens = new Array(maxPos + 1).fill("");
  for (const [word, positions] of Object.entries(indexObj)) {
    if (!Array.isArray(positions)) continue;
    for (const pos of positions) {
      if (typeof pos === "number" && pos >= 0 && pos < tokens.length) {
        tokens[pos] = word;
      }
    }
  }
  return tokens.join(" ").replace(/\s+/g, " ").trim();
}

async function loadJson(filePath, fallback = null) {
  try {
    const content = await fs.readFile(filePath, "utf8");
    return JSON.parse(content);
  } catch (err) {
    if (err.code === "ENOENT") return fallback;
    throw err;
  }
}

async function saveJson(filePath, data) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJson(url) {
  const res = await fetch(url, {
    headers: {
      "User-Agent": "awesome-affective-computing-researcher-pipeline/1.0",
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

async function fetchAuthorProfile(authorId) {
  const url = `${OPENALEX_BASE_URL}/authors/${authorId}`;
  return fetchJson(url);
}

function normalizeTitle(title) {
  return String(title || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function isPreprintWork(work) {
  const type = String(work?.type || "").toLowerCase();
  const sourceName = String(work?.source?.display_name || work?.primary_source || "").toLowerCase();
  return type === "preprint" || sourceName.includes("arxiv");
}

function isArxivOrgSource(work) {
  const sourceName = String(work?.source?.display_name || work?.primary_source || "").toLowerCase();
  return sourceName === "arxiv.org";
}

function publishedVenuePriority(work) {
  const sourceType = String(work?.source?.type || "").toLowerCase();
  if (sourceType === "journal") return 5;
  if (sourceType === "conference") return 4;
  if (sourceType === "book_series") return 3;
  if (sourceType === "repository") return 1;
  return 2;
}

function isBetterWorkCandidate(candidate, current) {
  const candidatePreprint = isPreprintWork(candidate);
  const currentPreprint = isPreprintWork(current);
  if (candidatePreprint !== currentPreprint) return !candidatePreprint;

  if (candidatePreprint && currentPreprint) {
    const candidateArxivOrg = isArxivOrgSource(candidate);
    const currentArxivOrg = isArxivOrgSource(current);
    if (candidateArxivOrg !== currentArxivOrg) return candidateArxivOrg;
  }

  const venueA = publishedVenuePriority(candidate);
  const venueB = publishedVenuePriority(current);
  if (venueA !== venueB) return venueA > venueB;

  const citeA = candidate.cited_by_count || 0;
  const citeB = current.cited_by_count || 0;
  if (citeA !== citeB) return citeA > citeB;

  const dateA = candidate.publication_date || "";
  const dateB = current.publication_date || "";
  if (dateA !== dateB) return dateA > dateB;

  const idA = String(candidate.id || "");
  const idB = String(current.id || "");
  return idA > idB;
}

function dedupeWorksByTitle(works) {
  const byTitle = new Map();
  for (const work of works) {
    const key = normalizeTitle(work.title);
    if (!key) {
      byTitle.set(`__id__${work.id}`, work);
      continue;
    }
    const existing = byTitle.get(key);
    if (!existing || isBetterWorkCandidate(work, existing)) {
      byTitle.set(key, work);
    }
  }
  return Array.from(byTitle.values());
}

async function fetchAuthorWorks(authorId, { maxPapers = null, knownWorkIds = null, fullRefresh = false } = {}) {
  const works = [];
  const dedupedByTitle = new Map();
  let cursor = "*";
  const perPage = 200;
  const knownIds = knownWorkIds && !fullRefresh ? knownWorkIds : null;

  while (cursor) {
    const url = new URL(`${OPENALEX_BASE_URL}/works`);
    url.searchParams.set("filter", `author.id:https://openalex.org/${authorId}`);
    url.searchParams.set("per-page", String(perPage));
    url.searchParams.set("cursor", cursor);
    url.searchParams.set("sort", "publication_date:desc");

    const payload = await fetchJson(url.toString());
    const pageResults = payload.results || [];
    let pageKnownCount = 0;
    for (const work of pageResults) {
      if (knownIds && knownIds.has(work.id)) {
        pageKnownCount += 1;
        continue;
      }
      const abstract = invertedIndexToText(work.abstract_inverted_index);
      const primaryLocation = work.primary_location || null;
      const primarySource = primaryLocation?.source || null;
      const primaryTopic = work.primary_topic || null;
      const mappedWork = {
        id: work.id,
        openalex_url: work.id || null,
        title: work.display_name || "",
        publication_year: work.publication_year || null,
        publication_date: work.publication_date || null,
        doi: work.doi || null,
        doi_url: work.doi ? `https://doi.org/${String(work.doi).replace(/^https?:\/\/doi.org\//, "")}` : null,
        type: work.type || null,
        type_crossref: work.type_crossref || null,
        language: work.language || null,
        is_retracted: Boolean(work.is_retracted),
        is_paratext: Boolean(work.is_paratext),
        cited_by_count: work.cited_by_count || 0,
        primary_source: primarySource?.display_name || null,
        source: {
          id: primarySource?.id || null,
          display_name: primarySource?.display_name || null,
          type: primarySource?.type || null,
          issn_l: primarySource?.issn_l || null,
          is_in_doaj: primarySource?.is_in_doaj ?? null,
          host_organization_name: primarySource?.host_organization_name || null,
          host_organization_lineage_names: primarySource?.host_organization_lineage_names || [],
        },
        links: {
          openalex: work.id || null,
          landing_page: primaryLocation?.landing_page_url || null,
          pdf: primaryLocation?.pdf_url || null,
          primary_topic_openalex: primaryTopic?.id || null,
          source_openalex: primarySource?.id || null,
        },
        concepts: (work.concepts || []).slice(0, 8).map((c) => c.display_name).filter(Boolean),
        abstract,
        openalex_analysis: {
          primary_topic: primaryTopic
            ? {
                id: primaryTopic.id || null,
                name: primaryTopic.display_name || null,
                score: typeof primaryTopic.score === "number" ? primaryTopic.score : null,
                subfield: primaryTopic.subfield?.display_name || null,
                field: primaryTopic.field?.display_name || null,
                domain: primaryTopic.domain?.display_name || null,
              }
            : null,
          topics: (work.topics || []).slice(0, 12).map((t) => ({
            id: t.id || null,
            name: t.display_name || null,
            score: typeof t.score === "number" ? t.score : null,
            subfield: t.subfield?.display_name || null,
            field: t.field?.display_name || null,
            domain: t.domain?.display_name || null,
          })),
          concepts: (work.concepts || []).slice(0, 20).map((c) => ({
            id: c.id || null,
            name: c.display_name || null,
            score: typeof c.score === "number" ? c.score : null,
            level: typeof c.level === "number" ? c.level : null,
          })),
          keywords: (work.keywords || []).slice(0, 20).map((k) => ({
            id: k.id || null,
            name: k.display_name || null,
            score: typeof k.score === "number" ? k.score : null,
          })),
          sustainable_development_goals: (work.sustainable_development_goals || []).slice(0, 17).map((s) => ({
            id: s.id || null,
            display_name: s.display_name || null,
            score: typeof s.score === "number" ? s.score : null,
          })),
          open_access: work.open_access || null,
          citation_normalized_percentile: work.citation_normalized_percentile || null,
          fwci: typeof work.fwci === "number" ? work.fwci : null,
          counts_by_year: work.counts_by_year || [],
          institutions_distinct_count: work.institutions_distinct_count || 0,
          countries_distinct_count: work.countries_distinct_count || 0,
          referenced_works_count: Array.isArray(work.referenced_works) ? work.referenced_works.length : 0,
        },
      };
      works.push(mappedWork);

      const titleKey = normalizeTitle(mappedWork.title);
      if (titleKey) {
        const existing = dedupedByTitle.get(titleKey);
        if (!existing || isBetterWorkCandidate(mappedWork, existing)) {
          dedupedByTitle.set(titleKey, mappedWork);
        }
      }

      if (maxPapers && dedupedByTitle.size >= maxPapers) {
        return dedupeWorksByTitle(works).slice(0, maxPapers);
      }
    }

    // Incremental mode: once we hit a full known page, we can stop paging.
    if (knownIds && pageResults.length > 0 && pageKnownCount === pageResults.length) {
      break;
    }

    cursor = payload.meta?.next_cursor || null;
    if (!cursor) break;
  }

  return dedupeWorksByTitle(works);
}

function paperCacheKey(work) {
  const fingerprint = `${work.id}|${work.title}|${work.abstract}`;
  return crypto.createHash("sha256").update(fingerprint).digest("hex");
}

function buildPaperPrompt(researcher, work) {
  return `You are analyzing whether a paper matches user interest topics.\n\nResearcher: ${researcher.name}\nInterest topics: ${(researcher.interest_topics || []).join(", ")}\n\nPaper metadata:\n- Title: ${work.title}\n- Year: ${work.publication_year || "unknown"}\n- Venue: ${work.primary_source || "unknown"}\n- Concepts: ${(work.concepts || []).join(", ") || "none"}\n- Abstract: ${work.abstract || "(empty)"}\n\nReturn strict JSON with this schema:\n{\n  "is_interesting": boolean,\n  "relevance_score": number,\n  "confidence": number,\n  "reason": string,\n  "evidence": string[],\n  "keywords": string[],\n  "research_directions": string[]\n}\n\nRules:\n- relevance_score and confidence must be in [0,1]\n- evidence max 3 short strings\n- if is_interesting=false, keywords/research_directions can be empty arrays.`;
}

function buildSummaryPrompt(researcher, analyzedWorks) {
  const interesting = analyzedWorks
    .filter((w) => w.analysis?.is_interesting)
    .map((w) => ({
      title: w.title,
      year: w.publication_year,
      cited_by_count: w.cited_by_count,
      directions: w.analysis.research_directions || [],
      keywords: w.analysis.keywords || [],
    }))
    .slice(0, 120);

  return `Based on interesting papers for researcher ${researcher.name}, summarize main research directions.\n\nInput papers JSON:\n${JSON.stringify(interesting)}\n\nReturn strict JSON:\n{\n  "top_research_directions": [{"name": string, "weight": number}],\n  "trend_summary": string,\n  "representative_papers": [{"title": string, "why": string}]\n}\n\nRules:\n- max 8 top directions\n- weight in [0,1] and sorted desc\n- representative_papers max 8`;
}

async function callQwenChat({ apiKey, baseUrl, model, userPrompt, temperature = 0 }) {
  const url = `${baseUrl.replace(/\/$/, "")}/chat/completions`;
  const body = {
    model,
    temperature,
    response_format: { type: "json_object" },
    messages: [
      { role: "system", content: "You are a precise research analysis assistant. Return valid JSON only." },
      { role: "user", content: userPrompt },
    ],
  };

  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify(body),
  });

  const text = await res.text();
  if (!res.ok) {
    throw new Error(`Qwen API error: ${res.status} ${res.statusText} - ${text}`);
  }

  const payload = JSON.parse(text);
  const content = payload.choices?.[0]?.message?.content;
  if (!content) throw new Error("Qwen API returned empty content");

  return JSON.parse(content);
}

function clamp01(value, fallback = 0) {
  const num = Number(value);
  if (!Number.isFinite(num)) return fallback;
  if (num < 0) return 0;
  if (num > 1) return 1;
  return num;
}

function normalizePaperAnalysis(raw) {
  const keywords = Array.isArray(raw?.keywords) ? raw.keywords.filter(Boolean).slice(0, 12) : [];
  const directions = Array.isArray(raw?.research_directions)
    ? raw.research_directions.filter(Boolean).slice(0, 10)
    : [];
  const evidence = Array.isArray(raw?.evidence) ? raw.evidence.filter(Boolean).slice(0, 3) : [];

  return {
    is_interesting: Boolean(raw?.is_interesting),
    relevance_score: clamp01(raw?.relevance_score, 0),
    confidence: clamp01(raw?.confidence, 0),
    reason: typeof raw?.reason === "string" ? raw.reason : "",
    evidence,
    keywords,
    research_directions: directions,
  };
}

function fallbackSummary(analyzedWorks) {
  const directionCounts = new Map();
  for (const work of analyzedWorks) {
    const analysis = work.analysis;
    if (!analysis?.is_interesting) continue;
    for (const direction of analysis.research_directions || []) {
      directionCounts.set(direction, (directionCounts.get(direction) || 0) + 1);
    }
  }

  const total = [...directionCounts.values()].reduce((a, b) => a + b, 0) || 1;
  const top = [...directionCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 8)
    .map(([name, count]) => ({ name, weight: Number((count / total).toFixed(3)) }));

  const representatives = analyzedWorks
    .filter((w) => w.analysis?.is_interesting)
    .sort((a, b) => b.cited_by_count - a.cited_by_count)
    .slice(0, 8)
    .map((w) => ({ title: w.title, why: `Highly cited (${w.cited_by_count}).` }));

  return {
    top_research_directions: top,
    trend_summary: "AI summary unavailable. Generated by frequency fallback.",
    representative_papers: representatives,
  };
}

async function analyzePaper({ researcher, work, args, cache, qwenConfig }) {
  const cacheKey = paperCacheKey(work);
  if (cache[cacheKey]) return { analysis: cache[cacheKey], fromCache: true };

  if (args.skipAi) {
    const skipped = {
      is_interesting: false,
      relevance_score: 0,
      confidence: 0,
      reason: "AI skipped by --skip-ai",
      evidence: [],
      keywords: [],
      research_directions: [],
    };
    cache[cacheKey] = skipped;
    return { analysis: skipped, fromCache: false };
  }

  const prompt = buildPaperPrompt(researcher, work);
  let attempt = 0;
  let lastErr = null;

  while (attempt < 3) {
    attempt += 1;
    try {
      const raw = await callQwenChat({
        apiKey: qwenConfig.apiKey,
        baseUrl: qwenConfig.baseUrl,
        model: args.model,
        userPrompt: prompt,
        temperature: 0,
      });
      const normalized = normalizePaperAnalysis(raw);
      cache[cacheKey] = normalized;
      return { analysis: normalized, fromCache: false };
    } catch (err) {
      lastErr = err;
      await sleep(500 * attempt);
    }
  }

  throw lastErr;
}

async function run() {
  const args = parseArgs(process.argv.slice(2));
  const seed = await loadJson(args.seed);
  if (!seed?.researchers?.length) {
    throw new Error(`No researchers found in ${args.seed}`);
  }

  const qwenApiKey = process.env.QWEN_API_KEY;
  if (!args.skipAi && !qwenApiKey) {
    throw new Error("QWEN_API_KEY is required unless --skip-ai is set");
  }
  const qwenBaseUrl = process.env.QWEN_BASE_URL || DEFAULT_QWEN_BASE_URL;

  const cache = (await loadJson(args.cache, {})) || {};
  const previousOutput = (await loadJson(args.out, null)) || null;
  const generatedAt = new Date().toISOString();
  const output = {
    generated_at: generatedAt,
    pipeline_version: "v0.1.0",
    run_config: {
      model: args.model,
      skip_ai: args.skipAi,
      full_refresh: args.fullRefresh,
      max_papers: args.maxPapers,
      delay_ms: args.delayMs,
    },
    researchers: [],
  };

  for (const researcher of seed.researchers) {
    const authorId = normalizeAuthorId(researcher.openalex_author_id);
    const previousResearcher = previousOutput?.researchers?.find(
      (item) => item?.identity?.openalex_author_id === authorId
    );
    const previousWorks = Array.isArray(previousResearcher?.works) ? previousResearcher.works : [];
    const knownWorkIds = new Set(previousWorks.map((w) => w.id).filter(Boolean));

    console.log(`Fetching OpenAlex author: ${authorId}`);
    const authorProfile = await fetchAuthorProfile(authorId);

    console.log(`Fetching works for ${researcher.name} (${args.fullRefresh ? "full" : "incremental"})`);
    const newWorks = await fetchAuthorWorks(authorId, {
      maxPapers: args.maxPapers,
      knownWorkIds,
      fullRefresh: args.fullRefresh,
    });
    console.log(`Fetched ${newWorks.length} new works`);

    const analyzedNewWorks = [];
    for (let i = 0; i < newWorks.length; i += 1) {
      const work = newWorks[i];
      process.stdout.write(`Analyzing new paper ${i + 1}/${newWorks.length}\r`);
      const { analysis, fromCache } = await analyzePaper({
        researcher,
        work,
        args,
        cache,
        qwenConfig: { apiKey: qwenApiKey, baseUrl: qwenBaseUrl },
      });
      analyzedNewWorks.push({ ...work, analysis });
      if (!fromCache && args.delayMs > 0) await sleep(args.delayMs);
    }
    if (newWorks.length > 0) process.stdout.write("\n");

    const mergedWorks = [...analyzedNewWorks];
    if (!args.fullRefresh) {
      for (const oldWork of previousWorks) {
        if (!oldWork?.id) continue;
        if (newWorks.some((nw) => nw.id === oldWork.id)) continue;
        mergedWorks.push(oldWork);
      }
    }
    const dedupedMergedWorks = dedupeWorksByTitle(mergedWorks);

    dedupedMergedWorks.sort((a, b) => {
      const dateA = a.publication_date || "";
      const dateB = b.publication_date || "";
      if (dateA && dateB) return dateA > dateB ? -1 : dateA < dateB ? 1 : 0;
      const yearA = a.publication_year || 0;
      const yearB = b.publication_year || 0;
      return yearB - yearA;
    });

    const interestingWorks = dedupedMergedWorks.filter((w) => w.analysis?.is_interesting);

    let topicSummary = fallbackSummary(dedupedMergedWorks);
    if (!args.skipAi) {
      try {
        const summaryRaw = await callQwenChat({
          apiKey: qwenApiKey,
          baseUrl: qwenBaseUrl,
          model: args.model,
          userPrompt: buildSummaryPrompt(researcher, dedupedMergedWorks),
          temperature: 0,
        });
        topicSummary = {
          top_research_directions: Array.isArray(summaryRaw?.top_research_directions)
            ? summaryRaw.top_research_directions.slice(0, 8).map((d) => ({
                name: String(d?.name || ""),
                weight: clamp01(d?.weight, 0),
              }))
            : topicSummary.top_research_directions,
          trend_summary:
            typeof summaryRaw?.trend_summary === "string"
              ? summaryRaw.trend_summary
              : topicSummary.trend_summary,
          representative_papers: Array.isArray(summaryRaw?.representative_papers)
            ? summaryRaw.representative_papers.slice(0, 8).map((p) => ({
                title: String(p?.title || ""),
                why: String(p?.why || ""),
              }))
            : topicSummary.representative_papers,
        };
      } catch (err) {
        console.warn(`Summary generation failed, fallback used: ${err.message}`);
      }
    }

    output.researchers.push({
      identity: {
        name: researcher.name,
        google_scholar: researcher.google_scholar,
        openalex_author_id: authorId,
        openalex_author_url: `https://openalex.org/${authorId}`,
        interest_topics: researcher.interest_topics || [],
      },
      affiliation: {
        last_known_institution: authorProfile.last_known_institutions?.[0]?.display_name || null,
        last_known_country: authorProfile.last_known_institutions?.[0]?.country_code || null,
      },
      metrics: {
        works_count: authorProfile.works_count || 0,
        cited_by_count: authorProfile.cited_by_count || 0,
        h_index: authorProfile.summary_stats?.h_index || null,
        i10_index: authorProfile.summary_stats?.i10_index || null,
      },
      topic_summary: topicSummary,
      stats: {
        analyzed_works_count: dedupedMergedWorks.length,
        interesting_works_count: interestingWorks.length,
        new_works_count: analyzedNewWorks.length,
        deduped_works_count: dedupedMergedWorks.length,
      },
      works: dedupedMergedWorks,
    });
  }

  await saveJson(args.cache, cache);
  await saveJson(args.out, output);
  console.log(`Profile exported to ${args.out}`);
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
