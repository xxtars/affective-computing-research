import fs from "node:fs/promises";
import path from "node:path";

const OPENALEX_BASE = "https://api.openalex.org";

function parseArgs(argv) {
  const args = {
    seed: "data/researchers/researcher.seed.json",
    out: "data/researchers/potential-authors.json",
    years: 5,
    perPage: 200,
    maxWorksPerSeed: 500,
    maxCandidateWorks: 120,
    candidateConcurrency: 6,
    minScore: 0.35,
    minSharedWorks: 1,
    maxCandidates: 1000,
    requireAffectiveSharedWorks: true,
    requireRecentAdditionalAffectiveWorks: true,
    seenCache: "data/researchers/potential-authors.seen.json",
    excludeSeen: true,
    markSeen: true,
    reviewedOut: "",
    reviewedCitationThreshold: 120,
    delayMs: 0,
    mailto: "",
    keywords:
      "emotion,affective,affect,sentiment,depression,anxiety,stress,mood,empathy,micro-expression,facial expression,emotion recognition,multimodal sentiment",
  };

  for (let i = 2; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--seed") args.seed = argv[++i];
    else if (token === "--out") args.out = argv[++i];
    else if (token === "--years") args.years = Number(argv[++i] || args.years);
    else if (token === "--per-page") args.perPage = Number(argv[++i] || args.perPage);
    else if (token === "--max-works-per-seed")
      args.maxWorksPerSeed = Number(argv[++i] || args.maxWorksPerSeed);
    else if (token === "--max-candidate-works")
      args.maxCandidateWorks = Number(argv[++i] || args.maxCandidateWorks);
    else if (token === "--candidate-concurrency")
      args.candidateConcurrency = Number(argv[++i] || args.candidateConcurrency);
    else if (token === "--min-score") args.minScore = Number(argv[++i] || args.minScore);
    else if (token === "--min-shared-works")
      args.minSharedWorks = Number(argv[++i] || args.minSharedWorks);
    else if (token === "--max-candidates") args.maxCandidates = Number(argv[++i] || args.maxCandidates);
    else if (token === "--require-affective-shared-works")
      args.requireAffectiveSharedWorks = String(argv[++i] || "true").toLowerCase() !== "false";
    else if (token === "--require-recent-additional-affective-works")
      args.requireRecentAdditionalAffectiveWorks =
        String(argv[++i] || "true").toLowerCase() !== "false";
    else if (token === "--seen-cache") args.seenCache = String(argv[++i] || args.seenCache);
    else if (token === "--exclude-seen")
      args.excludeSeen = String(argv[++i] || "true").toLowerCase() !== "false";
    else if (token === "--mark-seen")
      args.markSeen = String(argv[++i] || "true").toLowerCase() !== "false";
    else if (token === "--reviewed-out") args.reviewedOut = String(argv[++i] || "");
    else if (token === "--reviewed-citation-threshold")
      args.reviewedCitationThreshold = Number(argv[++i] || args.reviewedCitationThreshold);
    else if (token === "--delay-ms") args.delayMs = Number(argv[++i] || args.delayMs);
    else if (token === "--mailto") args.mailto = String(argv[++i] || "");
    else if (token === "--keywords") args.keywords = String(argv[++i] || args.keywords);
  }

  args.years = Math.max(1, Number.isFinite(args.years) ? args.years : 5);
  args.perPage = Math.max(25, Math.min(200, Number.isFinite(args.perPage) ? args.perPage : 200));
  args.maxWorksPerSeed = Math.max(1, Number.isFinite(args.maxWorksPerSeed) ? args.maxWorksPerSeed : 500);
  args.maxCandidateWorks = Math.max(
    10,
    Number.isFinite(args.maxCandidateWorks) ? args.maxCandidateWorks : 120
  );
  args.candidateConcurrency = Math.max(
    1,
    Number.isFinite(args.candidateConcurrency) ? args.candidateConcurrency : 6
  );
  args.minScore = Math.max(0, Math.min(1, Number.isFinite(args.minScore) ? args.minScore : 0.35));
  args.minSharedWorks = Math.max(1, Number.isFinite(args.minSharedWorks) ? args.minSharedWorks : 1);
  args.maxCandidates = Math.max(1, Number.isFinite(args.maxCandidates) ? args.maxCandidates : 1000);
  args.delayMs = Math.max(0, Number.isFinite(args.delayMs) ? args.delayMs : 0);
  args.reviewedCitationThreshold = Math.max(
    0,
    Number.isFinite(args.reviewedCitationThreshold) ? args.reviewedCitationThreshold : 120
  );
  args.keywordList = String(args.keywords)
    .split(",")
    .map((x) => x.trim().toLowerCase())
    .filter(Boolean);
  return args;
}

async function loadJson(filePath, fallback) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch {
    return fallback;
  }
}

async function saveJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function normalizeAuthorId(id) {
  const raw = String(id || "").trim();
  if (!raw) return "";
  const m = raw.match(/A\d+/i);
  return (m ? m[0] : raw).toUpperCase();
}

function normalizeOrcid(value) {
  const raw = String(value || "").trim();
  if (!raw) return null;
  const m = raw.match(/(\d{4}-\d{4}-\d{4}-[\dX]{4})/i);
  return m ? `https://orcid.org/${m[1].toUpperCase()}` : null;
}

function normalizeName(name) {
  return String(name || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function cutoffDateByYears(years) {
  const now = new Date();
  const y = now.getUTCFullYear() - years + 1;
  return `${y}-01-01`;
}

async function sleep(ms) {
  if (ms > 0) await new Promise((resolve) => setTimeout(resolve, ms));
}

async function fetchJsonWithRetry(url, { retries = 3, delayMs = 400, headers = {} } = {}) {
  let lastErr = null;
  for (let i = 0; i < retries; i += 1) {
    try {
      const res = await fetch(url, { headers });
      if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
      return await res.json();
    } catch (err) {
      lastErr = err;
      if (i < retries - 1) await sleep(delayMs * (i + 1));
    }
  }
  throw lastErr || new Error(`Request failed: ${url}`);
}

async function fetchAuthorWorks(authorId, args) {
  const works = [];
  const cutoff = cutoffDateByYears(args.years);
  let page = 1;
  while (works.length < args.maxWorksPerSeed) {
    const url = new URL(`${OPENALEX_BASE}/works`);
    url.searchParams.set("filter", `authorships.author.id:https://openalex.org/${authorId},from_publication_date:${cutoff}`);
    url.searchParams.set("per-page", String(args.perPage));
    url.searchParams.set("page", String(page));
    url.searchParams.set("sort", "publication_date:desc");
    if (args.mailto) url.searchParams.set("mailto", args.mailto);
    const json = await fetchJsonWithRetry(url.toString());
    const rows = Array.isArray(json?.results) ? json.results : [];
    if (rows.length === 0) break;
    works.push(...rows);
    if (rows.length < args.perPage) break;
    page += 1;
    await sleep(args.delayMs);
  }
  return works.slice(0, args.maxWorksPerSeed);
}

async function fetchCandidateProfile(authorId, args) {
  const authorUrl = new URL(`${OPENALEX_BASE}/authors/${authorId}`);
  if (args.mailto) authorUrl.searchParams.set("mailto", args.mailto);
  const profile = await fetchJsonWithRetry(authorUrl.toString());

  const works = [];
  const cutoff = cutoffDateByYears(args.years);
  let page = 1;
  while (works.length < args.maxCandidateWorks) {
    const worksUrl = new URL(`${OPENALEX_BASE}/works`);
    worksUrl.searchParams.set("filter", `authorships.author.id:https://openalex.org/${authorId},from_publication_date:${cutoff}`);
    worksUrl.searchParams.set("per-page", String(Math.min(200, args.perPage)));
    worksUrl.searchParams.set("page", String(page));
    worksUrl.searchParams.set("sort", "publication_date:desc");
    if (args.mailto) worksUrl.searchParams.set("mailto", args.mailto);
    const json = await fetchJsonWithRetry(worksUrl.toString());
    const rows = Array.isArray(json?.results) ? json.results : [];
    if (rows.length === 0) break;
    works.push(...rows);
    if (rows.length < args.perPage) break;
    page += 1;
    await sleep(args.delayMs);
  }
  return { profile, works: works.slice(0, args.maxCandidateWorks) };
}

function textOfWork(work) {
  const title = String(work?.title || "");
  const concepts = (work?.concepts || []).map((x) => x?.display_name || "").join(" ");
  const topic = String(work?.primary_topic?.display_name || "");
  const keywords = (work?.keywords || []).map((x) => x?.display_name || "").join(" ");
  return `${title} ${concepts} ${topic} ${keywords}`.toLowerCase();
}

function analyzeKeywords(works, keywordList) {
  const matchedKeywordSet = new Set();
  let worksWithHit = 0;
  let totalHits = 0;

  for (const work of works || []) {
    const text = textOfWork(work);
    let workHit = false;
    for (const keyword of keywordList) {
      if (text.includes(keyword)) {
        matchedKeywordSet.add(keyword);
        totalHits += 1;
        workHit = true;
      }
    }
    if (workHit) worksWithHit += 1;
  }

  const considered = Math.max(1, works?.length || 0);
  return {
    considered_works: works?.length || 0,
    works_with_keyword_hit: worksWithHit,
    affective_ratio: Number((worksWithHit / considered).toFixed(4)),
    keyword_density: Number((totalHits / considered).toFixed(4)),
    matched_keywords: Array.from(matchedKeywordSet).sort((a, b) => a.localeCompare(b)),
  };
}

function workHasKeywordHit(work, keywordList) {
  const text = textOfWork(work);
  return keywordList.some((keyword) => text.includes(keyword));
}

function clamp01(x) {
  const n = Number(x);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

async function runPool(items, concurrency, worker) {
  const out = new Array(items.length);
  let cursor = 0;
  const workers = new Array(concurrency).fill(null).map(async () => {
    while (true) {
      const idx = cursor;
      if (idx >= items.length) break;
      cursor += 1;
      out[idx] = await worker(items[idx], idx);
    }
  });
  await Promise.all(workers);
  return out;
}

async function main() {
  const args = parseArgs(process.argv);
  const seedPath = path.resolve(args.seed);
  const outPath = path.resolve(args.out);
  const reviewedOutPath = args.reviewedOut
    ? path.resolve(args.reviewedOut)
    : path.join(path.dirname(outPath), "potential-authors.tobereviewed.json");
  const seenCachePath = path.resolve(args.seenCache);

  const seed = (await loadJson(seedPath, { researchers: [] })) || { researchers: [] };
  const seenCache = (await loadJson(seenCachePath, { authors: {} })) || { authors: {} };
  const seenAuthors = seenCache?.authors && typeof seenCache.authors === "object" ? seenCache.authors : {};
  const seenIds = new Set(Object.keys(seenAuthors).map((id) => normalizeAuthorId(id)).filter(Boolean));
  const seedResearchers = Array.isArray(seed?.researchers) ? seed.researchers : [];
  const seedAuthorIds = new Set(seedResearchers.map((x) => normalizeAuthorId(x?.openalex_author_id)).filter(Boolean));
  const seedById = new Map(
    seedResearchers.map((x) => [normalizeAuthorId(x?.openalex_author_id), x]).filter((x) => x[0])
  );
  const seedOrcidMap = new Map();
  const seedNameMap = new Map();
  for (const s of seedResearchers) {
    const oid = normalizeOrcid(s?.orcid);
    if (oid) {
      if (!seedOrcidMap.has(oid)) seedOrcidMap.set(oid, []);
      seedOrcidMap.get(oid).push({
        openalex_author_id: normalizeAuthorId(s?.openalex_author_id),
        name: String(s?.name || ""),
      });
    }
    const n = normalizeName(s?.name);
    if (n) {
      if (!seedNameMap.has(n)) seedNameMap.set(n, []);
      seedNameMap.get(n).push({
        openalex_author_id: normalizeAuthorId(s?.openalex_author_id),
        name: String(s?.name || ""),
      });
    }
  }

  console.log(`[discover] seed researchers: ${seedResearchers.length}`);
  console.log(`[discover] seen cache entries: ${seenIds.size}`);
  const candidateMap = new Map();
  let seedWorksTotal = 0;

  for (const seedResearcher of seedResearchers) {
    const seedId = normalizeAuthorId(seedResearcher?.openalex_author_id);
    if (!seedId) continue;
    console.log(`[discover] fetch seed works: ${seedResearcher.name} (${seedId})`);
    let works = [];
    try {
      works = await fetchAuthorWorks(seedId, args);
    } catch (err) {
      console.warn(`[discover] failed to fetch works for ${seedId}: ${err.message}`);
      continue;
    }
    seedWorksTotal += works.length;

    for (const work of works) {
      if (args.requireAffectiveSharedWorks && !workHasKeywordHit(work, args.keywordList)) {
        continue;
      }
      const workId = String(work?.id || "").trim();
      const workTitle = String(work?.title || "").trim();
      const publicationYear = Number(work?.publication_year || 0) || null;
      const authorships = Array.isArray(work?.authorships) ? work.authorships : [];

      for (const authorship of authorships) {
        const coauthorId = normalizeAuthorId(authorship?.author?.id || authorship?.author_id || "");
        if (!coauthorId || seedAuthorIds.has(coauthorId)) continue;
        const coauthorName = String(authorship?.author?.display_name || "").trim() || coauthorId;

        if (!candidateMap.has(coauthorId)) {
          candidateMap.set(coauthorId, {
            openalex_author_id: coauthorId,
            name: coauthorName,
            shared_work_ids: new Set(),
            shared_works_sample: [],
            shared_with_seed_ids: new Set(),
            shared_with_seed_names: new Set(),
            shared_years: [],
          });
        }
        const item = candidateMap.get(coauthorId);
        item.shared_work_ids.add(workId || `${seedId}-${workTitle}`);
        item.shared_with_seed_ids.add(seedId);
        item.shared_with_seed_names.add(String(seedResearcher?.name || seedId));
        if (item.shared_works_sample.length < 8) {
          item.shared_works_sample.push({
            id: workId || null,
            title: workTitle || null,
            publication_year: publicationYear,
            with_seed_author_id: seedId,
            with_seed_name: String(seedResearcher?.name || ""),
          });
        }
        if (publicationYear) item.shared_years.push(publicationYear);
      }
    }
  }

  let candidates = Array.from(candidateMap.values())
    .map((x) => ({
      ...x,
      shared_work_count: x.shared_work_ids.size,
      shared_with_seed_count: x.shared_with_seed_ids.size,
    }))
    .filter((x) => x.shared_work_count >= args.minSharedWorks)
    .filter((x) => !(args.excludeSeen && seenIds.has(normalizeAuthorId(x.openalex_author_id))))
    .sort((a, b) => b.shared_work_count - a.shared_work_count)
    .slice(0, args.maxCandidates);

  console.log(`[discover] coauthor candidates: ${candidates.length}`);

  const enriched = await runPool(candidates, args.candidateConcurrency, async (candidate, idx) => {
    process.stdout.write(`[discover] profiling candidate ${idx + 1}/${candidates.length}\r`);
    const authorId = candidate.openalex_author_id;
    try {
      const { profile, works } = await fetchCandidateProfile(authorId, args);
      const keywordStats = analyzeKeywords(works, args.keywordList);
      const affectiveWorkIds = new Set(
        (works || [])
          .filter((w) => workHasKeywordHit(w, args.keywordList))
          .map((w) => String(w?.id || "").trim())
          .filter(Boolean)
      );
      const sharedAffectiveWorkIds = new Set(
        Array.from(candidate.shared_work_ids || [])
          .map((id) => String(id || "").trim())
          .filter(Boolean)
      );
      const additionalAffectiveWorkIds = new Set(
        Array.from(affectiveWorkIds).filter((id) => !sharedAffectiveWorkIds.has(id))
      );
      const currentYear = new Date().getUTCFullYear();
      const recent3yAdditionalAffectiveWorksCount = (works || []).filter((w) => {
        const id = String(w?.id || "").trim();
        if (!id || !additionalAffectiveWorkIds.has(id)) return false;
        const y = Number(w?.publication_year || 0);
        return Number.isFinite(y) && y >= currentYear - 2;
      }).length;
      const coauthorStrength = clamp01(candidate.shared_work_count / 6);
      const recentSharedCount = candidate.shared_years.filter(
        (y) => y >= new Date().getUTCFullYear() - 1
      ).length;
      const recentSharedRatio = clamp01(recentSharedCount / Math.max(1, candidate.shared_work_count));
      const score =
        0.45 * keywordStats.affective_ratio +
        0.25 * coauthorStrength +
        0.2 * clamp01(keywordStats.keyword_density / Math.max(1, args.keywordList.length * 0.6)) +
        0.1 * recentSharedRatio;

      const orcid = normalizeOrcid(profile?.orcid);
      const normalizedName = normalizeName(profile?.display_name || candidate.name);
      const duplicateByOrcid = orcid && seedOrcidMap.has(orcid) ? seedOrcidMap.get(orcid) : [];
      const duplicateByName = normalizedName && seedNameMap.has(normalizedName) ? seedNameMap.get(normalizedName) : [];

      return {
        name: String(profile?.display_name || candidate.name || authorId),
        openalex_author_id: authorId,
        openalex_author_url: `https://openalex.org/${authorId}`,
        orcid: orcid || null,
        metrics: {
          works_count: Number(profile?.works_count || 0),
          cited_by_count: Number(profile?.cited_by_count || 0),
          h_index: Number(profile?.summary_stats?.h_index || 0),
        },
        affiliations: {
          institutions: (profile?.last_known_institutions || []).map((i) => i?.display_name).filter(Boolean),
          countries: (profile?.last_known_institutions || []).map((i) => i?.country_code).filter(Boolean),
        },
        linkage: {
          shared_work_count: candidate.shared_work_count,
          shared_with_seed_count: candidate.shared_with_seed_count,
          shared_with_seed_authors: Array.from(candidate.shared_with_seed_names).sort((a, b) =>
            a.localeCompare(b)
          ),
          shared_works_sample: candidate.shared_works_sample,
        },
        keyword_stats: keywordStats,
        quality_gates: {
          additional_affective_works_count: additionalAffectiveWorkIds.size,
          recent_3y_additional_affective_works_count: recent3yAdditionalAffectiveWorksCount,
        },
        score: Number(score.toFixed(4)),
        duplicate_signals: {
          orcid_matches_seed: duplicateByOrcid || [],
          name_matches_seed: duplicateByName || [],
          possible_unmerged_author:
            (duplicateByOrcid && duplicateByOrcid.length > 0) ||
            (duplicateByName && duplicateByName.length > 0),
        },
        review_status: "pending",
        recommendation:
          score >= 0.6 ? "strong_candidate" : score >= 0.45 ? "candidate" : "weak_candidate",
      };
    } catch (err) {
      return {
        name: candidate.name,
        openalex_author_id: authorId,
        openalex_author_url: `https://openalex.org/${authorId}`,
        error: String(err?.message || err),
        linkage: {
          shared_work_count: candidate.shared_work_count,
          shared_with_seed_count: candidate.shared_with_seed_count,
          shared_with_seed_authors: Array.from(candidate.shared_with_seed_names).sort((a, b) =>
            a.localeCompare(b)
          ),
          shared_works_sample: candidate.shared_works_sample,
        },
        score: 0,
        review_status: "pending",
        recommendation: "unknown",
      };
    }
  });
  process.stdout.write("\n");

  const valid = enriched
    .filter((x) => Number(x?.score || 0) >= args.minScore)
    .filter((x) =>
      !args.requireRecentAdditionalAffectiveWorks
        ? true
        : Number(x?.quality_gates?.recent_3y_additional_affective_works_count || 0) > 0
    )
    .sort((a, b) => Number(b?.score || 0) - Number(a?.score || 0));

  if (args.markSeen) {
    const now = new Date().toISOString();
    for (const item of enriched) {
      const authorId = normalizeAuthorId(item?.openalex_author_id);
      if (!authorId) continue;
      const prev = seenAuthors[authorId] || {};
      const timesSeen = Number(prev.times_seen || 0) + 1;
      seenAuthors[authorId] = {
        openalex_author_id: authorId,
        name: String(item?.name || prev.name || authorId),
        first_seen_at: prev.first_seen_at || now,
        last_seen_at: now,
        times_seen: timesSeen,
        last_score: Number(item?.score || 0),
        last_recommendation: String(item?.recommendation || "unknown"),
        last_error: item?.error ? String(item.error) : null,
      };
    }
    await saveJson(seenCachePath, { generated_at: new Date().toISOString(), authors: seenAuthors });
  }

  const output = {
    generated_at: new Date().toISOString(),
    source: {
      seed_path: seedPath,
      openalex_base: OPENALEX_BASE,
    },
    config: {
      years: args.years,
      per_page: args.perPage,
      max_works_per_seed: args.maxWorksPerSeed,
      max_candidate_works: args.maxCandidateWorks,
      candidate_concurrency: args.candidateConcurrency,
      min_score: args.minScore,
      min_shared_works: args.minSharedWorks,
      max_candidates: args.maxCandidates,
      require_affective_shared_works: args.requireAffectiveSharedWorks,
      require_recent_additional_affective_works: args.requireRecentAdditionalAffectiveWorks,
      seen_cache_path: seenCachePath,
      exclude_seen: args.excludeSeen,
      mark_seen: args.markSeen,
      keywords: args.keywordList,
    },
    summary: {
      seed_researchers: seedResearchers.length,
      seed_works_scanned: seedWorksTotal,
      coauthor_candidates: candidates.length,
      excluded_by_seen_cache: Array.from(candidateMap.keys()).filter((id) =>
        args.excludeSeen && seenIds.has(normalizeAuthorId(id))
      ).length,
      shortlisted_candidates: valid.length,
    },
    candidates: valid,
  };

  await saveJson(outPath, output);

  const toBeReviewed = valid
    .filter(
      (x) =>
        String(x?.recommendation || "").toLowerCase() === "strong_candidate" &&
        Number(x?.metrics?.cited_by_count || 0) > args.reviewedCitationThreshold
    )
    .sort((a, b) => Number(b?.metrics?.cited_by_count || 0) - Number(a?.metrics?.cited_by_count || 0));

  const reviewedOutput = {
    researchers: toBeReviewed.map((item) => ({
      name: String(item?.name || "").trim(),
      openalex_author_id: normalizeAuthorId(item?.openalex_author_id),
      orcid: normalizeOrcid(item?.orcid) || null,
      google_scholar: null,
    })),
  };
  await saveJson(reviewedOutPath, reviewedOutput);

  console.log(`[discover] output: ${outPath}`);
  console.log(`[discover] reviewed output: ${reviewedOutPath}`);
  console.log(
    `[discover] shortlisted=${output.summary.shortlisted_candidates} / coauthor_candidates=${output.summary.coauthor_candidates}`
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
