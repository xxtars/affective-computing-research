import type {ReactNode} from 'react';
import {useEffect, useMemo, useState} from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import {buildResearchDataUrl, useResearchDataBaseUrl} from '../lib/researchData';
import styles from './landscape.module.css';

type Axis = 'problem' | 'method';

type TaxonomySummary = {
  generated_at?: string | null;
};

type TaxonomyAxisData = {
  axis: Axis;
  l1_categories?: Array<{
    name?: string | null;
    definition?: string | null;
    l1_child_categories?: Array<{
      name?: string | null;
      definition?: string | null;
    }>;
  }>;
  assignments: Array<{
    paper_id?: string | null;
    publication_year?: number | null;
    l1_name?: string | null;
    l1_child_name?: string | null;
  }>;
};

type YearTopicPoint = {
  year: number;
  byTopic: Record<string, number>;
};
type MethodGroup = {
  parent: string;
  parentDefinition: string;
  children: string[];
};
type HeatmapData = {
  rows: string[];
  cols: string[];
  matrix: number[][];
  max: number;
  methodParentByTopic: Record<string, string>;
  years: number[];
};

const EMPTY_DATA: TaxonomyAxisData = {axis: 'problem', assignments: []};
const COLORS = ['#2a9d8f', '#e76f51', '#457b9d', '#f4a261', '#8ab17d', '#7b6d8d', '#4d908e', '#f28482', '#6d597a', '#90be6d', '#577590', '#43aa8b'];

function normalize(text: string) {
  return String(text || '').trim().toLowerCase();
}

function compactDate(text: string | null | undefined) {
  if (!text) return '-';
  const m = String(text).match(/^(\d{4}-\d{2}-\d{2})/);
  return m ? m[1] : String(text);
}

function buildYearTopicSeries(assignments: TaxonomyAxisData['assignments'], axis: Axis) {
  const byYear = new Map<number, YearTopicPoint>();
  for (const row of assignments || []) {
    const year = Number(row.publication_year || 0);
    const topic =
      axis === 'method'
        ? String(row.l1_child_name || row.l1_name || '').trim()
        : String(row.l1_name || '').trim();
    if (!Number.isFinite(year) || year < 1900 || !topic) continue;
    if (!byYear.has(year)) byYear.set(year, {year, byTopic: {}});
    const p = byYear.get(year)!;
    p.byTopic[topic] = (p.byTopic[topic] || 0) + 1;
  }
  return Array.from(byYear.values()).sort((a, b) => a.year - b.year);
}

function topTopics(points: YearTopicPoint[]) {
  const counter = new Map<string, number>();
  for (const p of points) {
    for (const [k, v] of Object.entries(p.byTopic)) counter.set(k, (counter.get(k) || 0) + v);
  }
  const all = Array.from(counter.entries()).sort((a, b) => b[1] - a[1]).map(([k]) => k);
  return all;
}

function buildThemeriverPaths(points: YearTopicPoint[], topics: string[], width: number, height: number) {
  const margin = {top: 20, right: 12, bottom: 20, left: 12};
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;
  const years = points.map((p) => p.year);
  const xPos = (idx: number) => margin.left + (years.length <= 1 ? innerW / 2 : (idx / (years.length - 1)) * innerW);

  const totals = points.map((p) => topics.reduce((sum, t) => sum + Number(p.byTopic[t] || 0), 0));
  const maxTotal = Math.max(1, ...totals);
  const scale = innerH / maxTotal;
  const centerY = margin.top + innerH / 2;

  const paths: Array<{topic: string; color: string; d: string}> = [];
  const stacks = topics.map(() => ({upper: [] as Array<{x: number; y: number}>, lower: [] as Array<{x: number; y: number}>}));

  points.forEach((p, i) => {
    let baseline = -totals[i] / 2;
    topics.forEach((t, ti) => {
      const v = Number(p.byTopic[t] || 0);
      const y0 = centerY + baseline * scale;
      const y1 = centerY + (baseline + v) * scale;
      stacks[ti].upper.push({x: xPos(i), y: y0});
      stacks[ti].lower.push({x: xPos(i), y: y1});
      baseline += v;
    });
  });

  topics.forEach((topic, ti) => {
    const color = COLORS[ti % COLORS.length];
    const up = stacks[ti].upper;
    const lo = [...stacks[ti].lower].reverse();
    if (up.length === 0) return;
    const start = `M ${up[0].x.toFixed(2)} ${up[0].y.toFixed(2)}`;
    const lineUp = up.slice(1).map((p) => `L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ');
    const lineLo = lo.map((p) => `L ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(' ');
    paths.push({topic, color, d: `${start} ${lineUp} ${lineLo} Z`});
  });

  return {paths, years};
}

function buildHeatmapData(
  problemAssignments: TaxonomyAxisData['assignments'],
  methodAssignments: TaxonomyAxisData['assignments'],
  problemOrder: string[],
  methodOrder: string[],
  selectedYear?: number | null,
): HeatmapData {
  const problemByPaper = new Map<string, Set<string>>();
  const methodByPaper = new Map<string, Set<string>>();
  const methodParentByTopic: Record<string, string> = {};
  const yearSet = new Set<number>();

  for (const row of problemAssignments || []) {
    const paperId = String(row.paper_id || '').trim();
    const topic = String(row.l1_name || '').trim();
    const y = Number(row.publication_year || 0);
    if (Number.isFinite(y) && y >= 1900) yearSet.add(y);
    if (selectedYear && y !== selectedYear) continue;
    if (!paperId || !topic) continue;
    if (!problemByPaper.has(paperId)) problemByPaper.set(paperId, new Set());
    problemByPaper.get(paperId)!.add(topic);
  }

  for (const row of methodAssignments || []) {
    const paperId = String(row.paper_id || '').trim();
    const topic = String(row.l1_child_name || row.l1_name || '').trim();
    const parent = String(row.l1_name || '').trim();
    const y = Number(row.publication_year || 0);
    if (Number.isFinite(y) && y >= 1900) yearSet.add(y);
    if (selectedYear && y !== selectedYear) continue;
    if (!paperId || !topic) continue;
    if (!methodByPaper.has(paperId)) methodByPaper.set(paperId, new Set());
    methodByPaper.get(paperId)!.add(topic);
    if (parent) methodParentByTopic[topic] = parent;
  }

  const rowSet = new Set<string>();
  const colSet = new Set<string>();
  const count = new Map<string, number>();

  for (const [paperId, problemTopics] of problemByPaper.entries()) {
    const methodTopics = methodByPaper.get(paperId);
    if (!methodTopics || methodTopics.size === 0) continue;
    for (const p of problemTopics) {
      rowSet.add(p);
      for (const m of methodTopics) {
        colSet.add(m);
        const key = `${p}|||${m}`;
        count.set(key, (count.get(key) || 0) + 1);
      }
    }
  }

  // Keep full axes visible even if a year has no observations.
  const rows = problemOrder.length > 0
    ? [...problemOrder]
    : Array.from(rowSet).sort((a, b) => a.localeCompare(b));
  const cols = methodOrder.length > 0
    ? [...methodOrder]
    : Array.from(colSet).sort((a, b) => a.localeCompare(b));

  const matrix = rows.map((r) => cols.map((c) => count.get(`${r}|||${c}`) || 0));
  const max = Math.max(0, ...matrix.flat());
  const years = Array.from(yearSet).sort((a, b) => a - b);
  return {rows, cols, matrix, max, methodParentByTopic, years};
}

function heatColor(value: number, max: number) {
  if (!max || value <= 0) return 'rgba(87,117,144,0.08)';
  const t = value / max;
  const alpha = 0.16 + t * 0.74;
  return `rgba(42,157,143,${alpha.toFixed(3)})`;
}

export default function LandscapePage(): ReactNode {
  const dataBaseUrl = useResearchDataBaseUrl();
  const [axis, setAxis] = useState<Axis>('problem');
  const [hoverTopic, setHoverTopic] = useState<string | null>(null);
  const [pinnedTopic, setPinnedTopic] = useState<string | null>(null);
  const [heatYear, setHeatYear] = useState<number | null>(null);
  const [heatPlaying, setHeatPlaying] = useState(false);
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<TaxonomySummary | null>(null);
  const [problemData, setProblemData] = useState<TaxonomyAxisData>(EMPTY_DATA);
  const [methodData, setMethodData] = useState<TaxonomyAxisData>({...EMPTY_DATA, axis: 'method'});

  useEffect(() => {
    let disposed = false;
    async function loadAll() {
      setLoading(true);
      try {
        const summaryUrl = buildResearchDataUrl(dataBaseUrl, 'data/taxonomy/taxonomy.summary.json');
        const problemUrl = buildResearchDataUrl(dataBaseUrl, 'data/taxonomy/problem/taxonomy.json');
        const methodUrl = buildResearchDataUrl(dataBaseUrl, 'data/taxonomy/method/taxonomy.json');
        const [sRes, pRes, mRes] = await Promise.all([fetch(summaryUrl), fetch(problemUrl), fetch(methodUrl)]);
        if (disposed) return;
        setSummary(sRes.ok ? ((await sRes.json()) as TaxonomySummary) : null);
        setProblemData(pRes.ok ? ((await pRes.json()) as TaxonomyAxisData) : EMPTY_DATA);
        setMethodData(mRes.ok ? ((await mRes.json()) as TaxonomyAxisData) : {...EMPTY_DATA, axis: 'method'});
      } catch (err) {
        console.error(err);
        if (!disposed) {
          setSummary(null);
          setProblemData(EMPTY_DATA);
          setMethodData({...EMPTY_DATA, axis: 'method'});
        }
      } finally {
        if (!disposed) setLoading(false);
      }
    }
    loadAll();
    return () => {
      disposed = true;
    };
  }, [dataBaseUrl]);

  const current = axis === 'problem' ? problemData : methodData;
  const problemOrder = useMemo(
    () => (problemData.l1_categories || []).map((x) => String(x?.name || '').trim()).filter(Boolean),
    [problemData.l1_categories],
  );
  const methodOrder = useMemo(() => {
    const out: string[] = [];
    const seen = new Set<string>();
    for (const parent of methodData.l1_categories || []) {
      for (const child of parent.l1_child_categories || []) {
        const name = String(child?.name || '').trim();
        if (!name || seen.has(name)) continue;
        seen.add(name);
        out.push(name);
      }
    }
    return out;
  }, [methodData.l1_categories]);
  const points = useMemo(() => buildYearTopicSeries(current.assignments || [], axis), [current.assignments, axis]);
  const topicList = useMemo(() => {
    if (axis !== 'method') return topTopics(points);
    const present = new Set<string>();
    for (const p of points) {
      for (const [k, v] of Object.entries(p.byTopic)) {
        if ((v || 0) > 0) present.add(k);
      }
    }
    const ordered: string[] = [];
    const seen = new Set<string>();
    for (const parent of current.l1_categories || []) {
      for (const child of parent.l1_child_categories || []) {
        const childName = String(child?.name || '').trim();
        if (!childName || seen.has(childName)) continue;
        if (present.has(childName)) {
          ordered.push(childName);
          seen.add(childName);
        }
      }
    }
    for (const t of topTopics(points)) {
      if (!seen.has(t)) {
        ordered.push(t);
        seen.add(t);
      }
    }
    return ordered;
  }, [axis, points, current.l1_categories]);
  const methodGroups = useMemo<MethodGroup[]>(() => {
    if (axis !== 'method') return [];
    const present = new Set(topicList);
    const groups: MethodGroup[] = [];
    for (const parent of current.l1_categories || []) {
      const parentName = String(parent?.name || '').trim();
      if (!parentName) continue;
      const children = (parent.l1_child_categories || [])
        .map((child) => String(child?.name || '').trim())
        .filter((name) => name && present.has(name));
      if (children.length === 0) continue;
      groups.push({
        parent: parentName,
        parentDefinition: String(parent?.definition || '').trim(),
        children,
      });
    }
    return groups;
  }, [axis, current.l1_categories, topicList]);
  const l1Definitions = useMemo(() => {
    const map = new Map<string, string>();
    for (const item of current.l1_categories || []) {
      const name = String(item?.name || '').trim();
      if (!name) continue;
      map.set(name, String(item?.definition || '').trim());
      for (const child of item.l1_child_categories || []) {
        const childName = String(child?.name || '').trim();
        if (!childName) continue;
        map.set(childName, String(child?.definition || '').trim());
      }
    }
    return map;
  }, [current.l1_categories]);
  const topicColor = useMemo(() => {
    const map = new Map<string, string>();
    topicList.forEach((t, i) => map.set(t, COLORS[i % COLORS.length]));
    return map;
  }, [topicList]);
  const heatYears = useMemo(
    () => buildHeatmapData(problemData.assignments || [], methodData.assignments || [], problemOrder, methodOrder, null).years,
    [problemData.assignments, methodData.assignments, problemOrder, methodOrder],
  );
  useEffect(() => {
    if (heatYears.length === 0) {
      setHeatYear(null);
      setHeatPlaying(false);
      return;
    }
    const maxYear = heatYears[heatYears.length - 1];
    if (heatYear == null || !heatYears.includes(heatYear)) setHeatYear(maxYear);
  }, [heatYears, heatYear]);
  useEffect(() => {
    if (!heatPlaying || heatYears.length <= 1 || heatYear == null) return;
    const timer = window.setInterval(() => {
      setHeatYear((prev) => {
        if (prev == null) return heatYears[0];
        const idx = heatYears.indexOf(prev);
        if (idx < 0 || idx === heatYears.length - 1) return heatYears[0];
        return heatYears[idx + 1];
      });
    }, 900);
    return () => window.clearInterval(timer);
  }, [heatPlaying, heatYears, heatYear]);
  const heatmap = useMemo(
    () => buildHeatmapData(problemData.assignments || [], methodData.assignments || [], problemOrder, methodOrder, heatYear),
    [problemData.assignments, methodData.assignments, problemOrder, methodOrder, heatYear],
  );
  const river = useMemo(() => buildThemeriverPaths(points, topicList, 1200, 360), [points, topicList]);
  const activeTopic = pinnedTopic || hoverTopic;

  return (
    <Layout title="Landscape">
      <main className={styles.page}>
        <div className="container">
          <Heading as="h1">Landscape</Heading>
          <p className={styles.muted}>Generated at: {compactDate(summary?.generated_at)} · Axis: {axis}</p>

          <div className={styles.controls}>
            <div>
              <label className={styles.controlLabel}>Axis</label>
              <select className={styles.controlSelect} value={axis} onChange={(e) => setAxis(e.target.value as Axis)}>
                <option value="problem">Problem</option>
                <option value="method">Method</option>
              </select>
            </div>
          </div>

          {loading ? (
            <p>Loading landscape snapshot...</p>
          ) : points.length === 0 || topicList.length === 0 ? (
            <p>No trend data yet.</p>
          ) : (
            <>
              <section className={styles.panel}>
                <Heading as="h2">Analysis 1: ThemeRiver</Heading>
                <div className={styles.chartWrap}>
                  <svg viewBox="0 0 1200 360" className={styles.chartSvg}>
                    <rect x="0" y="0" width="1200" height="360" className={styles.chartBg} />
                    {river.paths.map((p) => (
                      <path
                        key={p.topic}
                        d={p.d}
                        fill={p.color}
                        opacity={activeTopic ? (activeTopic === p.topic ? 0.92 : 0.12) : 0.72}
                        stroke={activeTopic === p.topic ? 'rgba(0,0,0,0.35)' : 'none'}
                        strokeWidth={activeTopic === p.topic ? 1.3 : 0}
                      >
                        <title>{p.topic}</title>
                      </path>
                    ))}
                    {river.years.map((y, i) => {
                      const x = river.years.length <= 1 ? 600 : 12 + (i / (river.years.length - 1)) * (1200 - 24);
                      return (
                        <g key={`year-${y}`}>
                          <line x1={x} y1={330} x2={x} y2={338} className={styles.axisTick} />
                          <text x={x} y={352} textAnchor="middle" className={styles.axisText}>
                            {y}
                          </text>
                        </g>
                      );
                    })}
                  </svg>
                </div>
                <div className={styles.legend}>
                  {axis === 'method' ? (
                    methodGroups.map((group) => (
                      <div className={styles.legendGroup} key={`legend-${group.parent}`}>
                        <div className={styles.legendGroupTitle}>{group.parent}</div>
                        <div className={styles.legendGroupItems}>
                          {group.children.map((t) => (
                            <button
                              type="button"
                              className={`${styles.legendItem} ${activeTopic === t ? styles.legendItemActive : ''}`}
                              key={t}
                              onMouseEnter={() => setHoverTopic(t)}
                              onMouseLeave={() => setHoverTopic(null)}
                              onClick={() => setPinnedTopic((prev) => (prev === t ? null : t))}
                              title={pinnedTopic === t ? 'Click to unpin highlight' : 'Click to pin highlight'}
                            >
                              <span className={styles.legendDot} style={{background: topicColor.get(t) || COLORS[0]}} />
                              {t}
                            </button>
                          ))}
                        </div>
                      </div>
                    ))
                  ) : (
                    topicList.map((t) => (
                      <button
                        type="button"
                        className={`${styles.legendItem} ${activeTopic === t ? styles.legendItemActive : ''}`}
                        key={t}
                        onMouseEnter={() => setHoverTopic(t)}
                        onMouseLeave={() => setHoverTopic(null)}
                        onClick={() => setPinnedTopic((prev) => (prev === t ? null : t))}
                        title={pinnedTopic === t ? 'Click to unpin highlight' : 'Click to pin highlight'}
                      >
                        <span className={styles.legendDot} style={{background: topicColor.get(t) || COLORS[0]}} />
                        {t}
                      </button>
                    ))
                  )}
                </div>
                <div className={styles.definitionNote}>
                  L1 definitions below are AI-generated summaries, not field-wide consensus.
                </div>
                <div className={styles.definitionGrid}>
                  {axis === 'method' ? (
                    methodGroups.map((group) => (
                      <article className={styles.definitionCard} key={`def-group-${group.parent}`}>
                        <h3>{group.parent}</h3>
                        <p>{group.parentDefinition || 'Definition unavailable.'}</p>
                        <div className={styles.childList}>
                          {group.children.map((topic) => (
                            <div className={styles.childItem} key={`def-child-${group.parent}-${topic}`}>
                              <div className={styles.childTitle}>
                                <span className={styles.legendDot} style={{background: topicColor.get(topic) || COLORS[0]}} />
                                {topic}
                              </div>
                              <p>{l1Definitions.get(topic) || 'Definition unavailable.'}</p>
                            </div>
                          ))}
                        </div>
                      </article>
                    ))
                  ) : (
                    topicList.map((topic) => (
                      <article className={styles.definitionCard} key={`def-${topic}`}>
                        <h3>{topic}</h3>
                        <p>{l1Definitions.get(topic) || 'Definition unavailable.'}</p>
                      </article>
                    ))
                  )}
                </div>
              </section>

              <section className={styles.panel}>
                <Heading as="h2">Analysis 2: Animated Problem x Method Heatmap</Heading>
                <p className={styles.muted}>
                  Yearly co-occurrence by paper (same paper contributes to one problem topic and one method topic).
                </p>
                {heatYears.length > 0 ? (
                  <div className={styles.heatControls}>
                    <button
                      type="button"
                      className={styles.playBtn}
                      onClick={() => setHeatPlaying((v) => !v)}
                    >
                      {heatPlaying ? 'Pause' : 'Play'}
                    </button>
                    <div className={styles.yearSliderWrap}>
                      <label className={styles.controlLabel}>Year: {heatYear ?? '-'}</label>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(0, heatYears.length - 1)}
                        step={1}
                        value={Math.max(0, heatYears.indexOf(heatYear ?? heatYears[0]))}
                        onChange={(e) => {
                          const idx = Number(e.target.value || 0);
                          setHeatYear(heatYears[Math.max(0, Math.min(heatYears.length - 1, idx))]);
                          setHeatPlaying(false);
                        }}
                      />
                    </div>
                  </div>
                ) : null}
                {heatmap.rows.length === 0 || heatmap.cols.length === 0 ? (
                  <p className={styles.muted}>No axis definition yet. Add problem/method manual categories first.</p>
                ) : (
                  <div className={styles.heatWrap}>
                    <table className={styles.heatTable}>
                      <thead>
                        <tr>
                          <th className={styles.heatCorner}>Problem \\ Method</th>
                          {heatmap.cols.map((col) => (
                            <th key={`h-col-${col}`} title={heatmap.methodParentByTopic[col] ? `${col} (${heatmap.methodParentByTopic[col]})` : col}>
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {heatmap.rows.map((row, ri) => (
                          <tr key={`h-row-${row}`}>
                            <th className={styles.heatRow}>{row}</th>
                            {heatmap.cols.map((col, ci) => {
                              const v = heatmap.matrix[ri][ci];
                              return (
                                <td
                                  key={`h-cell-${row}-${col}`}
                                  style={{background: heatColor(v, heatmap.max)}}
                                  title={`${row} x ${col}: ${v}`}
                                >
                                  {v > 0 ? v : ''}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </>
          )}
        </div>
      </main>
    </Layout>
  );
}
