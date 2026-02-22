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
  assignments: Array<{
    publication_year?: number | null;
    l1_name?: string | null;
  }>;
};

type YearTopicPoint = {
  year: number;
  byTopic: Record<string, number>;
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

function buildYearTopicSeries(assignments: TaxonomyAxisData['assignments']) {
  const byYear = new Map<number, YearTopicPoint>();
  for (const row of assignments || []) {
    const year = Number(row.publication_year || 0);
    const topic = String(row.l1_name || '').trim();
    if (!Number.isFinite(year) || year < 1900 || !topic) continue;
    if (!byYear.has(year)) byYear.set(year, {year, byTopic: {}});
    const p = byYear.get(year)!;
    p.byTopic[topic] = (p.byTopic[topic] || 0) + 1;
  }
  return Array.from(byYear.values()).sort((a, b) => a.year - b.year);
}

function topTopics(points: YearTopicPoint[], topN: number, keyword: string) {
  const counter = new Map<string, number>();
  for (const p of points) {
    for (const [k, v] of Object.entries(p.byTopic)) counter.set(k, (counter.get(k) || 0) + v);
  }
  const all = Array.from(counter.entries()).sort((a, b) => b[1] - a[1]).map(([k]) => k);
  const filtered = keyword ? all.filter((x) => normalize(x).includes(keyword)) : all;
  return filtered.slice(0, topN);
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

export default function LandscapePage(): ReactNode {
  const dataBaseUrl = useResearchDataBaseUrl();
  const [axis, setAxis] = useState<Axis>('problem');
  const [topN, setTopN] = useState(10);
  const [query, setQuery] = useState('');
  const [hoverTopic, setHoverTopic] = useState<string | null>(null);
  const [pinnedTopic, setPinnedTopic] = useState<string | null>(null);
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
  const points = useMemo(() => buildYearTopicSeries(current.assignments || []), [current.assignments]);
  const topicList = useMemo(
    () => topTopics(points, Math.max(1, topN), normalize(query)),
    [points, topN, query],
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
            <div>
              <label className={styles.controlLabel}>Top L1 Topics</label>
              <select className={styles.controlSelect} value={topN} onChange={(e) => setTopN(Number(e.target.value))}>
                <option value={8}>8</option>
                <option value={10}>10</option>
                <option value={12}>12</option>
                <option value={15}>15</option>
              </select>
            </div>
            <div>
              <label className={styles.controlLabel}>Filter Topic</label>
              <input
                className={styles.controlInput}
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="keyword in L1 topic"
              />
            </div>
          </div>

          {loading ? (
            <p>Loading landscape snapshot...</p>
          ) : points.length === 0 || topicList.length === 0 ? (
            <p>No trend data yet.</p>
          ) : (
            <>
              <section className={styles.panel}>
                <Heading as="h2">ThemeRiver</Heading>
                <p className={styles.muted}>Time trend of L1 topics (stream graph). Thickness indicates annual volume.</p>
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
                  {topicList.map((t, i) => (
                    <button
                      type="button"
                      className={`${styles.legendItem} ${activeTopic === t ? styles.legendItemActive : ''}`}
                      key={t}
                      onMouseEnter={() => setHoverTopic(t)}
                      onMouseLeave={() => setHoverTopic(null)}
                      onClick={() => setPinnedTopic((prev) => (prev === t ? null : t))}
                      title={pinnedTopic === t ? 'Click to unpin highlight' : 'Click to pin highlight'}
                    >
                      <span className={styles.legendDot} style={{background: COLORS[i % COLORS.length]}} />
                      {t}
                    </button>
                  ))}
                </div>
              </section>
            </>
          )}
        </div>
      </main>
    </Layout>
  );
}
