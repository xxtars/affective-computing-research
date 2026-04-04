import type {ReactNode} from 'react';
import {useEffect, useMemo, useState} from 'react';
import BrowserOnly from '@docusaurus/BrowserOnly';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import {buildResearchDataUrl, useResearchDataBaseUrl} from '../lib/researchData';
import styles from './network-lab.module.css';

type NetworkWindow = 'all' | 'recent_5y' | 'recent_3y' | 'recent_1y';

type ResearcherIndexRecord = {
  profile_path?: string | null;
};

type ResearchersIndexFile = {
  researchers?: ResearcherIndexRecord[];
};

type ResearcherProfileWork = {
  title?: string | null;
  publication_year?: number | null;
  analysis?: {
    is_interesting?: boolean | null;
  };
};

type ResearcherProfileLite = {
  identity?: {
    name?: string | null;
    openalex_author_id?: string | null;
  };
  topic_summary?: {
    top_research_directions?: Array<{name?: string | null}>;
  };
  works?: ResearcherProfileWork[];
};

type CoauthorNode = {
  id: string;
  name: string;
  degree: number;
  paperCount: number;
  topDirection?: string;
};

type CoauthorEdge = {
  source: string;
  target: string;
  weight: number;
};

type CoauthorGraphData = {
  nodes: CoauthorNode[];
  edges: CoauthorEdge[];
  paperCount: number;
};

function normalizeTitle(title: string | null | undefined) {
  return String(title || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

function getWindowMinYear(windowType: NetworkWindow, currentYear: number) {
  if (windowType === 'recent_1y') return currentYear;
  if (windowType === 'recent_3y') return currentYear - 2;
  if (windowType === 'recent_5y') return currentYear - 4;
  return null;
}

function buildCoauthorGraph(
  profiles: ResearcherProfileLite[],
  windowType: NetworkWindow,
  minEdgeWeight: number,
) {
  const currentYear = new Date().getFullYear();
  const minYear = getWindowMinYear(windowType, currentYear);
  const paperAuthors = new Map<string, Set<string>>();
  const paperCounts = new Map<string, number>();
  const nameById = new Map<string, string>();
  const directionById = new Map<string, string>();

  for (const profile of profiles || []) {
    const id = String(profile?.identity?.openalex_author_id || '').trim();
    const name = String(profile?.identity?.name || '').trim();
    if (!id || !name) continue;
    nameById.set(id, name);
    paperCounts.set(id, 0);
    directionById.set(
      id,
      String(profile?.topic_summary?.top_research_directions?.[0]?.name || 'Uncategorized').trim() ||
        'Uncategorized',
    );

    for (const work of profile.works || []) {
      if (!work?.analysis?.is_interesting) continue;
      const year = Number(work.publication_year || 0);
      if (minYear != null && (!Number.isFinite(year) || year < minYear)) continue;
      const key = normalizeTitle(work.title);
      if (!key) continue;
      if (!paperAuthors.has(key)) paperAuthors.set(key, new Set());
      paperAuthors.get(key)!.add(id);
      paperCounts.set(id, (paperCounts.get(id) || 0) + 1);
    }
  }

  const edgeCounter = new Map<string, number>();
  const degreeCounter = new Map<string, number>();

  for (const authorsSet of paperAuthors.values()) {
    const authors = Array.from(authorsSet).sort((a, b) => a.localeCompare(b));
    if (authors.length < 2) continue;
    for (let i = 0; i < authors.length; i += 1) {
      for (let j = i + 1; j < authors.length; j += 1) {
        const key = `${authors[i]}|||${authors[j]}`;
        edgeCounter.set(key, (edgeCounter.get(key) || 0) + 1);
      }
    }
  }

  const edges: CoauthorEdge[] = [];
  for (const [key, weight] of edgeCounter.entries()) {
    if (weight < minEdgeWeight) continue;
    const [source, target] = key.split('|||');
    edges.push({source, target, weight});
    degreeCounter.set(source, (degreeCounter.get(source) || 0) + weight);
    degreeCounter.set(target, (degreeCounter.get(target) || 0) + weight);
  }

  const nodeIds = new Set<string>();
  edges.forEach((edge) => {
    nodeIds.add(edge.source);
    nodeIds.add(edge.target);
  });
  if (edges.length === 0) {
    for (const [id, count] of paperCounts.entries()) {
      if (count > 0) nodeIds.add(id);
    }
  }

  const nodes: CoauthorNode[] = Array.from(nodeIds)
    .map((id) => ({
      id,
      name: nameById.get(id) || id,
      degree: degreeCounter.get(id) || 0,
      paperCount: paperCounts.get(id) || 0,
      topDirection: directionById.get(id) || 'Uncategorized',
    }))
    .sort((a, b) => {
      if (b.degree !== a.degree) return b.degree - a.degree;
      if (b.paperCount !== a.paperCount) return b.paperCount - a.paperCount;
      return a.name.localeCompare(b.name, 'en', {sensitivity: 'base'});
    });

  return {nodes, edges, paperCount: paperAuthors.size} satisfies CoauthorGraphData;
}

function buildEgoGraph(graph: CoauthorGraphData, centerId: string) {
  const nodeMap = new Map(graph.nodes.map((n) => [n.id, n]));
  const neighborSet = new Set<string>([centerId]);
  for (const edge of graph.edges) {
    if (edge.source === centerId) neighborSet.add(edge.target);
    else if (edge.target === centerId) neighborSet.add(edge.source);
  }
  const nodes = Array.from(neighborSet)
    .map((id) => nodeMap.get(id))
    .filter(Boolean) as CoauthorNode[];
  const edges = graph.edges.filter((edge) => neighborSet.has(edge.source) && neighborSet.has(edge.target));
  return {nodes, edges};
}

function splitNameParts(fullName: string) {
  const parts = String(fullName || '')
    .trim()
    .split(/\s+/)
    .filter(Boolean);
  if (parts.length === 0) return {familyName: '', givenName: ''};
  if (parts.length === 1) return {familyName: parts[0], givenName: ''};
  return {
    familyName: parts[parts.length - 1],
    givenName: parts.slice(0, -1).join(' '),
  };
}

export default function NetworkLabPage(): ReactNode {
  const dataBaseUrl = useResearchDataBaseUrl();
  const [networkView, setNetworkView] = useState<'ego' | 'cluster'>('ego');
  const [coauthorWindow, setCoauthorWindow] = useState<NetworkWindow>('recent_5y');
  const [coauthorMinEdge, setCoauthorMinEdge] = useState(2);
  const [egoCenterId, setEgoCenterId] = useState('');
  const [loading, setLoading] = useState(true);
  const [researcherProfiles, setResearcherProfiles] = useState<ResearcherProfileLite[]>([]);

  useEffect(() => {
    let disposed = false;
    async function load() {
      setLoading(true);
      try {
        const researcherIndexUrl = buildResearchDataUrl(dataBaseUrl, 'data/researchers/researchers.index.json');
        const rIdxRes = await fetch(researcherIndexUrl);
        if (!rIdxRes.ok) throw new Error(`Failed to load researcher index: ${rIdxRes.status}`);
        const indexJson = (await rIdxRes.json()) as ResearchersIndexFile;
        const records = indexJson.researchers || [];
        const loaded = await Promise.all(
          records.map(async (record) => {
            const rel = String(record.profile_path || '').replace(/^\/+/, '');
            if (!rel) return null;
            const profileUrl = buildResearchDataUrl(dataBaseUrl, rel);
            const res = await fetch(profileUrl);
            if (!res.ok) return null;
            return (await res.json()) as ResearcherProfileLite;
          }),
        );
        if (!disposed) setResearcherProfiles(loaded.filter(Boolean) as ResearcherProfileLite[]);
      } catch (err) {
        console.error(err);
        if (!disposed) setResearcherProfiles([]);
      } finally {
        if (!disposed) setLoading(false);
      }
    }
    load();
    return () => {
      disposed = true;
    };
  }, [dataBaseUrl]);

  const coauthorGraph = useMemo(
    () => buildCoauthorGraph(researcherProfiles, coauthorWindow, coauthorMinEdge),
    [researcherProfiles, coauthorWindow, coauthorMinEdge],
  );

  useEffect(() => {
    if (coauthorGraph.nodes.length === 0) {
      if (egoCenterId) setEgoCenterId('');
      return;
    }
    const exists = coauthorGraph.nodes.some((n) => n.id === egoCenterId);
    if (!exists) setEgoCenterId(coauthorGraph.nodes[0].id);
  }, [coauthorGraph.nodes, egoCenterId]);

  const egoGraph = useMemo(
    () => (egoCenterId ? buildEgoGraph(coauthorGraph, egoCenterId) : {nodes: [], edges: []}),
    [coauthorGraph, egoCenterId],
  );
  const networkNodesForRender = networkView === 'ego' ? egoGraph.nodes : coauthorGraph.nodes;
  const networkEdgesForRender = networkView === 'ego' ? egoGraph.edges : coauthorGraph.edges;
  const centerResearcherOptions = useMemo(
    () =>
      [...coauthorGraph.nodes].sort((a, b) => {
        const aName = splitNameParts(a.name);
        const bName = splitNameParts(b.name);
        const familyCmp = aName.familyName.localeCompare(bName.familyName, 'en', {sensitivity: 'base'});
        if (familyCmp !== 0) return familyCmp;
        const givenCmp = aName.givenName.localeCompare(bName.givenName, 'en', {sensitivity: 'base'});
        if (givenCmp !== 0) return givenCmp;
        return a.name.localeCompare(b.name, 'en', {sensitivity: 'base'});
      }),
    [coauthorGraph.nodes],
  );

  return (
    <Layout title="Network Lab">
      <main className={styles.page}>
        <div className="container">
          <Heading as="h1">Network Lab</Heading>
          <p className={styles.note}>
            Experimental co-authorship visualization. Not part of release page behavior.
          </p>

          <section className={styles.panel}>
            <Heading as="h2" className={styles.panelTitle}>Co-authorship Network</Heading>
            <p className={styles.panelDesc}>
              Nodes are tracked researchers; edges connect researchers with shared affective-related papers (deduplicated by title).
            </p>

            <div className={styles.controls}>
              <div>
                <label className={styles.controlLabel}>View</label>
                <select
                  className={styles.controlSelect}
                  value={networkView}
                  onChange={(e) => setNetworkView(e.target.value as 'ego' | 'cluster')}
                >
                  <option value="ego">Ego Network</option>
                  <option value="cluster">Full Force Graph</option>
                </select>
              </div>
              <div>
                <label className={styles.controlLabel}>Time Window</label>
                <select
                  className={styles.controlSelect}
                  value={coauthorWindow}
                  onChange={(e) => setCoauthorWindow(e.target.value as NetworkWindow)}
                >
                  <option value="all">All time</option>
                  <option value="recent_5y">Recent 5 years</option>
                  <option value="recent_3y">Recent 3 years</option>
                  <option value="recent_1y">Recent 1 year</option>
                </select>
              </div>
              <div>
                <label className={styles.controlLabel}>Min Shared Papers</label>
                <select
                  className={styles.controlSelect}
                  value={String(coauthorMinEdge)}
                  onChange={(e) => setCoauthorMinEdge(Number(e.target.value))}
                >
                  <option value="1">1+ papers</option>
                  <option value="2">2+ papers</option>
                  <option value="3">3+ papers</option>
                  <option value="5">5+ papers</option>
                </select>
              </div>
              {networkView === 'ego' && (
                <div>
                  <label className={styles.controlLabel}>Center Researcher</label>
                  <select
                    className={styles.controlSelect}
                    value={egoCenterId}
                    onChange={(e) => setEgoCenterId(e.target.value)}
                  >
                    {centerResearcherOptions.map((node) => (
                      <option key={node.id} value={node.id}>
                        {node.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>

            {/* Stats bar */}
            <div className={styles.statsBar}>
              <div className={styles.statItem}>
                <span className={styles.statValue}>{networkNodesForRender.length}</span>
                <span className={styles.statLabel}>Researchers</span>
              </div>
              <div className={styles.statItem}>
                <span className={styles.statValue}>{networkEdgesForRender.length}</span>
                <span className={styles.statLabel}>Connections</span>
              </div>
              <div className={styles.statItem}>
                <span className={styles.statValue}>{coauthorGraph.paperCount}</span>
                <span className={styles.statLabel}>Papers in scope</span>
              </div>
            </div>

            {/* Graph area */}
            <div className={styles.graphWrapper}>
              {loading ? (
                <div className={styles.loadingState}>
                  <div className={styles.loadingSpinner} />
                  <span>Loading network data…</span>
                </div>
              ) : networkNodesForRender.length === 0 ? (
                <div className={styles.emptyState}>
                  No co-authorship data under current filters.
                </div>
              ) : (
                <BrowserOnly fallback={<div className={styles.loadingState}><div className={styles.loadingSpinner} /></div>}>
                  {() => {
                    const {CoauthorSigma} = require('../components/CoauthorSigma');
                    return <CoauthorSigma nodes={networkNodesForRender} edges={networkEdgesForRender} height={680} />;
                  }}
                </BrowserOnly>
              )}
            </div>
          </section>
        </div>
      </main>
    </Layout>
  );
}
