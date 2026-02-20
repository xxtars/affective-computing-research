import type {ReactNode} from 'react';
import {useMemo} from 'react';
import {useState} from 'react';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import profileData from '@site/data/researchers/researcher.profile.json';
import styles from './papers.module.css';

type WorkItem = {
  id: string;
  title: string;
  publication_year: number | null;
  cited_by_count: number;
  primary_source: string | null;
  source?: {display_name: string | null};
  links?: {
    openalex: string | null;
    source_openalex: string | null;
    landing_page: string | null;
  };
  analysis: {
    is_interesting: boolean;
    relevance_score: number;
  };
};

type ResearcherProfile = {
  identity: {
    name: string;
    openalex_author_id: string;
  };
  works: WorkItem[];
};

type ProfileFile = {
  researchers: ResearcherProfile[];
};

const profile = profileData as ProfileFile;

function normalizeTitle(title: string) {
  return String(title || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();
}

export default function PapersPage(): ReactNode {
  const [query, setQuery] = useState('');

  const papers = useMemo(() => {
    const byTitle = new Map<string, WorkItem & {researcherName: string; researcherId: string}>();

    for (const researcher of profile.researchers) {
      for (const work of researcher.works || []) {
        if (!work.analysis?.is_interesting) continue;
        const key = normalizeTitle(work.title);
        if (!key) continue;

        const enriched = {
          ...work,
          researcherName: researcher.identity.name,
          researcherId: researcher.identity.openalex_author_id,
        };

        const existing = byTitle.get(key);
        if (!existing) {
          byTitle.set(key, enriched);
          continue;
        }

        if ((enriched.analysis.relevance_score || 0) > (existing.analysis.relevance_score || 0)) {
          byTitle.set(key, enriched);
          continue;
        }

        if ((enriched.cited_by_count || 0) > (existing.cited_by_count || 0)) {
          byTitle.set(key, enriched);
        }
      }
    }

    return Array.from(byTitle.values()).sort((a, b) => {
      if ((b.publication_year || 0) !== (a.publication_year || 0)) {
        return (b.publication_year || 0) - (a.publication_year || 0);
      }
      return (b.analysis.relevance_score || 0) - (a.analysis.relevance_score || 0);
    });
  }, []);

  const filteredPapers = useMemo(() => {
    const keyword = query.trim().toLowerCase();
    if (!keyword) return papers;

    return papers.filter((paper) => {
      const yearText = String(paper.publication_year || '');
      return (
        paper.title.toLowerCase().includes(keyword) ||
        paper.researcherName.toLowerCase().includes(keyword) ||
        (paper.source?.display_name || paper.primary_source || '').toLowerCase().includes(keyword) ||
        yearText.includes(keyword)
      );
    });
  }, [papers, query]);

  return (
    <Layout title="Papers">
      <main className={styles.page}>
        <div className="container">
          <Heading as="h1">Papers</Heading>
          <p>Main affective-related papers from tracked researchers (deduplicated by title).</p>
          <section className={styles.searchSection}>
            <label>
              Search
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="title / researcher / venue / year"
                type="text"
              />
            </label>
          </section>
          <p className={styles.resultCount}>Papers: {filteredPapers.length}</p>

          {filteredPapers.length === 0 ? (
            <p>No papers yet. Run `npm run researcher:build` first.</p>
          ) : (
            <div className={styles.tableWrap}>
              <table>
                <thead>
                  <tr>
                    <th>Year</th>
                    <th>Title</th>
                    <th>Researcher</th>
                    <th>Venue</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredPapers.map((paper) => (
                    <tr key={paper.id}>
                      <td>{paper.publication_year || '-'}</td>
                      <td>{paper.title}</td>
                      <td>
                        <Link to={`/researchers/detail?id=${encodeURIComponent(paper.researcherId)}`}>
                          {paper.researcherName}
                        </Link>
                      </td>
                      <td>
                        <a
                          href={paper.links?.landing_page || paper.links?.openalex || '#'}
                          rel="noreferrer"
                          target="_blank">
                          {paper.source?.display_name || paper.primary_source || '-'}
                        </a>
                        {paper.links?.source_openalex && (
                          <>
                            {' '}
                            |{' '}
                            <a href={paper.links.source_openalex} rel="noreferrer" target="_blank">
                              Source(OpenAlex)
                            </a>
                          </>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </main>
    </Layout>
  );
}
