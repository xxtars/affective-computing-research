import type {ReactNode} from 'react';
import {useEffect, useMemo, useState} from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';

import researchersData from '../data/researchers.json';
import {
  getAuthorCandidatesByName,
  getWorksByAuthorId,
  pickBestAuthorCandidate,
  type OpenAlexAuthorCandidate,
  type OpenAlexWork,
} from '../lib/openalex';
import styles from './index.module.css';

export default function ResearcherPage(): ReactNode {
  const [works, setWorks] = useState<OpenAlexWork[]>([]);
  const [authorId, setAuthorId] = useState<string | undefined>();
  const [authorName, setAuthorName] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | undefined>();
  const [candidates, setCandidates] = useState<OpenAlexAuthorCandidate[]>([]);
  const [yearFrom, setYearFrom] = useState('');
  const [keyword, setKeyword] = useState('');

  useEffect(() => {
    if (typeof window === 'undefined') {
      return;
    }

    const params = new URLSearchParams(window.location.search);
    const idParam = params.get('id') ?? undefined;
    const nameParam = params.get('name') ?? undefined;

    async function load() {
      setLoading(true);
      setError(undefined);
      try {
        let resolvedId = idParam;
        let resolvedName = nameParam ?? '';

        if (!resolvedName && resolvedId) {
          const fromDataset = researchersData.find((r) => r.openalex_author_id === resolvedId);
          resolvedName = fromDataset?.name ?? resolvedId;
        }

        if (!resolvedId && resolvedName) {
          const foundInDataset = researchersData.find((r) => r.name === resolvedName);
          if (foundInDataset?.openalex_author_id) {
            resolvedId = foundInDataset.openalex_author_id;
          } else {
            const possible = await getAuthorCandidatesByName(resolvedName);
            setCandidates(possible);
            resolvedId = pickBestAuthorCandidate(resolvedName, possible)?.id;
          }
        }

        if (!resolvedId) {
          throw new Error('Missing researcher identifier. Please open this page from Teams.');
        }

        setAuthorId(resolvedId);
        setAuthorName(resolvedName || resolvedId);

        const fetchedWorks = await getWorksByAuthorId(resolvedId, {perPage: 50});
        setWorks(fetchedWorks);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load researcher data.');
      } finally {
        setLoading(false);
      }
    }

    load();
  }, []);

  const filteredWorks = useMemo(() => {
    const kw = keyword.trim().toLowerCase();
    const yearFloor = Number(yearFrom);

    return works.filter((work) => {
      const byYear = Number.isNaN(yearFloor) || yearFrom.length === 0 || work.publication_year >= yearFloor;
      const byKeyword = kw.length === 0 || work.title.toLowerCase().includes(kw);
      return byYear && byKeyword;
    });
  }, [works, keyword, yearFrom]);

  return (
    <Layout title="Researcher" description="Live OpenAlex paper stream">
      <header className={styles.pageHeader}>
        <div className="container">
          <Heading as="h1" className={styles.pageTitle}>
            {authorName || 'Researcher'}
          </Heading>
          <p className={styles.pageSubtitle}>Live papers from OpenAlex (latest 50 works)</p>
        </div>
      </header>

      <main className="container margin-vert--lg">
        <p>
          <Link to="/teams">← Back to Teams</Link>
        </p>
        {authorId && <p>OpenAlex ID: {authorId}</p>}

        {candidates.length > 1 && (
          <div className="alert alert--warning margin-bottom--md">
            Multiple author candidates found. If results look wrong, please add{' '}
            <code>openalex_author_id</code> in <code>researchers.json</code>.
          </div>
        )}

        {candidates.length > 1 && (
          <details className="margin-bottom--md">
            <summary>Candidate authors</summary>
            <ul>
              {candidates.map((candidate) => (
                <li key={candidate.id}>
                  {candidate.display_name} · works: {candidate.works_count} · institution:{' '}
                  {candidate.last_known_institutions?.[0]?.display_name ?? 'N/A'}
                </li>
              ))}
            </ul>
          </details>
        )}

        <section className="margin-bottom--lg">
          <div className={styles.filterGrid}>
            <label>
              Year from
              <input
                type="number"
                value={yearFrom}
                onChange={(e) => setYearFrom(e.target.value)}
                placeholder="e.g. 2021"
              />
            </label>
            <label>
              Title keyword
              <input
                type="text"
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                placeholder="e.g. emotion recognition"
              />
            </label>
          </div>
        </section>

        {loading && <p>Loading OpenAlex data...</p>}
        {error && <p>{error}</p>}

        {!loading && !error && (
          <section>
            <Heading as="h2">Works ({filteredWorks.length})</Heading>
            <div className={styles.tableWrap}>
              <table>
                <thead>
                  <tr>
                    <th>Title</th>
                    <th>Year</th>
                    <th>Venue</th>
                    <th>Co-authors</th>
                    <th>Links</th>
                    <th>Concepts</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredWorks.map((work) => {
                    const venue = work.primary_location?.source?.display_name ?? 'N/A';
                    const landing = work.primary_location?.landing_page_url;
                    const doiUrl = work.doi ? `https://doi.org/${work.doi.replace('https://doi.org/', '')}` : undefined;

                    return (
                      <tr key={work.id}>
                        <td>{work.title}</td>
                        <td>{work.publication_year}</td>
                        <td>{venue}</td>
                        <td>
                          <details>
                            <summary>{work.authorships?.length ?? 0} authors</summary>
                            <ul>
                              {work.authorships?.map((a, index) => (
                                <li key={`${work.id}-${index}`}>{a.author?.display_name ?? 'Unknown'}</li>
                              ))}
                            </ul>
                          </details>
                        </td>
                        <td>
                          {landing && (
                            <a href={landing} target="_blank" rel="noreferrer">
                              landing
                            </a>
                          )}
                          {landing && doiUrl && ' · '}
                          {doiUrl && (
                            <a href={doiUrl} target="_blank" rel="noreferrer">
                              DOI
                            </a>
                          )}
                        </td>
                        <td>
                          {(work.concepts ?? []).slice(0, 5).map((concept) => (
                            <span key={`${work.id}-${concept.display_name}`} className={styles.tagPill}>
                              {concept.display_name}
                            </span>
                          ))}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}
