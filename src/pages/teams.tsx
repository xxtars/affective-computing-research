import type {ReactNode} from 'react';
import {useEffect, useMemo, useState} from 'react';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import Link from '@docusaurus/Link';

import teamsData from '../data/teams.json';
import {getWorksByAuthorId} from '../lib/openalex';
import styles from './index.module.css';

type TeamItem = (typeof teamsData)[number];

function useRecentPaperCounts(teams: TeamItem[]) {
  const [counts, setCounts] = useState<Record<string, number>>({});

  useEffect(() => {
    teams.forEach(async (team) => {
      if (!team.openalex_author_id || counts[team.name] !== undefined) {
        return;
      }

      try {
        const works = await getWorksByAuthorId(team.openalex_author_id, {perPage: 50});
        const currentYear = new Date().getFullYear();
        const recentCount = works.filter((w) => w.publication_year >= currentYear - 3).length;

        setCounts((prev) => ({...prev, [team.name]: recentCount}));
      } catch {
        setCounts((prev) => ({...prev, [team.name]: -1}));
      }
    });
  }, [teams, counts]);

  return counts;
}

function getUniqueCountries(teams: TeamItem[]) {
  return Array.from(new Set(teams.map((t) => t.country))).sort();
}

function getUniqueDirections(teams: TeamItem[]) {
  return Array.from(new Set(teams.flatMap((t) => t.directions))).sort();
}

function groupByInstitution(teams: TeamItem[]) {
  return teams.reduce<Record<string, TeamItem[]>>((acc, team) => {
    const key = team.institution;
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(team);
    return acc;
  }, {});
}

function TeamCard({team, recentCount}: {team: TeamItem; recentCount?: number}) {
  const researcherLink = team.openalex_author_id
    ? `/researcher?id=${encodeURIComponent(team.openalex_author_id)}`
    : `/researcher?name=${encodeURIComponent(team.name)}`;

  return (
    <article className={styles.teamCard}>
      <Heading as="h3" className={styles.teamName}>
        <Link to={researcherLink}>{team.name}</Link>
      </Heading>
      <p className={styles.teamInstitution}>{team.institution}</p>

      <div className={styles.teamMeta}>
        <div className={styles.teamMetaRow}>
          <span className={styles.metaKey}>Country</span>
          <span>{team.country}</span>
        </div>
        <div className={styles.teamMetaRow}>
          <span className={styles.metaKey}>Directions</span>
          <div className={styles.directionTags}>
            {team.directions.map((dir) => (
              <span key={dir} className={styles.directionTag}>
                {dir}
              </span>
            ))}
          </div>
        </div>
        <div className={styles.teamMetaRow}>
          <span className={styles.metaKey}>Recent papers (3y)</span>
          <span>
            {recentCount === undefined
              ? 'Loading...'
              : recentCount >= 0
                ? `${recentCount}`
                : 'Unavailable'}
          </span>
        </div>
      </div>

      <div className={styles.teamPapers}>
        <a href={team.homepage} className={styles.paperLink} target="_blank" rel="noreferrer">
          Homepage
        </a>
        <a href={team.google_scholar} className={styles.paperLink} target="_blank" rel="noreferrer">
          Google Scholar
        </a>
      </div>
    </article>
  );
}

export default function Teams(): ReactNode {
  const [countryFilter, setCountryFilter] = useState('All');
  const [directionFilter, setDirectionFilter] = useState('All');
  const [query, setQuery] = useState('');

  const counts = useRecentPaperCounts(teamsData);
  const countries = useMemo(() => getUniqueCountries(teamsData), []);
  const directions = useMemo(() => getUniqueDirections(teamsData), []);

  const filteredTeams = useMemo(() => {
    const q = query.trim().toLowerCase();

    return teamsData.filter((team) => {
      const countryMatch = countryFilter === 'All' || team.country === countryFilter;
      const directionMatch =
        directionFilter === 'All' || team.directions.includes(directionFilter);
      const searchMatch =
        q.length === 0 ||
        team.name.toLowerCase().includes(q) ||
        team.institution.toLowerCase().includes(q);

      return countryMatch && directionMatch && searchMatch;
    });
  }, [countryFilter, directionFilter, query]);

  const groupedTeams = useMemo(() => groupByInstitution(filteredTeams), [filteredTeams]);

  return (
    <Layout title="Teams" description="Follow affective computing researchers by institution">
      <header className={styles.pageHeader}>
        <div className="container">
          <Heading as="h1" className={styles.pageTitle}>
            Teams & Researchers
          </Heading>
          <p className={styles.pageSubtitle}>
            {teamsData.length} researchers · grouped by institution · OpenAlex live paper counts
          </p>
        </div>
      </header>

      <main className="container margin-vert--lg">
        <section className="margin-bottom--lg">
          <div className={styles.filterGrid}>
            <label>
              Country
              <select value={countryFilter} onChange={(e) => setCountryFilter(e.target.value)}>
                <option value="All">All</option>
                {countries.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Research Direction
              <select
                value={directionFilter}
                onChange={(e) => setDirectionFilter(e.target.value)}>
                <option value="All">All</option>
                {directions.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            </label>
            <label className={styles.searchInputWrap}>
              Search researcher / institution
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g. Imperial, Picard"
              />
            </label>
          </div>
        </section>

        <p className="margin-bottom--md">
          Showing {filteredTeams.length} of {teamsData.length} researchers
        </p>

        {Object.entries(groupedTeams)
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([institution, members]) => (
            <section key={institution} className="margin-bottom--lg">
              <Heading as="h2">{institution}</Heading>
              <div className={styles.teamGrid}>
                {members.map((team) => (
                  <TeamCard key={team.name} team={team} recentCount={counts[team.name]} />
                ))}
              </div>
            </section>
          ))}

        {filteredTeams.length === 0 && <p>No researchers matched the current filters.</p>}
      </main>
    </Layout>
  );
}
