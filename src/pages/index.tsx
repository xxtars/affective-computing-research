import type {ReactNode} from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import teamsData from '../data/teams.json';
import directionsData from '../data/directions.json';
import researchersData from '../data/researchers.json';

import styles from './index.module.css';

function HeroSection() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={styles.hero}>
      <div className="container">
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>
          Minimal researcher follow list + live OpenAlex paper stream
        </p>
      </div>
    </header>
  );
}

function StatsSection() {
  const stats = [
    {label: 'Researchers', value: researchersData.length},
    {label: 'Team cards', value: teamsData.length},
    {label: 'Directions', value: directionsData.length},
  ];

  return (
    <section className={styles.statsSection}>
      <div className="container">
        <div className={styles.statsGrid}>
          {stats.map((stat) => (
            <div key={stat.label} className={styles.statCard}>
              <span className={styles.statValue}>{stat.value}</span>
              <span className={styles.statLabel}>{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

const entryCards = [
  {
    to: '/teams',
    title: 'Teams',
    description:
      'Filter researchers by country and direction, grouped by institution, then jump to live OpenAlex papers.',
    icon: '🏛️',
  },
  {
    to: '/directions',
    title: 'Directions',
    description:
      'Use affective-computing directions as tags for browsing and quick filtering.',
    icon: '🧭',
  },
];

function EntryCards() {
  return (
    <section className={styles.cardsSection}>
      <div className="container">
        <div className={styles.cardsGrid}>
          {entryCards.map((card) => (
            <Link key={card.to} to={card.to} className={styles.entryCard}>
              <span className={styles.cardIcon}>{card.icon}</span>
              <Heading as="h2" className={styles.cardTitle}>
                {card.title}
              </Heading>
              <p className={styles.cardDescription}>{card.description}</p>
              <span className={styles.cardLink}>Browse {card.title} →</span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout title="Home" description="Minimal follow workflow for affective-computing researchers">
      <HeroSection />
      <main>
        <StatsSection />
        <EntryCards />
      </main>
    </Layout>
  );
}
