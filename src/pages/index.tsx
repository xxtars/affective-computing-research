import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();

  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">Personal Tracking for Affective Computing Research</p>
      </div>
    </header>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout title="Awesome Affective Computing Collection">
      <HomepageHeader />
      <main className={styles.mainContent}>
        <div className="container">
          <section className={styles.section}>
            <Heading as="h2">Project Purpose</Heading>
            <p>
              This project is a personal tracking workspace for affective computing research. It focuses on
              organizing teams, directions, and representative papers in a structured way, so updates and
              comparisons are easier over time.
            </p>
          </section>

          <section className={styles.section}>
            <Heading as="h2">Quick Access</Heading>
            <div className={styles.linkGrid}>
              <Link className={styles.linkCard} to="/researchers">
                <Heading as="h3">Researchers</Heading>
                <p>View AI-assisted researcher profiles, topics, and selected papers.</p>
              </Link>
              <Link className={styles.linkCard} to="/papers">
                <Heading as="h3">Papers</Heading>
                <p>Browse main selected papers aggregated from all tracked researchers.</p>
              </Link>
            </div>
          </section>

          <section className={styles.section}>
            <Heading as="h2">What You&apos;ll Find</Heading>
            <ul className={styles.list}>
              <li>Researcher-centered snapshots with interests, institutions, and representative outputs.</li>
              <li>Topic-level organization across emotion, multimodality, speech, vision, and related areas.</li>
              <li>A maintainable personal workflow for incrementally updating tracked researchers.</li>
            </ul>
          </section>

          <section className={styles.section}>
            <Heading as="h2">Workflow</Heading>
            <ul className={styles.list}>
              <li>Add or edit researcher seeds in `data/researchers/researcher.seed.json`.</li>
              <li>Run the pipeline to fetch OpenAlex works and update AI-assisted profile output.</li>
              <li>Review results in the Researchers page and manually validate important records.</li>
            </ul>
          </section>
        </div>

        <section className={styles.disclaimer}>
          <div className="container">
            <Heading as="h3">Disclaimer</Heading>
            <p>
              Researchers are continuously being added. The current list is not a filtered shortlist, ranking, or
              complete coverage of the field.
            </p>
            <p>
              Parts of this project are AI-assisted. Metadata extraction, topic labeling, and summaries may
              contain errors or omissions. Please verify important details with official paper pages, publishers,
              and OpenAlex records.
            </p>
          </div>
        </section>
      </main>
    </Layout>
  );
}
