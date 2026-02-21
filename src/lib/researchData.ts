import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';

export function useResearchDataBaseUrl() {
  const {siteConfig} = useDocusaurusContext();
  const localBaseUrl = useBaseUrl('/');
  const fromConfig = String(siteConfig.customFields?.researchDataBaseUrl || '').trim();
  if (fromConfig) return fromConfig.replace(/\/+$/, '');
  return localBaseUrl.replace(/\/+$/, '');
}

export function buildResearchDataUrl(base: string, relativePath: string) {
  const rel = String(relativePath || '').replace(/^\/+/, '');
  if (!rel) return base;
  return `${base}/${rel}`;
}
