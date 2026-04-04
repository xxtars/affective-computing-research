import {useEffect, useMemo, useRef, useState, useCallback} from 'react';
import {UndirectedGraph} from 'graphology';
import forceAtlas2 from 'graphology-layout-forceatlas2';
import louvain from 'graphology-communities-louvain';
import {
  SigmaContainer,
  useLoadGraph,
  useSigma,
  ZoomControl,
  FullScreenControl,
  ControlsContainer,
} from '@react-sigma/core';
import '@react-sigma/core/lib/style.css';

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

// ── Gephi-style vivid community palette ───────────────────────────────────────
// 15 perceptually distinct colours, high saturation, medium lightness
const COMMUNITY_PALETTE = [
  '#e15759', '#4e79a7', '#59a14f', '#f28e2b', '#b07aa1',
  '#76b7b2', '#ff9da7', '#9c755f', '#edc948', '#bab0ac',
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf',
];

function communityColor(communityId: number): string {
  return COMMUNITY_PALETTE[communityId % COMMUNITY_PALETTE.length];
}

/** Parse hex #rrggbb → [r,g,b] */
function hexRgb(hex: string): [number, number, number] {
  const h = hex.replace('#', '');
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

/** Blend two hex colours and alpha-composite onto white → opaque hex string.
 *  Sigma's WebGL edge renderer ignores rgba alpha, so we pre-multiply. */
function blendEdge(c1: string, c2: string, alpha: number): string {
  const [r1, g1, b1] = hexRgb(c1);
  const [r2, g2, b2] = hexRgb(c2);
  // midpoint of the two endpoint colours
  const mr = (r1 + r2) >> 1, mg = (g1 + g2) >> 1, mb = (b1 + b2) >> 1;
  // alpha-composite onto white (#fff) background
  const r = Math.round(mr * alpha + 255 * (1 - alpha));
  const g = Math.round(mg * alpha + 255 * (1 - alpha));
  const b = Math.round(mb * alpha + 255 * (1 - alpha));
  return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.max(lo, Math.min(hi, v));
}

// ── graph construction ────────────────────────────────────────────────────────

function buildGraph(nodes: CoauthorNode[], edges: CoauthorEdge[]): UndirectedGraph {
  const graph = new UndirectedGraph({multi: false});
  if (nodes.length === 0) return graph;

  const maxDegree = Math.max(1, ...nodes.map((n) => n.degree));
  const maxWeight = Math.max(1, ...edges.map((e) => e.weight));

  // ① Add nodes with placeholder positions
  nodes.forEach((node, i) => {
    const t = node.degree / maxDegree;
    const size = clamp(3 + Math.sqrt(t) * 22, 3, 28);
    const r = (1 - t * 0.5) * 10;
    const angle = (i / nodes.length) * 2 * Math.PI;
    graph.addNode(node.id, {
      label: node.name,
      size,
      x: r * Math.cos(angle),
      y: r * Math.sin(angle),
      degree: node.degree,
      paperCount: node.paperCount,
      topDirection: node.topDirection || 'Uncategorized',
    });
  });

  // ② Add edges
  edges.forEach((edge, idx) => {
    if (!graph.hasNode(edge.source) || !graph.hasNode(edge.target)) return;
    if (graph.hasEdge(edge.source, edge.target) || graph.hasEdge(edge.target, edge.source)) return;
    graph.addEdge(edge.source, edge.target, {
      key: `e${idx}`,
      weight: edge.weight || 1,
      label: `${edge.weight}`,
    });
  });

  // ③ Louvain community detection (uses edge weight for modularity)
  const communities: Record<string, number> =
    graph.order > 0 && graph.size > 0
      ? louvain(graph, {
          getEdgeWeight: 'weight',
          resolution: 1,
        })
      : Object.fromEntries(graph.nodes().map((nodeId) => [nodeId, 0]));

  // Rank communities by size so the biggest gets colour index 0
  const commSize = new Map<number, number>();
  Object.values(communities).forEach((c) => commSize.set(c, (commSize.get(c) || 0) + 1));
  const commRank = new Map<number, number>();
  [...commSize.entries()].sort((a, b) => b[1] - a[1]).forEach(([c], i) => commRank.set(c, i));

  // Assign community colour to nodes
  const nodeColor = new Map<string, string>();
  graph.forEachNode((n) => {
    const cId = communities[n] ?? 0;
    const col = communityColor(commRank.get(cId) ?? cId);
    nodeColor.set(n, col);
    graph.setNodeAttribute(n, 'color', col);
    graph.setNodeAttribute(n, 'nodeColor', col);
    graph.setNodeAttribute(n, 'community', cId);
  });

  // Assign edge colour: blend of endpoint colours, very low alpha
  graph.forEachEdge((e, _attrs, src, tgt) => {
    const w = graph.getEdgeAttribute(e, 'weight') as number;
    const wt = maxWeight <= 1 ? 0 : (w - 1) / (maxWeight - 1);
    const c1 = nodeColor.get(src) || '#aaa';
    const c2 = nodeColor.get(tgt) || '#aaa';
    // Default: visible enough to see structure; hover will boost connected edges
    const alpha = +(0.2 + wt * 0.35).toFixed(2);
    const col = blendEdge(c1, c2, alpha);
    const hoverCol = blendEdge(c1, c2, 0.9);
    graph.setEdgeAttribute(e, 'color', col);
    graph.setEdgeAttribute(e, 'edgeColor', col);
    graph.setEdgeAttribute(e, 'hoverColor', hoverCol);
    graph.setEdgeAttribute(e, 'size', clamp(4.5 + wt * 15, 4.5, 21));
  });

  // ④ ForceAtlas2 layout
  if (graph.order > 1) {
    const inferred = forceAtlas2.inferSettings(graph);
    forceAtlas2.assign(graph, {
      iterations: 800,
      settings: {
        ...inferred,
        gravity: 1,                // strong centre pull — keeps outliers from flying away
        scalingRatio: 4,           // lower repulsion — lets clusters spread evenly
        slowDown: 8,               // damp oscillation for stable convergence
        strongGravityMode: true,   // gravity ∝ degree — hubs stay centred, leaves pulled in
        barnesHutOptimize: graph.order > 80,
        barnesHutTheta: 0.6,
      },
    });
  }

  return graph;
}

// ── sub-components ────────────────────────────────────────────────────────────

function LoadGraph({graph}: {graph: UndirectedGraph}) {
  const loadGraph = useLoadGraph();
  useEffect(() => { loadGraph(graph); }, [graph, loadGraph]);
  return null;
}

function FitCamera() {
  const sigma = useSigma();
  useEffect(() => { sigma.getCamera().animatedReset({duration: 500}); }, [sigma]);
  return null;
}

/** Refresh sigma on container resize (e.g. fullscreen toggle) */
function AutoResize() {
  const sigma = useSigma();
  useEffect(() => {
    const container = sigma.getContainer();
    if (!container || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(() => {
      // Give the browser a frame to finish the layout change
      requestAnimationFrame(() => {
        sigma.refresh();
        sigma.getCamera().animatedReset({duration: 300});
      });
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, [sigma]);
  return null;
}

/** Hide Sigma's built-in hover overlay (white ring + label redraw).
 *  We handle hover entirely via nodeReducer/edgeReducer + tooltip. */
function DisableBuiltinHover() {
  const sigma = useSigma();
  useEffect(() => {
    const container = sigma.getContainer();
    // Sigma creates stacked canvases; the "hovers" canvas draws the white ring.
    // The "mouse-hovernodes" webgl canvas redraws the hovered node on top.
    const layers = container.querySelectorAll('canvas');
    layers.forEach((c) => {
      const key = c.getAttribute('data-sigma-layer') ?? '';
      if (key === 'hovers' || key === 'hoverNodes') {
        c.style.display = 'none';
      }
    });
  }, [sigma]);
  return null;
}

/** Drives nodeReducer/edgeReducer for hover highlight */
function HoverController({
  hoveredNode, onHover, onUnhover,
}: {
  hoveredNode: string | null;
  onHover: (node: string) => void;
  onUnhover: () => void;
}) {
  const sigma = useSigma();

  useEffect(() => {
    const g = sigma.getGraph();

    if (hoveredNode) {
      const ego = new Set(g.neighbors(hoveredNode));
      ego.add(hoveredNode);

      sigma.setSetting('nodeReducer', (node, data) => {
        const inEgo = ego.has(node);
        return {
          ...data,
          color: inEgo ? (g.getNodeAttribute(node, 'color') as string) : '#f0f1f3',
          size: inEgo ? (g.getNodeAttribute(node, 'size') as number) * 1.4 : (g.getNodeAttribute(node, 'size') as number) * 0.4,
          label: inEgo ? (data.label as string) : '',
          zIndex: inEgo ? 1 : 0,
        };
      });
      sigma.setSetting('edgeReducer', (edge, _data) => {
        const [src, tgt] = [g.source(edge), g.target(edge)];
        const connected = src === hoveredNode || tgt === hoveredNode;
        return {
          color: connected ? (g.getEdgeAttribute(edge, 'hoverColor') as string) : '#f8f8f9',
          size: connected ? (g.getEdgeAttribute(edge, 'size') as number) : 0.1,
          zIndex: connected ? 1 : 0,
        };
      });
    } else {
      sigma.setSetting('nodeReducer', (node, data) => ({
        ...data,
        color: g.getNodeAttribute(node, 'color') as string,
        size: g.getNodeAttribute(node, 'size') as number,
      }));
      sigma.setSetting('edgeReducer', (edge, _data) => ({
        color: g.getEdgeAttribute(edge, 'color') as string,
        size: g.getEdgeAttribute(edge, 'size') as number,
      }));
    }
    sigma.refresh();
  }, [hoveredNode, sigma]);

  useEffect(() => {
    const handleEnter = ({node}: {node: string}) => onHover(node);
    const handleLeave = () => onUnhover();
    sigma.on('enterNode', handleEnter);
    sigma.on('leaveNode', handleLeave);
    return () => {
      sigma.off('enterNode', handleEnter);
      sigma.off('leaveNode', handleLeave);
    };
  }, [sigma, onHover, onUnhover]);

  return null;
}

/** Floating tooltip showing ego details */
function HoverTooltip({hoveredNode}: {hoveredNode: string | null}) {
  const sigma = useSigma();
  const tipRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = sigma.getContainer();
    const parent = container?.parentElement;
    if (!parent) return;
    parent.style.position = 'relative';

    const tip = document.createElement('div');
    tip.style.cssText = [
      'position:absolute', 'pointer-events:none', 'z-index:200',
      'top:12px', 'left:12px',                   // fixed top-left corner
      'background:rgba(255,255,255,0.95)', 'border:1px solid #e5e7eb',
      'border-radius:10px', 'box-shadow:0 4px 18px rgba(0,0,0,0.10)',
      'padding:0.55rem 0.85rem', 'font-size:0.82rem', 'line-height:1.6',
      'color:#111827', 'min-width:180px', 'max-width:260px', 'display:none',
    ].join(';');
    parent.appendChild(tip);
    tipRef.current = tip;

    return () => { tip.remove(); };
  }, [sigma]);

  useEffect(() => {
    const tip = tipRef.current;
    if (!tip) return;
    if (!hoveredNode) { tip.style.display = 'none'; return; }
    const g = sigma.getGraph();
    if (!g.hasNode(hoveredNode)) { tip.style.display = 'none'; return; }
    const a = g.getNodeAttributes(hoveredNode);

    const collabs: {name: string; w: number}[] = [];
    g.forEachNeighbor(hoveredNode, (nbr, nAttrs) => {
      const eKey = g.edge(hoveredNode, nbr) ?? g.edge(nbr, hoveredNode);
      const w = eKey ? (g.getEdgeAttribute(eKey, 'weight') as number) : 1;
      collabs.push({name: nAttrs.label as string, w});
    });
    collabs.sort((a, b) => b.w - a.w);

    const rows = collabs.slice(0, 5)
      .map((c) => `<li style="margin:0.08rem 0">${c.name}<span style="float:right;color:#9ca3af;margin-left:0.5rem">${c.w}</span></li>`)
      .join('');

    tip.innerHTML = [
      `<div style="font-weight:700;font-size:0.9rem;margin-bottom:0.2rem">${a.label}</div>`,
      `<div style="display:flex;align-items:center;gap:0.35rem;margin-bottom:0.3rem">`,
      `  <span style="width:9px;height:9px;border-radius:50%;background:${a.color};display:inline-block"></span>`,
      `  <span style="color:#6b7280;font-size:0.79rem">${a.topDirection}</span>`,
      `</div>`,
      `<div style="display:flex;gap:1rem;font-size:0.82rem;margin-bottom:${rows ? '0.4rem' : '0'}">`,
      `  <span><span style="margin-right:0.22rem">🤝</span><span style="color:#6b7280">Co-authored</span> <b>${a.degree}</b></span>`,
      `  <span><span style="margin-right:0.22rem">📄</span><span style="color:#6b7280">Papers</span> <b>${a.paperCount}</b></span>`,
      `</div>`,
      rows
        ? `<div style="font-size:0.73rem;color:#9ca3af;font-weight:600;letter-spacing:0.03em;margin-bottom:0.15rem">CO-AUTHORS</div>`
          + `<ul style="margin:0;padding-left:1rem;font-size:0.8rem;list-style:disc">${rows}</ul>`
        : '',
    ].join('');
    tip.style.display = 'block';
  }, [hoveredNode, sigma]);

  return null;
}

// ── public component ──────────────────────────────────────────────────────────

export function CoauthorSigma(props: {nodes: CoauthorNode[]; edges: CoauthorEdge[]; height?: number}) {
  const {nodes, edges, height = 700} = props;
  const graph = useMemo(() => buildGraph(nodes, edges), [nodes, edges]);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);
  const onHover = useCallback((n: string) => setHoveredNode(n), []);
  const onUnhover = useCallback(() => setHoveredNode(null), []);

  // Default reducers: read colour & size from graph attributes
  const nodeReducer = useCallback((_: string, data: Record<string, unknown>) => ({
    ...data,
    color: (data.color as string) || '#2563eb',
    size: (data.size as number) || 8,
    label: (data.label as string) || '',
  }), []);
  const edgeReducer = useCallback((_: string, data: Record<string, unknown>) => ({
    ...data,
    color: (data.color as string) || '#cbd5e1',
    size: (data.size as number) || 1.5,
  }), []);

  return (
    <div style={{height}}>
      <SigmaContainer
        style={{height: '100%', width: '100%', background: '#ffffff'}}
        settings={{
          renderEdgeLabels: false,
          renderLabels: true,
          labelRenderedSizeThreshold: 0,   // show ALL labels by default
          defaultEdgeType: 'line',
          zIndex: true,
          allowInvalidContainer: true,
          // Label style
          labelSize: 12,
          labelWeight: '500',
          labelColor: {color: '#374151'},
          labelDensity: 1,                 // render every label
          labelGridCellSize: 60,           // small grid → more labels fit
          defaultNodeColor: '#2563eb',
          defaultEdgeColor: '#cbd5e1',
          // Node border: white ring for depth (Gephi default)
          nodeReducer,
          edgeReducer,
        }}
      >
        <LoadGraph graph={graph} />
        <FitCamera />
        <AutoResize />
        <DisableBuiltinHover />
        <HoverController hoveredNode={hoveredNode} onHover={onHover} onUnhover={onUnhover} />
        <HoverTooltip hoveredNode={hoveredNode} />
        <ControlsContainer position={'bottom-right'}>
          <ZoomControl />
          <FullScreenControl />
        </ControlsContainer>
      </SigmaContainer>
    </div>
  );
}
