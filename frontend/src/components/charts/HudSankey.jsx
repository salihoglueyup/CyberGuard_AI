import { useMemo } from 'react';

const HUD_COLORS = ['var(--hud-cyan)', '#10b981', '#ffab00', '#b388ff', '#ef4444'];

export default function HudSankey({
  nodes = [],
  links = [],
  width = 700,
  height = 400,
  nodeWidth = 18,
  nodePadding = 12,
  colors = HUD_COLORS,
  title,
  className = '',
}) {
  const layout = useMemo(() => {
    if (!nodes.length || !links.length) return { lNodes: [], lLinks: [] };

    // Build node columns
    const nodeMap = {};
    nodes.forEach((n, i) => {
      nodeMap[n.id || n.name] = { ...n, idx: i, col: n.col ?? 0, inTotal: 0, outTotal: 0, inOffset: 0, outOffset: 0 };
    });
    links.forEach(l => {
      const src = nodeMap[l.source];
      const tgt = nodeMap[l.target];
      if (src) src.outTotal += l.value;
      if (tgt) tgt.inTotal += l.value;
    });

    // Columns
    const cols = {};
    Object.values(nodeMap).forEach(n => {
      if (!cols[n.col]) cols[n.col] = [];
      cols[n.col].push(n);
    });
    const numCols = Object.keys(cols).length;

    // Position nodes
    const marginX = 60;
    const colWidth = numCols > 1 ? (width - marginX * 2 - nodeWidth) / (numCols - 1) : 0;

    Object.entries(cols).forEach(([col, colNodes]) => {
      const totalVal = colNodes.reduce((s, n) => s + Math.max(n.inTotal, n.outTotal, 1), 0);
      const availH = height - 40 - (colNodes.length - 1) * nodePadding;
      let y = 20;
      colNodes.forEach(n => {
        const val = Math.max(n.inTotal, n.outTotal, 1);
        const h = Math.max(8, (val / totalVal) * availH);
        n.x = marginX + parseInt(col) * colWidth;
        n.y = y;
        n.h = h;
        y += h + nodePadding;
      });
    });

    // Position links
    const lLinks = links.map(l => {
      const src = nodeMap[l.source];
      const tgt = nodeMap[l.target];
      if (!src || !tgt) return null;
      const srcVal = Math.max(src.outTotal, 1);
      const tgtVal = Math.max(tgt.inTotal, 1);
      const srcH = (l.value / srcVal) * src.h;
      const tgtH = (l.value / tgtVal) * tgt.h;
      const sy = src.y + src.outOffset;
      const ty = tgt.y + tgt.inOffset;
      src.outOffset += srcH;
      tgt.inOffset += tgtH;
      return { ...l, sx: src.x + nodeWidth, sy: sy + srcH / 2, tx: tgt.x, ty: ty + tgtH / 2, srcH, tgtH };
    }).filter(Boolean);

    return { lNodes: Object.values(nodeMap), lLinks };
  }, [nodes, links, width, height, nodeWidth, nodePadding]);

  return (
    <div className={`hud-chart-wrapper ${className}`}>
      {title && <div className="hud-chart-title">{title}</div>}
      <svg width={width} height={height} style={{ fontFamily: 'var(--font-mono)', overflow: 'visible' }}>
        {/* Links */}
        {layout.lLinks.map((l, i) => {
          const midX = (l.sx + l.tx) / 2;
          const color = colors[i % colors.length];
          return (
            <g key={i}>
              <path
                d={`M${l.sx},${l.sy} C${midX},${l.sy} ${midX},${l.ty} ${l.tx},${l.ty}`}
                fill="none"
                stroke={color}
                strokeWidth={Math.max(2, l.srcH * 0.6)}
                strokeOpacity={0.25}
              />
              <title>{`${l.source} → ${l.target}: ${l.value}`}</title>
            </g>
          );
        })}
        {/* Nodes */}
        {layout.lNodes.map((n, i) => {
          const color = colors[i % colors.length];
          return (
            <g key={n.idx}>
              <rect
                x={n.x}
                y={n.y}
                width={nodeWidth}
                height={n.h}
                rx={3}
                fill={color}
                fillOpacity={0.6}
                stroke={color}
                strokeWidth={1}
              />
              <text
                x={n.x + nodeWidth + 5}
                y={n.y + n.h / 2 + 3}
                fill="#c8d6e5"
                fontSize={9}
              >
                {n.name || n.id}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
