import { motion } from 'framer-motion';

/**
 * GothamPageShell — Wraps any page with consistent styling.
 */
export default function GothamPageShell({
  icon,
  title,
  subtitle,
  badge,
  headerRight,
  children,
  className = '',
  noPadding = false,
}) {
  return (
    <div className={`relative min-h-screen w-full ${className}`} style={{ background: 'var(--hud-bg)', color: 'var(--hud-text)' }}>
      {/* Page Header */}
      <div className="sticky top-0 z-30 px-6 py-4 flex items-center justify-between" style={{
        background: 'var(--hud-surface)',
        borderBottom: '1px solid var(--hud-border)',
      }}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2.5">
            {icon && <span style={{ color: 'var(--hud-cyan)' }}>{icon}</span>}
            <span className="text-base font-semibold" style={{ color: 'var(--hud-text-bright)' }}>
              {title}
            </span>
          </div>
          {badge && (
            <span className="text-[11px] font-medium px-2.5 py-0.5 rounded-md" style={{
              background: 'var(--hud-cyan-ghost)',
              color: 'var(--hud-cyan)',
              border: '1px solid var(--hud-border)',
            }}>
              {badge}
            </span>
          )}
          {subtitle && (
            <span className="text-[13px]" style={{ color: 'var(--hud-text-muted)' }}>
              {subtitle}
            </span>
          )}
        </div>
        {headerRight && <div className="flex items-center gap-2">{headerRight}</div>}
      </div>

      {/* Content */}
      <motion.div
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25, ease: 'easeOut' }}
        className={noPadding ? '' : 'p-6'}
      >
        {children}
      </motion.div>
    </div>
  );
}

/**
 * GothamCard — Reusable card component
 */
export function GothamCard({ title, icon, headerRight, children, className = '', noPad = false }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.2, ease: 'easeOut' }}
      className={`rounded-lg overflow-hidden ${className}`}
      style={{
        background: 'var(--hud-surface)',
        border: '1px solid var(--hud-border)',
        boxShadow: 'var(--hud-shadow)',
      }}
    >
      {title && (
        <div className="flex items-center justify-between px-4 py-3 border-b" style={{ borderColor: 'var(--hud-border)' }}>
          <div className="flex items-center gap-2">
            {icon && <span style={{ color: 'var(--hud-cyan)' }}>{icon}</span>}
            <span className="text-[13px] font-medium" style={{ color: 'var(--hud-text)' }}>
              {title}
            </span>
          </div>
          {headerRight}
        </div>
      )}
      <div className={noPad ? '' : 'p-4'}>
        {children}
      </div>
    </motion.div>
  );
}

/**
 * GothamStat — Stat chip for page headers
 */
export function GothamStat({ label, value, color = 'var(--hud-cyan)', icon, trend }) {
  return (
    <div className="px-3 py-2.5 rounded-lg" style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
    }}>
      <div className="flex items-center gap-1.5 mb-1">
        {icon && <span style={{ color }}>{icon}</span>}
        <span className="text-[11px] font-medium uppercase tracking-wide" style={{ color: 'var(--hud-text-dim)' }}>
          {label}
        </span>
      </div>
      <div className="flex items-end gap-1.5">
        <span className="font-mono text-xl font-bold" style={{ color }}>{value}</span>
        {trend && (
          <span className="text-[11px] font-medium mb-0.5" style={{
            color: trend > 0 ? 'var(--hud-red)' : 'var(--hud-emerald)',
          }}>
            {trend > 0 ? '▲' : '▼'}{Math.abs(trend)}%
          </span>
        )}
      </div>
    </div>
  );
}

/**
 * GothamTable — Styled table for HUD pages
 */
export function GothamTable({ columns, data, onRowClick, className = '' }) {
  return (
    <div className={`overflow-auto rounded-lg ${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
    }}>
      <table className="w-full">
        <thead>
          <tr style={{ borderBottom: '1px solid var(--hud-border)' }}>
            {columns.map((col, i) => (
              <th key={i} className="px-4 py-2.5 text-left font-mono text-[8px] tracking-widest uppercase"
                style={{ color: 'var(--hud-text-dim)' }}>
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, ri) => (
            <tr
              key={ri}
              className="transition-colors hover:bg-white/[0.02] cursor-pointer"
              style={{ borderBottom: '1px solid var(--hud-border-subtle)' }}
              onClick={() => onRowClick?.(row, ri)}
            >
              {columns.map((col, ci) => (
                <td key={ci} className="px-4 py-2 font-mono text-[11px]"
                  style={{ color: 'var(--hud-text)' }}>
                  {col.render ? col.render(row[col.key], row) : row[col.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * GothamBadge — Status/severity badge
 */
export function GothamBadge({ text, color = 'var(--hud-cyan)' }) {
  return (
    <span className="font-mono text-[8px] tracking-wider px-1.5 py-0.5 rounded" style={{
      color,
      background: `${color}15`,
      border: `1px solid ${color}30`,
    }}>
      {text}
    </span>
  );
}

/**
 * GothamTab — Tab bar for page sections
 */
export function GothamTabs({ tabs, active, onChange }) {
  return (
    <div className="flex gap-0.5 p-1 rounded-lg" style={{
      background: 'rgba(56,189,248,0.03)',
      border: '1px solid var(--hud-border)',
    }}>
      {tabs.map(t => (
        <button
          key={t.key}
          onClick={() => onChange(t.key)}
          className="px-4 py-1.5 rounded-md font-mono text-[10px] tracking-wider transition-all"
          style={{
            background: active === t.key ? 'rgba(56,189,248,0.1)' : 'transparent',
            color: active === t.key ? 'var(--hud-cyan)' : 'var(--hud-text-muted)',
            border: active === t.key ? '1px solid rgba(56,189,248,0.2)' : '1px solid transparent',
          }}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}
