import { useState, useEffect } from 'react';
import { Shield, Check, X, AlertTriangle } from 'lucide-react';

const DEFAULT_RULES = [
  { name: 'Inbound SSH', port: 22, status: 'allow', source: '10.0.0.0/8', hits: 1247 },
  { name: 'HTTP/HTTPS', port: '80,443', status: 'allow', source: '0.0.0.0/0', hits: 89432 },
  { name: 'Block Tor Exit', port: '*', status: 'block', source: 'TOR_EXIT_NODES', hits: 342 },
  { name: 'Block CN Range', port: '*', status: 'block', source: '223.0.0.0/8', hits: 1893 },
  { name: 'Allow DNS', port: 53, status: 'allow', source: '0.0.0.0/0', hits: 45201 },
  { name: 'Block RDP', port: 3389, status: 'block', source: '0.0.0.0/0', hits: 8921 },
  { name: 'IDS Alert', port: '*', status: 'alert', source: 'FLAGGED_IPS', hits: 156 },
];

export default function FirewallStatus({ rules = DEFAULT_RULES, isActive = true, className = '' }) {
  const blocked = rules.filter(r => r.status === 'block').reduce((a, r) => a + r.hits, 0);
  const alerts = rules.filter(r => r.status === 'alert').reduce((a, r) => a + r.hits, 0);

  const statusIcon = {
    allow: <Check size={10} style={{ color: 'var(--hud-emerald)' }} />,
    block: <X size={10} style={{ color: 'var(--hud-red)' }} />,
    alert: <AlertTriangle size={10} style={{ color: 'var(--hud-amber)' }} />,
  };

  const statusColor = {
    allow: 'var(--hud-emerald)',
    block: 'var(--hud-red)',
    alert: 'var(--hud-amber)',
  };

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <Shield size={12} style={{ color: isActive ? 'var(--hud-emerald)' : 'var(--hud-red)' }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>FIREWALL</span>
        </div>
        <span className="font-mono text-[8px] px-1.5 py-0.5 rounded" style={{
          background: isActive ? 'rgba(0,230,118,0.1)' : 'rgba(255,0,60,0.1)',
          color: isActive ? 'var(--hud-emerald)' : 'var(--hud-red)',
          border: `1px solid ${isActive ? 'rgba(0,230,118,0.3)' : 'rgba(255,0,60,0.3)'}`,
        }}>
          {isActive ? 'ACTIVE' : 'DISABLED'}
        </span>
      </div>
      {/* Quick stats */}
      <div className="flex gap-3 px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border-subtle)' }}>
        <div className="flex-1 text-center">
          <div className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>BLOCKED</div>
          <div className="font-mono text-sm font-bold" style={{ color: 'var(--hud-red)' }}>{formatNum(blocked)}</div>
        </div>
        <div className="flex-1 text-center">
          <div className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>ALERTS</div>
          <div className="font-mono text-sm font-bold" style={{ color: 'var(--hud-amber)' }}>{formatNum(alerts)}</div>
        </div>
        <div className="flex-1 text-center">
          <div className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>RULES</div>
          <div className="font-mono text-sm font-bold" style={{ color: 'var(--hud-cyan)' }}>{rules.length}</div>
        </div>
      </div>
      {/* Rules */}
      <div className="max-h-40 overflow-y-auto" style={{ scrollbarWidth: 'thin' }}>
        {rules.map((r, i) => (
          <div key={i} className="flex items-center gap-2 px-3 py-1.5 border-b hover:bg-white/[0.02]"
            style={{ borderColor: 'var(--hud-border-subtle)' }}>
            {statusIcon[r.status]}
            <span className="font-mono text-[9px] flex-1" style={{ color: 'var(--hud-text)' }}>{r.name}</span>
            <span className="font-mono text-[8px]" style={{ color: 'var(--hud-text-dim)' }}>:{r.port}</span>
            <span className="font-mono text-[8px]" style={{ color: statusColor[r.status] }}>{formatNum(r.hits)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatNum(n) {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`;
  return n.toString();
}
