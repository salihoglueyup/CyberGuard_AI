import { useState, useEffect } from 'react';
import { List, AlertTriangle, Activity } from 'lucide-react';

const SAMPLE_PROCESSES = [
  { pid: 1842, name: 'ml_inference', cpu: 34.2, mem: 2800, status: 'running', threat: false },
  { pid: 2901, name: 'nginx', cpu: 2.1, mem: 120, status: 'running', threat: false },
  { pid: 3105, name: 'siem_collector', cpu: 8.4, mem: 512, status: 'running', threat: false },
  { pid: 4220, name: 'unknown_proc', cpu: 45.7, mem: 1100, status: 'running', threat: true },
  { pid: 5010, name: 'postgres', cpu: 5.6, mem: 950, status: 'running', threat: false },
  { pid: 6102, name: 'redis-server', cpu: 1.2, mem: 200, status: 'running', threat: false },
  { pid: 7301, name: 'crypto_miner_x', cpu: 89.3, mem: 3200, status: 'running', threat: true },
  { pid: 8001, name: 'tensorboard', cpu: 3.4, mem: 380, status: 'idle', threat: false },
];

export default function ProcessList({ processes = SAMPLE_PROCESSES, maxVisible = 6, className = '' }) {
  const sorted = [...processes].sort((a, b) => b.cpu - a.cpu).slice(0, maxVisible);

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <List size={12} style={{ color: 'var(--hud-cyan)' }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>PROCESS LIST</span>
        </div>
        {processes.some(p => p.threat) && (
          <AlertTriangle size={12} style={{ color: 'var(--hud-red)' }} className="animate-pulse" />
        )}
      </div>
      {/* Header */}
      <div className="grid grid-cols-[1fr_50px_50px_50px] gap-1 px-3 py-1 border-b"
        style={{ borderColor: 'var(--hud-border-subtle)' }}>
        {['PROCESS', 'CPU%', 'MEM', 'STATUS'].map(h => (
          <span key={h} className="font-mono text-[7px] tracking-wider" style={{ color: 'var(--hud-text-dim)' }}>{h}</span>
        ))}
      </div>
      {/* Rows */}
      {sorted.map(p => (
        <div key={p.pid} className="grid grid-cols-[1fr_50px_50px_50px] gap-1 px-3 py-1.5 border-b hover:bg-white/[0.02]"
          style={{
            borderColor: 'var(--hud-border-subtle)',
            background: p.threat ? 'rgba(255,0,60,0.04)' : undefined,
          }}>
          <div className="flex items-center gap-1.5 min-w-0">
            {p.threat && <AlertTriangle size={9} style={{ color: 'var(--hud-red)' }} />}
            <span className="font-mono text-[9px] truncate" style={{ color: p.threat ? 'var(--hud-red)' : 'var(--hud-text)' }}>
              {p.name}
            </span>
          </div>
          <span className="font-mono text-[9px]" style={{
            color: p.cpu > 80 ? 'var(--hud-red)' : p.cpu > 40 ? 'var(--hud-amber)' : 'var(--hud-text-muted)'
          }}>
            {p.cpu.toFixed(1)}
          </span>
          <span className="font-mono text-[9px]" style={{ color: 'var(--hud-text-muted)' }}>
            {p.mem >= 1024 ? `${(p.mem / 1024).toFixed(1)}G` : `${p.mem}M`}
          </span>
          <span className="font-mono text-[8px]" style={{
            color: p.status === 'running' ? 'var(--hud-emerald)' : 'var(--hud-text-dim)'
          }}>
            {p.status === 'running' ? '● RUN' : '○ IDLE'}
          </span>
        </div>
      ))}
    </div>
  );
}
