import { useState, useEffect } from 'react';
import { FileKey, AlertTriangle, CheckCircle, Clock } from 'lucide-react';

const DEFAULT_CERTS = [
  { domain: '*.cyberguard.ai', issuer: "Let's Encrypt", expires: '2025-08-15', status: 'valid', days: 142 },
  { domain: 'api.cyberguard.ai', issuer: 'DigiCert', expires: '2025-03-01', status: 'expiring', days: 12 },
  { domain: 'siem.internal', issuer: 'Self-Signed', expires: '2024-12-01', status: 'expired', days: -45 },
  { domain: 'mail.cyberguard.ai', issuer: 'Sectigo', expires: '2025-11-20', status: 'valid', days: 239 },
  { domain: 'vpn.cyberguard.ai', issuer: 'GlobalSign', expires: '2025-06-10', status: 'valid', days: 76 },
];

export default function CertificateMonitor({ certs = DEFAULT_CERTS, className = '' }) {
  const statusConfig = {
    valid: { icon: <CheckCircle size={10} />, color: 'var(--hud-emerald)', label: 'VALID' },
    expiring: { icon: <Clock size={10} />, color: 'var(--hud-amber)', label: 'EXPIRING' },
    expired: { icon: <AlertTriangle size={10} />, color: 'var(--hud-red)', label: 'EXPIRED' },
  };

  const summary = {
    valid: certs.filter(c => c.status === 'valid').length,
    expiring: certs.filter(c => c.status === 'expiring').length,
    expired: certs.filter(c => c.status === 'expired').length,
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
          <FileKey size={12} style={{ color: 'var(--hud-cyan)' }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>TLS/SSL CERTS</span>
        </div>
        <div className="flex gap-2">
          {Object.entries(summary).map(([status, count]) => count > 0 && (
            <span key={status} className="font-mono text-[8px]" style={{ color: statusConfig[status].color }}>
              {count}{status[0].toUpperCase()}
            </span>
          ))}
        </div>
      </div>
      <div className="max-h-44 overflow-y-auto" style={{ scrollbarWidth: 'thin' }}>
        {certs.map((cert, i) => {
          const cfg = statusConfig[cert.status] || statusConfig.valid;
          return (
            <div key={i} className="flex items-center gap-2 px-3 py-2 border-b hover:bg-white/[0.02]"
              style={{ borderColor: 'var(--hud-border-subtle)' }}>
              <span style={{ color: cfg.color }}>{cfg.icon}</span>
              <div className="flex-1 min-w-0">
                <div className="font-mono text-[9px] truncate" style={{ color: 'var(--hud-text)' }}>{cert.domain}</div>
                <div className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>
                  {cert.issuer} · {cert.expires}
                </div>
              </div>
              <span className="font-mono text-[8px] px-1 py-0.5 rounded flex-shrink-0" style={{
                color: cfg.color,
                background: `${cfg.color}15`,
                border: `1px solid ${cfg.color}30`,
              }}>
                {cert.days > 0 ? `${cert.days}g` : cfg.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
