import { useState, useEffect, useRef } from 'react';
import { Radio } from 'lucide-react';

const SAMPLE_FEEDS = [
  { id: 1, severity: 'critical', source: 'MITRE ATT&CK', text: 'APT29 — yeni lateral movement teknigi tespit edildi', ts: '12s' },
  { id: 2, severity: 'high', source: 'AlienVault OTX', text: 'Ransomware C2 IP: 185.220.101.xx aktif', ts: '34s' },
  { id: 3, severity: 'medium', source: 'VirusTotal', text: 'Yeni polimorfik malware imzasi: TR-MAL-0x4F2', ts: '1d' },
  { id: 4, severity: 'low', source: 'AbuseIPDB', text: 'SSH brute-force cluster: 103.xx.xx.0/24', ts: '2d' },
  { id: 5, severity: 'critical', source: 'CISA', text: 'CVE-2025-21298 — Windows OLE RCE, aktif exploit', ts: '5d' },
  { id: 6, severity: 'high', source: 'Shodan', text: 'Exposed RDP portlari: 14 yeni TR IP tespit', ts: '8d' },
];

const SEV_COLORS = { critical: '#ff003c', high: '#ff6a00', medium: '#ffc400', low: '#00e676' };

export default function ThreatFeed({ feeds = SAMPLE_FEEDS, maxVisible = 5, className = '' }) {
  const [items, setItems] = useState(feeds.slice(0, maxVisible));
  const containerRef = useRef(null);

  // Rotate items every 6s
  useEffect(() => {
    if (feeds.length <= maxVisible) return;
    let idx = maxVisible;
    const iv = setInterval(() => {
      setItems(prev => {
        const next = [...prev.slice(1), feeds[idx % feeds.length]];
        idx++;
        return next;
      });
    }, 6000);
    return () => clearInterval(iv);
  }, [feeds, maxVisible]);

  return (
    <div className={`hud-panel ${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <Radio size={12} style={{ color: 'var(--hud-red)' }} className="animate-pulse" />
        <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>
          THREAT INTELLIGENCE FEED
        </span>
      </div>
      <div ref={containerRef} className="divide-y" style={{ borderColor: 'var(--hud-border-subtle)' }}>
        {items.map((item, i) => (
          <div key={item.id + '-' + i} className="flex items-start gap-2 px-3 py-2 hover:bg-white/[0.02] transition-colors">
            <div className="w-1.5 h-1.5 rounded-full mt-1.5 flex-shrink-0"
              style={{ background: SEV_COLORS[item.severity] || '#00e5ff' }} />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-mono text-[8px] px-1 py-0.5 rounded"
                  style={{
                    color: SEV_COLORS[item.severity],
                    background: `${SEV_COLORS[item.severity]}15`,
                    border: `1px solid ${SEV_COLORS[item.severity]}30`,
                  }}>
                  {item.source}
                </span>
                <span className="font-mono text-[8px]" style={{ color: 'var(--hud-text-dim)' }}>{item.ts}</span>
              </div>
              <div className="font-mono text-[10px] mt-1 leading-snug" style={{ color: 'var(--hud-text)' }}>
                {item.text}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
