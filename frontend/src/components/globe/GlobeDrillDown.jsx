import { useState, useMemo } from 'react';
import { X, Shield, MapPin, Activity, AlertTriangle, Clock, ExternalLink } from 'lucide-react';


const SEVERITY_COLORS = {
  critical: '#ef4444',
  high: '#ff6a00',
  medium: '#ffc400',
  low: '#10b981',
  info: 'var(--hud-cyan)',
};

export default function GlobeDrillDown({ country, attacks = [], onClose }) {
  const [tab, setTab] = useState('overview');

  const countryAttacks = useMemo(() => {
    if (!country?.code) return [];
    return attacks.filter(a => a.source?.country === country.code);
  }, [country, attacks]);

  const severityCounts = useMemo(() => {
    const counts = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
    countryAttacks.forEach(a => {
      if (counts[a.severity] !== undefined) counts[a.severity]++;
    });
    return counts;
  }, [countryAttacks]);

  const topAttackTypes = useMemo(() => {
    const types = {};
    countryAttacks.forEach(a => {
      const t = a.attack_type || a.type || 'Unknown';
      types[t] = (types[t] || 0) + 1;
    });
    return Object.entries(types).sort((a, b) => b[1] - a[1]).slice(0, 5);
  }, [countryAttacks]);

  if (!country) return null;

  const threatLevel = country.count > 15 ? 'KRITIK' : country.count > 8 ? 'YUKSEK' : country.count > 3 ? 'ORTA' : 'DUSUK';
  const threatColor = country.count > 15 ? '#ef4444' : country.count > 8 ? '#ff6a00' : country.count > 3 ? '#ffc400' : 'var(--hud-cyan)';

  return (
    <div className="absolute right-3 top-3 bottom-3 w-80 z-20 flex flex-col"
      style={{
        background: 'var(--hud-gradient-surface)',
        border: '1px solid var(--hud-border-strong)',
        borderRadius: 'var(--radius-lg)',
        boxShadow: 'var(--hud-shadow-lg)',
        backdropFilter: 'blur(20px)',
      }}>
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <MapPin size={14} style={{ color: threatColor }} />
          <span className="font-mono text-xs font-bold tracking-widest" style={{ color: 'var(--hud-text-bright)' }}>
            {country.name || country.code}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[9px] px-2 py-0.5 rounded" style={{
            color: threatColor,
            border: `1px solid ${threatColor}33`,
            background: `${threatColor}15`,
            letterSpacing: '2px',
          }}>
            {threatLevel}
          </span>
          <button onClick={onClose} className="p-1 rounded hover:bg-white/5 transition-colors">
            <X size={14} style={{ color: 'var(--hud-text-muted)' }} />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex border-b" style={{ borderColor: 'var(--hud-border)' }}>
        {['overview', 'attacks', 'ips'].map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className="flex-1 py-2 font-mono text-[9px] tracking-wide uppercase transition-colors"
            style={{
              color: tab === t ? 'var(--hud-cyan)' : 'var(--hud-text-muted)',
              borderBottom: tab === t ? '2px solid var(--hud-cyan)' : '2px solid transparent',
              background: tab === t ? 'rgba(56,189,248,0.04)' : 'transparent',
            }}
          >
            {t === 'overview' ? 'Genel' : t === 'attacks' ? 'Saldirilar' : 'IP Listesi'}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-3" style={{ scrollbarWidth: 'thin' }}>
        {tab === 'overview' && (
          <div className="space-y-3">
            {/* Stats grid */}
            <div className="grid grid-cols-2 gap-2">
              <StatBox label="Toplam Saldiri" value={country.count || 0} color={threatColor} icon={<Activity size={12} />} />
              <StatBox label="Kritik" value={severityCounts.critical} color="#ef4444" icon={<AlertTriangle size={12} />} />
              <StatBox label="Yuksek" value={severityCounts.high} color="#ff6a00" icon={<Shield size={12} />} />
              <StatBox label="Orta/Dusuk" value={severityCounts.medium + severityCounts.low} color="#ffc400" icon={<Clock size={12} />} />
            </div>

            {/* Severity bar */}
            <div>
              <div className="font-mono text-[9px] text-[var(--hud-text-muted)] mb-1 tracking-wider">SEVERITY DAGILIM</div>
              <div className="flex h-2 rounded-full overflow-hidden" style={{ background: 'rgba(56,189,248,0.04)' }}>
                {Object.entries(severityCounts).map(([sev, count]) => {
                  const pct = country.count > 0 ? (count / country.count) * 100 : 0;
                  if (pct === 0) return null;
                  return <div key={sev} style={{ width: `${pct}%`, background: SEVERITY_COLORS[sev] }} />;
                })}
              </div>
            </div>

            {/* Top attack types */}
            <div>
              <div className="font-mono text-[9px] text-[var(--hud-text-muted)] mb-2 tracking-wider">TOP SALDIRI TIPLERI</div>
              {topAttackTypes.map(([type, count], i) => (
                <div key={type} className="flex items-center justify-between py-1.5 border-b" style={{ borderColor: 'var(--hud-border-subtle)' }}>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px]" style={{ color: 'var(--hud-cyan-dim)' }}>0{i + 1}</span>
                    <span className="font-mono text-[10px]" style={{ color: 'var(--hud-text)' }}>{type}</span>
                  </div>
                  <span className="font-mono text-[10px] font-bold" style={{ color: threatColor }}>{count}</span>
                </div>
              ))}
              {topAttackTypes.length === 0 && (
                <div className="font-mono text-[10px] text-center py-4" style={{ color: 'var(--hud-text-dim)' }}>
                  Veri yok
                </div>
              )}
            </div>
          </div>
        )}

        {tab === 'attacks' && (
          <div className="space-y-2">
            {countryAttacks.slice(0, 20).map((a, i) => (
              <div key={i} className="p-2 rounded" style={{
                background: 'rgba(56,189,248,0.02)',
                border: '1px solid var(--hud-border-subtle)',
              }}>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-mono text-[10px] font-bold" style={{ color: SEVERITY_COLORS[a.severity] || 'var(--hud-cyan)' }}>
                    {a.attack_type || a.type || 'Unknown'}
                  </span>
                  <span className="font-mono text-[8px] px-1.5 py-0.5 rounded" style={{
                    color: SEVERITY_COLORS[a.severity],
                    border: `1px solid ${SEVERITY_COLORS[a.severity]}33`,
                    background: `${SEVERITY_COLORS[a.severity]}10`,
                  }}>
                    {a.severity}
                  </span>
                </div>
                <div className="font-mono text-[9px]" style={{ color: 'var(--hud-text-muted)' }}>
                  {a.source?.ip || 'N/A'} → {a.target?.ip || 'TR'}
                </div>
                {a.ml_prediction?.confidence && (
                  <div className="font-mono text-[9px] mt-0.5" style={{ color: 'var(--hud-purple)' }}>
                    AI Conf: {(a.ml_prediction.confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            ))}
            {countryAttacks.length === 0 && (
              <div className="font-mono text-[10px] text-center py-8" style={{ color: 'var(--hud-text-dim)' }}>
                Bu ulkeden saldiri yok
              </div>
            )}
          </div>
        )}

        {tab === 'ips' && (
          <div className="space-y-1">
            {[...new Set(countryAttacks.map(a => a.source?.ip).filter(Boolean))].slice(0, 30).map((ip, i) => {
              const ipAttacks = countryAttacks.filter(a => a.source?.ip === ip);
              const worstSev = ipAttacks.reduce((w, a) => {
                const order = { critical: 4, high: 3, medium: 2, low: 1, info: 0 };
                return order[a.severity] > order[w] ? a.severity : w;
              }, 'info');
              return (
                <div key={ip} className="flex items-center justify-between py-1.5 px-2 rounded" style={{
                  background: i % 2 === 0 ? 'rgba(56,189,248,0.02)' : 'transparent',
                }}>
                  <span className="font-mono text-[10px]" style={{ color: 'var(--hud-text)' }}>{ip}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[9px]" style={{ color: 'var(--hud-text-muted)' }}>{ipAttacks.length}x</span>
                    <div className="w-2 h-2 rounded-full" style={{ background: SEVERITY_COLORS[worstSev] }} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

function StatBox({ label, value, color, icon }) {
  return (
    <div className="p-2 rounded" style={{
      background: `${color}08`,
      border: `1px solid ${color}20`,
    }}>
      <div className="flex items-center gap-1 mb-1">
        <span style={{ color: `${color}80` }}>{icon}</span>
        <span className="font-mono text-[8px] tracking-wide" style={{ color: 'var(--hud-text-muted)' }}>{label}</span>
      </div>
      <div className="font-mono text-lg font-bold" style={{ color }}>{value}</div>
    </div>
  );
}
