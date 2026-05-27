import { useState, useEffect } from 'react';
import { Clock } from 'lucide-react';

const ZONES = [
  { label: 'TR', tz: 'Europe/Istanbul', flag: '🇹🇷' },
  { label: 'UTC', tz: 'UTC', flag: '🌐' },
  { label: 'US-E', tz: 'America/New_York', flag: '🇺🇸' },
  { label: 'CN', tz: 'Asia/Shanghai', flag: '🇨🇳' },
  { label: 'RU', tz: 'Europe/Moscow', flag: '🇷🇺' },
  { label: 'JP', tz: 'Asia/Tokyo', flag: '🇯🇵' },
];

export default function WorldClock({ zones = ZONES, className = '' }) {
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const iv = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(iv);
  }, []);

  const fmt = (tz) => {
    try {
      return now.toLocaleTimeString('tr-TR', { timeZone: tz, hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
    } catch { return '--:--:--'; }
  };

  const getHour = (tz) => {
    try {
      return parseInt(now.toLocaleString('en-US', { timeZone: tz, hour: 'numeric', hour12: false }));
    } catch { return 12; }
  };

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
      overflow: 'hidden',
    }}>
      <div className="flex items-center gap-2 px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <Clock size={12} style={{ color: 'var(--hud-cyan)' }} />
        <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>
          WORLD CLOCK
        </span>
      </div>
      <div className="grid grid-cols-3 gap-0">
        {zones.map(z => {
          const h = getHour(z.tz);
          const isNight = h < 6 || h >= 20;
          return (
            <div key={z.label} className="flex flex-col items-center py-2 px-1 border-b border-r"
              style={{ borderColor: 'var(--hud-border-subtle)' }}>
              <span className="text-[10px]">{z.flag}</span>
              <span className="font-mono text-[8px] tracking-wider" style={{ color: 'var(--hud-text-muted)' }}>
                {z.label}
              </span>
              <span className="font-mono text-[11px] font-bold"
                style={{ color: isNight ? 'var(--hud-amber)' : 'var(--hud-cyan)' }}>
                {fmt(z.tz)}
              </span>
              <span className="font-mono text-[7px]" style={{ color: 'var(--hud-text-dim)' }}>
                {isNight ? 'GECE' : 'GUNDUZ'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
