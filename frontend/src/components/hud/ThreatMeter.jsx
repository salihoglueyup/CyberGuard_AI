import { useMemo } from 'react';

const LEVELS = [
    { label: 'LOW', color: 'var(--hud-emerald)', threshold: 20 },
    { label: 'GUARDED', color: '#4ade80', threshold: 40 },
    { label: 'ELEVATED', color: 'var(--hud-amber)', threshold: 60 },
    { label: 'HIGH', color: '#ff6b35', threshold: 80 },
    { label: 'CRITICAL', color: 'var(--hud-red)', threshold: 100 },
];

export default function ThreatMeter({ value = 0, className = '' }) {
    const level = useMemo(() => {
        for (const l of LEVELS) {
            if (value <= l.threshold) return l;
        }
        return LEVELS[LEVELS.length - 1];
    }, [value]);

    const pct = Math.min(100, Math.max(0, value));

    return (
        <div className={`font-mono ${className}`}>
            <div className="flex items-center justify-between mb-1">
                <span className="text-[9px] text-[var(--hud-text-dim)] tracking-[0.15em]">THREAT LEVEL</span>
                <span className="text-[10px] font-bold tracking-wider" style={{ color: level.color }}>
                    {level.label}
                </span>
            </div>

            {/* Bar track */}
            <div className="relative h-2 bg-[rgba(255,255,255,0.04)] rounded-sm overflow-hidden border border-[var(--hud-border)]">
                <div
                    className="absolute inset-y-0 left-0 rounded-sm transition-all duration-700"
                    style={{
                        width: `${pct}%`,
                        background: `linear-gradient(90deg, var(--hud-emerald), ${level.color})`,
                        boxShadow: `0 0 8px ${level.color}40`,
                    }}
                />
                {/* Tick marks */}
                {[20, 40, 60, 80].map(tick => (
                    <div
                        key={tick}
                        className="absolute top-0 bottom-0 w-px bg-[rgba(255,255,255,0.08)]"
                        style={{ left: `${tick}%` }}
                    />
                ))}
            </div>

            <div className="flex justify-between mt-0.5">
                <span className="text-[8px] text-[var(--hud-text-dim)]">0</span>
                <span className="text-[8px] font-bold" style={{ color: level.color }}>{pct}</span>
                <span className="text-[8px] text-[var(--hud-text-dim)]">100</span>
            </div>
        </div>
    );
}
