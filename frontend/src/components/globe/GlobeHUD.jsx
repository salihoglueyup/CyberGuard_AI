import React, { useState, useEffect, useMemo } from 'react';

// ─── Severity configuration ──────────────────────────────────
const SEVERITY_MAP = {
    critical: { label: 'KRİTİK', color: '#ef4444', bg: 'bg-red-500/10', border: 'border-red-500/40' },
    high:     { label: 'YÜKSEK', color: '#ff6a00', bg: 'bg-orange-500/10', border: 'border-orange-500/40' },
    medium:   { label: 'ORTA',   color: '#ffc400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/40' },
    low:      { label: 'DÜŞÜK',  color: '#10b981', bg: 'bg-green-500/10', border: 'border-green-500/40' },
    info:     { label: 'BİLGİ',  color: 'var(--hud-cyan)', bg: 'bg-cyan-500/10', border: 'border-cyan-500/40' },
};

// ─── System Clock ─────────────────────────────────────────────
function SystemClock() {
    const [now, setNow] = useState(new Date());
    useEffect(() => {
        const t = setInterval(() => setNow(new Date()), 1000);
        return () => clearInterval(t);
    }, []);
    return (
        <span className="font-mono text-[11px] text-[var(--hud-cyan)]/80 tracking-widest tabular-nums">
            {now.toLocaleTimeString('tr-TR', { hour12: false })}
            <span className="text-slate-500 mx-1">|</span>
            {now.toLocaleDateString('tr-TR')}
        </span>
    );
}

// ─── Threat Level Indicator Bar ───────────────────────────────
function ThreatLevelBar({ stats }) {
    const level = stats.critical > 5 ? 4 : stats.critical > 2 ? 3 : stats.critical > 0 ? 2 : stats.total > 0 ? 1 : 0;
    const labels = ['GÜVENLİ', 'DÜŞÜK', 'ORTA', 'YÜKSEK', 'KRİTİK'];
    const colors = ['#10b981', '#38bdf8', '#ffc400', '#ff6a00', '#ef4444'];

    return (
        <div className="flex items-center gap-2">
            <span className="text-[9px] text-slate-500 font-mono tracking-wider">TEHDİT</span>
            <div className="flex gap-0.5">
                {[0, 1, 2, 3, 4].map(i => (
                    <div
                        key={i}
                        className="w-3 h-1.5 rounded-sm transition-all duration-500"
                        style={{
                            backgroundColor: i <= level ? colors[level] : 'rgba(51, 65, 85, 0.4)',
                            boxShadow: i <= level ? `0 0 6px ${colors[level]}40` : 'none',
                        }}
                    />
                ))}
            </div>
            <span className="text-[9px] font-bold font-mono tracking-wider" style={{ color: colors[level] }}>
                {labels[level]}
            </span>
        </div>
    );
}

// ─── Main HUD Overlay ─────────────────────────────────────────
export default function GlobeHUD({
    stats = {},
    attacks = [],
    isLive = true,
    selectedCountry = null,
    onCloseCountry,
}) {
    // ─── Recent critical/high events ──────────────────────────
    const recentAlerts = useMemo(() => {
        return attacks
            .filter(a => a.severity === 'critical' || a.severity === 'high')
            .slice(0, 3);
    }, [attacks]);

    // ─── Top attack sources ───────────────────────────────────
    const topSources = useMemo(() => {
        const counts = {};
        attacks.forEach(a => {
            const src = a.source?.country || a.source?.name;
            if (src) counts[src] = (counts[src] || 0) + 1;
        });
        return Object.entries(counts)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 5);
    }, [attacks]);

    // ─── Attack type distribution ─────────────────────────────
    const attackTypes = useMemo(() => {
        const counts = {};
        attacks.forEach(a => {
            const type = a.threat_type || a.attack_type || 'unknown';
            counts[type] = (counts[type] || 0) + 1;
        });
        return Object.entries(counts)
            .sort(([, a], [, b]) => b - a)
            .slice(0, 4);
    }, [attacks]);

    return (
        <>
            {/* ── TOP-LEFT: System Status ─────────────────────── */}
            <div className="absolute top-3 left-3 z-30 pointer-events-auto">
                <div className="bg-slate-950/80 backdrop-blur-xl rounded-lg border border-slate-700/40 p-3 min-w-[220px]
                    shadow-[0_0_30px_rgba(56,189,248,0.05)]">
                    {/* Header */}
                    <div className="flex items-center justify-between mb-2.5 pb-2 border-b border-slate-800/60">
                        <div className="flex items-center gap-2">
                            <div className={`w-2 h-2 rounded-full ${isLive ? 'bg-cyan-400 animate-pulse shadow-[0_0_8px_rgba(56,189,248,0.6)]' : 'bg-slate-500'}`} />
                            <span className="text-[10px] font-bold font-mono tracking-wide text-slate-300">
                                {isLive ? 'CANLI İZLEME' : 'DURAKLADI'}
                            </span>
                        </div>
                        <SystemClock />
                    </div>

                    {/* Threat Level */}
                    <ThreatLevelBar stats={stats} />

                    {/* Mini stats grid */}
                    <div className="grid grid-cols-3 gap-2 mt-2.5">
                        <div className="text-center">
                            <div className="text-[9px] text-slate-500 font-mono">TOPLAM</div>
                            <div className="text-lg font-black text-[var(--hud-cyan)] tabular-nums">{stats.total || 0}</div>
                        </div>
                        <div className="text-center">
                            <div className="text-[9px] text-slate-500 font-mono">ENGELLENDİ</div>
                            <div className="text-lg font-black text-green-400 tabular-nums">{stats.blocked || 0}</div>
                        </div>
                        <div className="text-center">
                            <div className="text-[9px] text-slate-500 font-mono">KRİTİK</div>
                            <div className={`text-lg font-black tabular-nums ${stats.critical > 0 ? 'text-red-500 animate-pulse' : 'text-slate-600'}`}>
                                {stats.critical || 0}
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── TOP-RIGHT: Alert Feed ───────────────────────── */}
            {recentAlerts.length > 0 && (
                <div className="absolute top-3 right-3 z-30 pointer-events-auto max-w-[260px]">
                    <div className="bg-slate-950/80 backdrop-blur-xl rounded-lg border border-red-500/20 p-3
                        shadow-[0_0_30px_rgba(239,68,68,0.08)]">
                        <div className="flex items-center gap-2 mb-2.5 pb-2 border-b border-red-500/10">
                            <span className="text-red-400 text-sm">⚠</span>
                            <span className="text-[10px] font-bold font-mono tracking-wide text-red-400/80">
                                ALARM AKIŞI
                            </span>
                        </div>
                        <div className="space-y-2">
                            {recentAlerts.map((a, i) => {
                                const sev = SEVERITY_MAP[a.severity] || SEVERITY_MAP.info;
                                return (
                                    <div key={a.id || i}
                                        className={`${sev.bg} border ${sev.border} rounded-md p-2 text-[11px]`}>
                                        <div className="flex justify-between items-center mb-0.5">
                                            <span className="font-bold text-[var(--hud-text)] truncate max-w-[140px]">
                                                {a.threat_type || a.attack_type || 'Bilinmeyen'}
                                            </span>
                                            <span className="font-mono font-bold px-1.5 py-0.5 rounded text-[8px]"
                                                style={{ color: sev.color, border: `1px solid ${sev.color}40` }}>
                                                {sev.label}
                                            </span>
                                        </div>
                                        <div className="text-slate-400 font-mono">
                                            {a.source?.country || '?'} → {a.target?.country || 'TR'}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* ── BOTTOM-LEFT: Top Sources ────────────────────── */}
            {topSources.length > 0 && (
                <div className="absolute bottom-3 left-3 z-30 pointer-events-auto">
                    <div className="bg-slate-950/80 backdrop-blur-xl rounded-lg border border-slate-700/40 p-3 min-w-[200px]
                        shadow-[0_0_20px_rgba(0,0,0,0.3)]">
                        <div className="text-[9px] font-bold font-mono tracking-wide text-slate-400 mb-2">
                            EN YOĞUN KAYNAKLAR
                        </div>
                        <div className="space-y-1.5">
                            {topSources.map(([name, count], i) => {
                                const max = topSources[0][1];
                                const pct = (count / max) * 100;
                                return (
                                    <div key={name} className="flex items-center gap-2">
                                        <span className="text-[10px] text-slate-500 font-mono w-3">{i + 1}</span>
                                        <div className="flex-1 relative h-4 bg-slate-800/60 rounded overflow-hidden">
                                            <div
                                                className="absolute inset-y-0 left-0 rounded transition-all duration-700"
                                                style={{
                                                    width: `${pct}%`,
                                                    background: i === 0
                                                        ? 'linear-gradient(90deg, rgba(239,68,68,0.3), rgba(239,68,68,0.6))'
                                                        : `rgba(56,189,248,${0.15 + (1 - i / 5) * 0.3})`,
                                                }}
                                            />
                                            <div className="relative flex justify-between items-center px-2 h-full">
                                                <span className="text-[10px] font-mono text-slate-200 truncate">{name}</span>
                                                <span className="text-[10px] font-bold font-mono text-[var(--hud-text)]">{count}</span>
                                            </div>
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                </div>
            )}

            {/* ── BOTTOM-RIGHT: Attack Types ──────────────────── */}
            {attackTypes.length > 0 && (
                <div className="absolute bottom-3 right-3 z-30 pointer-events-auto">
                    <div className="bg-slate-950/80 backdrop-blur-xl rounded-lg border border-slate-700/40 p-3 min-w-[180px]">
                        <div className="text-[9px] font-bold font-mono tracking-wide text-slate-400 mb-2">
                            SALDIRI TÜRLERİ
                        </div>
                        <div className="space-y-1.5">
                            {attackTypes.map(([type, count]) => (
                                <div key={type} className="flex justify-between items-center">
                                    <span className="text-[10px] text-slate-300 font-mono truncate max-w-[120px]">{type}</span>
                                    <span className="text-[10px] font-bold text-[var(--hud-cyan)]/80 font-mono">{count}</span>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* ── Country Drill-Down Panel ─────────────────────── */}
            {selectedCountry && (
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-40 pointer-events-auto">
                    <div className="bg-slate-950/95 backdrop-blur-2xl rounded-xl border border-cyan-500/30 p-5 min-w-[320px]
                        shadow-[0_0_60px_rgba(56,189,248,0.12)]
                        animate-[fadeIn_0.3s_ease-out]">
                        <div className="flex justify-between items-start mb-4">
                            <div>
                                <h3 className="text-lg font-black text-[var(--hud-text)]">{selectedCountry.name || selectedCountry.code}</h3>
                                <p className="text-[11px] text-slate-400 font-mono">{selectedCountry.code} // DETAYLI ANALİZ</p>
                            </div>
                            <button
                                onClick={onCloseCountry}
                                aria-label="Kapat"
                                className="text-slate-500 hover:text-[var(--hud-text)] text-lg leading-none p-1 rounded hover:bg-slate-800 transition-colors"
                            >
                                ×
                            </button>
                        </div>
                        <div className="grid grid-cols-2 gap-3 mb-4">
                            <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                                <div className="text-[9px] text-slate-500 font-mono mb-1">SALDIRI</div>
                                <div className="text-2xl font-black text-[var(--hud-cyan)]">{selectedCountry.count || 0}</div>
                            </div>
                            <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                                <div className="text-[9px] text-slate-500 font-mono mb-1">TEHDİT SEVİYESİ</div>
                                <div className="text-2xl font-black" style={{
                                    color: (selectedCountry.count || 0) > 15 ? '#ef4444' :
                                           (selectedCountry.count || 0) > 8 ? '#ff6a00' :
                                           (selectedCountry.count || 0) > 3 ? '#ffc400' : '#10b981'
                                }}>
                                    {(selectedCountry.count || 0) > 15 ? 'KRİTİK' :
                                     (selectedCountry.count || 0) > 8 ? 'YÜKSEK' :
                                     (selectedCountry.count || 0) > 3 ? 'ORTA' : 'DÜŞÜK'}
                                </div>
                            </div>
                        </div>
                        <button
                            onClick={onCloseCountry}
                            className="w-full text-center text-[11px] font-mono text-slate-400 hover:text-[var(--hud-cyan)] py-2 border border-slate-700/50 hover:border-cyan-500/30 rounded-lg transition-all"
                        >
                            [ KAPAT ]
                        </button>
                    </div>
                </div>
            )}

            {/* ── Scanline Effect (Palantir signature) ────────── */}
            <div className="absolute inset-0 z-10 pointer-events-none overflow-hidden rounded-2xl">
                <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_2px,rgba(56,189,248,0.015)_2px,rgba(56,189,248,0.015)_4px)]" />
                <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-cyan-500/30 to-transparent
                    animate-[scanline_4s_linear_infinite]" />
            </div>

            <style>{`
                @keyframes scanline {
                    0% { transform: translateY(0); }
                    100% { transform: translateY(calc(100vh - 220px)); }
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translate(-50%, -50%) scale(0.95); }
                    to { opacity: 1; transform: translate(-50%, -50%) scale(1); }
                }
            `}</style>
        </>
    );
}
