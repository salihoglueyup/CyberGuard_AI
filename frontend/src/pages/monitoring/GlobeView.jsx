import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { Globe2, Crosshair, Satellite, Radio, Shield, Zap, Filter, Maximize2, RotateCcw, X, Play, Pause, SkipForward, MapPin, Info, Layers } from 'lucide-react';
import Globe from 'react-globe.gl';
import { motion, AnimatePresence } from 'framer-motion';
import { useWebSocket } from '../../hooks/useWebSocket';
import { useRealtimeStore } from '../../hooks/useRealtimeMetrics';

const ATTACK_ORIGINS = [
    { lat: 39.9, lng: 32.8, city: 'Ankara', country: 'TR' },
    { lat: 55.75, lng: 37.6, city: 'Moscow', country: 'RU' },
    { lat: 31.23, lng: 121.47, city: 'Shanghai', country: 'CN' },
    { lat: 37.57, lng: 126.97, city: 'Seoul', country: 'KR' },
    { lat: 35.68, lng: 139.69, city: 'Tokyo', country: 'JP' },
    { lat: 40.71, lng: -74.0, city: 'New York', country: 'US' },
    { lat: 51.51, lng: -0.13, city: 'London', country: 'GB' },
    { lat: -23.55, lng: -46.63, city: 'São Paulo', country: 'BR' },
    { lat: 1.35, lng: 103.82, city: 'Singapore', country: 'SG' },
    { lat: 28.61, lng: 77.21, city: 'New Delhi', country: 'IN' },
    { lat: 48.86, lng: 2.35, city: 'Paris', country: 'FR' },
    { lat: 52.52, lng: 13.41, city: 'Berlin', country: 'DE' },
    { lat: -33.87, lng: 151.21, city: 'Sydney', country: 'AU' },
    { lat: 25.2, lng: 55.27, city: 'Dubai', country: 'AE' },
    { lat: 19.43, lng: -99.13, city: 'Mexico City', country: 'MX' },
];

const ATTACK_TYPES = ['DDoS', 'Brute Force', 'SQL Injection', 'XSS', 'Ransomware', 'Phishing', 'Zero-Day', 'APT'];
const SEVERITY_COLORS = { critical: '#ef4444', high: '#ff6d00', medium: '#ffab00', low: 'var(--hud-cyan)' };
const SEVERITIES = ['critical', 'high', 'medium', 'low'];

// Country attack stats for heatmap rings
const COUNTRY_STATS = ATTACK_ORIGINS.slice(1).map(o => ({
    ...o,
    attacks: Math.floor(Math.random() * 500) + 50,
    blocked: Math.floor(Math.random() * 400) + 30,
    topAttack: ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)],
    risk: ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'][Math.floor(Math.random() * 4)],
}));

function generateArcs(count = 30) {
    const target = ATTACK_ORIGINS[0]; // Ankara = HQ
    return Array.from({ length: count }, (_, i) => {
        const src = ATTACK_ORIGINS[1 + (i % (ATTACK_ORIGINS.length - 1))];
        const sev = SEVERITIES[Math.floor(Math.random() * SEVERITIES.length)];
        return {
            startLat: src.lat + (Math.random() - 0.5) * 5,
            startLng: src.lng + (Math.random() - 0.5) * 5,
            endLat: target.lat + (Math.random() - 0.5) * 2,
            endLng: target.lng + (Math.random() - 0.5) * 2,
            color: SEVERITY_COLORS[sev],
            type: ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)],
            severity: sev,
            source: src,
            stroke: sev === 'critical' ? 1.8 : sev === 'high' ? 1.2 : 0.7,
        };
    });
}

export default function GlobeView() {
    const globeRef = useRef();
    const { threats } = useWebSocket();
    const liveAttacks = useRealtimeStore(s => s.liveAttacks);
    const [arcs, setArcs] = useState(() => generateArcs(30));
    const [hoveredArc, setHoveredArc] = useState(null);
    const [filter, setFilter] = useState('all');
    const [autoRotate, setAutoRotate] = useState(true);
    const [selectedCountry, setSelectedCountry] = useState(null);
    const [showHeatmap, setShowHeatmap] = useState(true);
    const [replayMode, setReplayMode] = useState(false);
    const [replayPaused, setReplayPaused] = useState(false);
    const [replayIndex, setReplayIndex] = useState(0);
    const [attackLog, setAttackLog] = useState([]);

    useEffect(() => {
        const iv = setInterval(() => {
            setArcs(prev => {
                const next = [...prev.slice(-40)];
                const src = ATTACK_ORIGINS[1 + Math.floor(Math.random() * (ATTACK_ORIGINS.length - 1))];
                const sev = SEVERITIES[Math.floor(Math.random() * SEVERITIES.length)];
                next.push({
                    startLat: src.lat + (Math.random() - 0.5) * 3,
                    startLng: src.lng + (Math.random() - 0.5) * 3,
                    endLat: 39.9 + (Math.random() - 0.5),
                    endLng: 32.8 + (Math.random() - 0.5),
                    color: SEVERITY_COLORS[sev],
                    type: ATTACK_TYPES[Math.floor(Math.random() * ATTACK_TYPES.length)],
                    severity: sev,
                    source: src,
                    stroke: sev === 'critical' ? 1.8 : 1,
                });
                return next;
            });
        }, 2500);
        return () => clearInterval(iv);
    }, []);

    useEffect(() => {
        if (globeRef.current) {
            const controls = globeRef.current.controls();
            controls.autoRotate = autoRotate;
            controls.autoRotateSpeed = 0.3;
        }
    }, [autoRotate]);

    // Log attacks for replay
    useEffect(() => {
        if (!replayMode) {
            setAttackLog(prev => [...prev, ...arcs.slice(-1)].slice(-200));
        }
    }, [arcs, replayMode]);

    // Replay mode: step through logged attacks
    useEffect(() => {
        if (!replayMode || replayPaused || attackLog.length === 0) return;
        const iv = setInterval(() => {
            setReplayIndex(prev => {
                if (prev >= attackLog.length - 1) {
                    setReplayPaused(true);
                    return prev;
                }
                return prev + 1;
            });
        }, 500);
        return () => clearInterval(iv);
    }, [replayMode, replayPaused, attackLog.length]);

    // Fly to country when clicked
    const flyToCountry = useCallback((countryData) => {
        if (globeRef.current) {
            globeRef.current.pointOfView({ lat: countryData.lat, lng: countryData.lng, altitude: 1.5 }, 1000);
            setAutoRotate(false);
        }
        setSelectedCountry(countryData);
    }, []);

    // Heatmap rings data
    const heatmapRings = useMemo(() => {
        if (!showHeatmap) return [];
        return COUNTRY_STATS.map(c => ({
            lat: c.lat,
            lng: c.lng,
            maxR: Math.min(8, Math.max(2, c.attacks / 80)),
            propagationSpeed: 2,
            repeatPeriod: 1200,
            color: c.risk === 'CRITICAL' ? '#ef4444' : c.risk === 'HIGH' ? '#ff6d00' : c.risk === 'MEDIUM' ? '#ffab00' : '#38bdf8',
        }));
    }, [showHeatmap]);

    const filteredArcs = useMemo(() => {
        const source = replayMode ? attackLog.slice(0, replayIndex + 1).slice(-30) : arcs;
        if (filter === 'all') return source;
        return source.filter(a => a.severity === filter);
    }, [arcs, filter, replayMode, attackLog, replayIndex]);

    const stats = useMemo(() => ({
        total: arcs.length,
        critical: arcs.filter(a => a.severity === 'critical').length,
        high: arcs.filter(a => a.severity === 'high').length,
        countries: new Set(arcs.map(a => a.source?.country)).size,
    }), [arcs]);

    const points = useMemo(() => ATTACK_ORIGINS.map(o => {
        const cs = COUNTRY_STATS.find(c => c.country === o.country);
        return {
            lat: o.lat, lng: o.lng, size: o.country === 'TR' ? 0.8 : 0.4,
            color: o.country === 'TR' ? '#38bdf8' : '#ef4444',
            label: `${o.city}, ${o.country}`,
            ...o,
            ...(cs || {}),
        };
    }), []);

    const handlePointClick = useCallback((point) => {
        if (point.country === 'TR') return;
        const cs = COUNTRY_STATS.find(c => c.country === point.country) || point;
        flyToCountry(cs);
    }, [flyToCountry]);

    const startReplay = useCallback(() => {
        setReplayMode(true);
        setReplayPaused(false);
        setReplayIndex(0);
    }, []);

    const stopReplay = useCallback(() => {
        setReplayMode(false);
        setReplayPaused(false);
        setReplayIndex(0);
    }, []);

    return (
        <div className="min-h-screen bg-[var(--hud-bg)] relative">
            {/* Header */}
            <div className="border-b border-[var(--hud-border)] px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Globe2 className="w-5 h-5 text-[var(--hud-cyan)]" />
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">Global Threat Globe</h1>
                    <span className="text-[9px] text-[var(--hud-text-dim)] bg-[rgba(56,189,248,0.08)] border border-[var(--hud-border)] px-2 py-0.5 rounded">3D INTERACTIVE</span>
                    {replayMode && <span className="text-[9px] text-[var(--hud-red)] bg-red-500/10 border border-red-500/30 px-2 py-0.5 rounded animate-pulse">REPLAY MODE</span>}
                </div>
                <div className="flex items-center gap-2">
                    {/* Heatmap toggle */}
                    <button onClick={() => setShowHeatmap(!showHeatmap)} className={`px-2 py-1 rounded border text-[10px] transition-all ${showHeatmap ? 'border-purple-500/40 text-purple-400 bg-purple-500/10' : 'border-[var(--hud-border)] text-[var(--hud-text-dim)]'}`}>
                        <Layers className="w-3 h-3 inline mr-1" />HEATMAP
                    </button>
                    {/* Replay controls */}
                    {!replayMode ? (
                        <button onClick={startReplay} className="px-2 py-1 rounded border text-[10px] border-amber-500/40 text-[var(--hud-amber)] bg-amber-500/10 transition-all hover:bg-amber-500/20">
                            <Play className="w-3 h-3 inline mr-1" />REPLAY
                        </button>
                    ) : (
                        <>
                            <button onClick={() => setReplayPaused(!replayPaused)} className="px-2 py-1 rounded border text-[10px] border-amber-500/40 text-[var(--hud-amber)] bg-amber-500/10">
                                {replayPaused ? <Play className="w-3 h-3 inline" /> : <Pause className="w-3 h-3 inline" />}
                            </button>
                            <span className="text-[9px] text-[var(--hud-amber)] tabular-nums">{replayIndex + 1}/{attackLog.length}</span>
                            <button onClick={stopReplay} className="px-2 py-1 rounded border text-[10px] border-red-500/40 text-red-400 bg-red-500/10">
                                <X className="w-3 h-3 inline" />
                            </button>
                        </>
                    )}
                    <div className="w-px h-5 bg-[var(--hud-border)]" />
                    <button onClick={() => setAutoRotate(!autoRotate)} className={`px-2 py-1 rounded border text-[10px] transition-all ${autoRotate ? 'border-[var(--hud-cyan)]/40 text-[var(--hud-cyan)] bg-cyan-500/10' : 'border-[var(--hud-border)] text-[var(--hud-text-dim)]'}`}>
                        <RotateCcw className="w-3 h-3 inline mr-1" />{autoRotate ? 'AUTO' : 'MANUAL'}
                    </button>
                    {['all', ...SEVERITIES].map(s => (
                        <button key={s} onClick={() => setFilter(s)}
                            className={`px-2 py-1 rounded border text-[10px] uppercase transition-all ${filter === s ? 'border-[var(--hud-cyan)]/40 text-[var(--hud-cyan)] bg-cyan-500/10' : 'border-[var(--hud-border)] text-[var(--hud-text-dim)]'}`}>
                            {s}
                        </button>
                    ))}
                </div>
            </div>

            {/* Stats bar */}
            <div className="flex gap-4 px-6 py-2 border-b border-[var(--hud-border)]">
                {[
                    { label: 'AKTIF ARC', value: stats.total, color: 'var(--hud-cyan)' },
                    { label: 'KRITIK', value: stats.critical, color: 'var(--hud-red)' },
                    { label: 'YUKSEK', value: stats.high, color: '#ff6d00' },
                    { label: 'ULKE', value: stats.countries, color: 'var(--hud-purple)' },
                ].map(s => (
                    <div key={s.label} className="flex items-center gap-2">
                        <span className="text-[9px] text-[var(--hud-text-dim)] tracking-wider">{s.label}</span>
                        <span className="text-sm font-bold tabular-nums" style={{ color: s.color }}>{s.value}</span>
                    </div>
                ))}
                {/* Clickable country shortcuts */}
                <div className="ml-auto flex gap-1">
                    {COUNTRY_STATS.slice(0, 6).map(c => (
                        <button key={c.country} onClick={() => flyToCountry(c)} className="px-1.5 py-0.5 text-[9px] rounded border border-[var(--hud-border)] text-[var(--hud-text-dim)] hover:border-[var(--hud-cyan)]/40 hover:text-[var(--hud-cyan)] transition-all">
                            {c.country}
                        </button>
                    ))}
                </div>
            </div>

            {/* Globe */}
            <div className="relative" style={{ height: 'calc(100vh - 120px)' }}>
                <Globe
                    ref={globeRef}
                    globeImageUrl="//unpkg.com/three-globe/example/img/earth-dark.jpg"
                    bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
                    backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
                    arcsData={filteredArcs}
                    arcColor={d => d.color}
                    arcStroke={d => d.stroke}
                    arcDashLength={() => 0.4}
                    arcDashGap={() => 0.2}
                    arcDashAnimateTime={() => 1500}
                    arcLabel={d => `<div style="font-family:monospace;font-size:11px;background:rgba(6,10,20,0.95);border:1px solid rgba(56,189,248,0.2);border-radius:4px;padding:6px 10px;color:#e0e0e0">
                        <b style="color:${d.color}">${d.type}</b><br/>
                        <span style="color:#888">${d.source?.city || '?'} → Ankara</span><br/>
                        <span style="color:${d.color};text-transform:uppercase;font-size:9px">${d.severity}</span>
                    </div>`}
                    pointsData={points}
                    pointLat="lat"
                    pointLng="lng"
                    pointColor="color"
                    pointAltitude={0.01}
                    pointRadius="size"
                    pointLabel="label"
                    onPointClick={handlePointClick}
                    ringsData={heatmapRings}
                    ringLat="lat"
                    ringLng="lng"
                    ringMaxRadius="maxR"
                    ringPropagationSpeed="propagationSpeed"
                    ringRepeatPeriod="repeatPeriod"
                    ringColor={d => d.color + '80'}
                    atmosphereColor="#38bdf8"
                    atmosphereAltitude={0.15}
                    width={typeof window !== 'undefined' ? window.innerWidth - 260 : 1000}
                    height={typeof window !== 'undefined' ? window.innerHeight - 120 : 700}
                />

                {/* HUD overlay corners */}
                <div className="absolute top-4 left-4 space-y-2 pointer-events-none">
                    <div className="text-[9px] text-[var(--hud-cyan)]/60 tracking-wide">CYBERGUARD // GLOBAL THREAT MONITOR</div>
                    <div className="text-[9px] text-[var(--hud-text-dim)]">CANLI AKIS: {filteredArcs.length} VEKTOR</div>
                    {showHeatmap && <div className="text-[9px] text-purple-400/60">HEATMAP: {COUNTRY_STATS.length} KAYNAK</div>}
                </div>
                <div className="absolute bottom-4 right-4 text-[9px] text-[var(--hud-text-dim)] pointer-events-none">
                    LAT: 39.9° N &nbsp;|&nbsp; LON: 32.8° E &nbsp;|&nbsp; ALT: ORBITAL
                </div>

                {/* Replay progress bar */}
                {replayMode && (
                    <div className="absolute bottom-12 left-6 right-6">
                        <div className="h-1 bg-[var(--hud-surface)] rounded overflow-hidden border border-[var(--hud-border)]">
                            <div className="h-full bg-amber-500/60 transition-all duration-300" style={{ width: `${attackLog.length ? ((replayIndex + 1) / attackLog.length * 100) : 0}%` }} />
                        </div>
                        <div className="flex justify-between mt-1">
                            <span className="text-[8px] text-[var(--hud-amber)]/60">ATTACK REPLAY</span>
                            <span className="text-[8px] text-[var(--hud-amber)]/60">{replayIndex + 1} / {attackLog.length}</span>
                        </div>
                    </div>
                )}

                {/* Country Detail Panel */}
                <AnimatePresence>
                    {selectedCountry && (
                        <motion.div
                            initial={{ x: 300, opacity: 0 }}
                            animate={{ x: 0, opacity: 1 }}
                            exit={{ x: 300, opacity: 0 }}
                            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
                            className="absolute top-4 right-4 w-72 bg-[var(--hud-surface)] backdrop-blur-md border border-[var(--hud-border)] rounded-lg overflow-hidden shadow-2xl"
                        >
                            <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--hud-border)]">
                                <div className="flex items-center gap-2">
                                    <MapPin className="w-4 h-4 text-[var(--hud-cyan)]" />
                                    <span className="text-sm font-bold text-[var(--hud-cyan)]">{selectedCountry.city}</span>
                                    <span className="text-[9px] text-[var(--hud-text-dim)] border border-[var(--hud-border)] px-1.5 rounded">{selectedCountry.country}</span>
                                </div>
                                <button onClick={() => setSelectedCountry(null)} className="text-[var(--hud-text-dim)] hover:text-[var(--hud-red)] transition-colors">
                                    <X className="w-4 h-4" />
                                </button>
                            </div>
                            <div className="p-4 space-y-3">
                                {/* Risk Level */}
                                <div className="flex items-center gap-2">
                                    <span className="text-[9px] text-[var(--hud-text-dim)] tracking-wider">RISK SEVIYESI</span>
                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                                        selectedCountry.risk === 'CRITICAL' ? 'bg-red-500/20 text-[var(--hud-red)]' :
                                        selectedCountry.risk === 'HIGH' ? 'bg-orange-500/20 text-orange-400' :
                                        selectedCountry.risk === 'MEDIUM' ? 'bg-amber-500/20 text-[var(--hud-amber)]' :
                                        'bg-cyan-500/20 text-[var(--hud-cyan)]'
                                    }`}>{selectedCountry.risk}</span>
                                </div>
                                {/* Stats */}
                                <div className="grid grid-cols-2 gap-2">
                                    {[
                                        { l: 'TOPLAM SALDIRI', v: selectedCountry.attacks, c: 'var(--hud-red)' },
                                        { l: 'ENGELLENEN', v: selectedCountry.blocked, c: 'var(--hud-emerald)' },
                                        { l: 'BASARI ORANI', v: `${selectedCountry.attacks ? Math.round(selectedCountry.blocked / selectedCountry.attacks * 100) : 0}%`, c: 'var(--hud-cyan)' },
                                        { l: 'TOP SALDIRI', v: selectedCountry.topAttack, c: 'var(--hud-amber)' },
                                    ].map(s => (
                                        <div key={s.l} className="bg-[rgba(0,0,0,0.3)] rounded p-2">
                                            <div className="text-[8px] text-[var(--hud-text-dim)] tracking-wider">{s.l}</div>
                                            <div className="text-sm font-bold mt-0.5" style={{ color: s.c }}>{s.v}</div>
                                        </div>
                                    ))}
                                </div>
                                {/* Coordinates */}
                                <div className="text-[9px] text-[var(--hud-text-dim)] flex justify-between border-t border-[var(--hud-border)] pt-2">
                                    <span>LAT: {selectedCountry.lat.toFixed(2)}°</span>
                                    <span>LNG: {selectedCountry.lng.toFixed(2)}°</span>
                                </div>
                                {/* Recent attacks from this country */}
                                <div>
                                    <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mb-1">SON SALDIRILAR</div>
                                    <div className="space-y-1 max-h-32 overflow-y-auto">
                                        {arcs.filter(a => a.source?.country === selectedCountry.country).slice(-5).reverse().map((a, i) => (
                                            <div key={i} className="flex items-center gap-2 text-[9px] py-0.5">
                                                <div className="w-1.5 h-1.5 rounded-full" style={{ background: a.color }} />
                                                <span className="text-[var(--hud-text)]">{a.type}</span>
                                                <span className="ml-auto" style={{ color: a.color }}>{a.severity?.toUpperCase()}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Hovered arc info */}
                {hoveredArc && (
                    <div className="absolute top-4 right-4 hud-panel p-3 min-w-48">
                        <div className="text-[10px] text-[var(--hud-cyan)] font-bold">{hoveredArc.type}</div>
                        <div className="text-[9px] text-[var(--hud-text-dim)] mt-1">{hoveredArc.source?.city} → Ankara</div>
                        <div className="text-[9px] mt-1" style={{ color: hoveredArc.color }}>{hoveredArc.severity?.toUpperCase()}</div>
                    </div>
                )}
            </div>
        </div>
    );
}
