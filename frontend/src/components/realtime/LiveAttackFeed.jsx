import { useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useRealtimeStore } from '../../hooks/useRealtimeMetrics';
import { AlertTriangle, ShieldCheck, ShieldX, Globe, Clock } from 'lucide-react';

const SEVERITY_COLORS = {
    critical: '#ef4444',
    high: '#ff6d00',
    medium: '#ffab00',
    low: 'var(--hud-cyan)',
};

const SEVERITY_LABELS = {
    critical: 'KRITIK',
    high: 'YUKSEK',
    medium: 'ORTA',
    low: 'DUSUK',
};

function AttackRow({ attack, isNew }) {
    return (
        <motion.div
            layout
            initial={isNew ? { opacity: 0, x: -20, height: 0 } : false}
            animate={{ opacity: 1, x: 0, height: 'auto' }}
            exit={{ opacity: 0, x: 20, height: 0 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className="border-b border-[rgba(255,255,255,0.03)] hover:bg-[rgba(0,229,255,0.03)] transition-colors"
        >
            <div className="flex items-center gap-2 py-2 px-2">
                {/* Severity indicator */}
                <div className="w-1 h-8 rounded-full" style={{ backgroundColor: SEVERITY_COLORS[attack.severity] }} />

                {/* Blocked/Allowed icon */}
                {attack.blocked ? (
                    <ShieldCheck className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
                ) : (
                    <ShieldX className="w-3.5 h-3.5 text-red-400 flex-shrink-0 animate-pulse" />
                )}

                {/* Info */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <span className="text-[10px] font-mono font-bold" style={{ color: SEVERITY_COLORS[attack.severity] }}>
                            {attack.type}
                        </span>
                        <span className="text-[8px] font-mono px-1 py-0.5 rounded" style={{
                            color: SEVERITY_COLORS[attack.severity],
                            backgroundColor: `${SEVERITY_COLORS[attack.severity]}15`,
                        }}>
                            {SEVERITY_LABELS[attack.severity]}
                        </span>
                    </div>
                    <div className="flex items-center gap-2 text-[9px] font-mono text-[var(--hud-text-dim)] mt-0.5">
                        <span>{attack.source_ip}</span>
                        <span className="text-[var(--hud-cyan)]">→</span>
                        <span>{attack.dest_ip}:{attack.port}</span>
                        <span className="flex items-center gap-0.5"><Globe className="w-2.5 h-2.5" />{attack.source_country}</span>
                    </div>
                </div>

                {/* Protocol & confidence */}
                <div className="text-right flex-shrink-0">
                    <div className="text-[9px] font-mono text-[var(--hud-text-muted)]">{attack.protocol}</div>
                    <div className="text-[8px] font-mono text-[var(--hud-text-dim)]">{attack.confidence}%</div>
                </div>

                {/* Time */}
                <div className="flex items-center gap-0.5 text-[8px] font-mono text-[var(--hud-text-dim)] flex-shrink-0">
                    <Clock className="w-2.5 h-2.5" />
                    {attack.time}
                </div>
            </div>
        </motion.div>
    );
}

export default function LiveAttackFeed({ maxItems = 15 }) {
    const liveAttacks = useRealtimeStore(s => s.liveAttacks);
    const totalAttacks = useRealtimeStore(s => s.totalAttacks);
    const totalBlocked = useRealtimeStore(s => s.totalBlocked);
    const attacksPerMinute = useRealtimeStore(s => s.attacksPerMinute);
    const scrollRef = useRef(null);

    const blockRate = totalAttacks > 0 ? Math.round((totalBlocked / totalAttacks) * 100) : 100;
    const displayAttacks = liveAttacks.slice(0, maxItems);

    return (
        <div className="h-full flex flex-col">
            {/* Header stats */}
            <div className="flex items-center justify-between mb-2 px-1">
                <div className="flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
                    <span className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide">CANLI SALDIRI AKISI</span>
                    <span className="text-[8px] text-red-400 animate-pulse font-mono">● LIVE</span>
                </div>
            </div>

            {/* Quick stats bar */}
            <div className="flex gap-3 mb-2 text-[9px] font-mono">
                <span className="text-[var(--hud-text-dim)]">
                    TOPLAM: <span className="text-[var(--hud-text)] font-bold">{totalAttacks}</span>
                </span>
                <span className="text-[var(--hud-text-dim)]">
                    ENGEL: <span className="text-emerald-400 font-bold">{totalBlocked}</span>
                </span>
                <span className="text-[var(--hud-text-dim)]">
                    ORAN: <span className={`font-bold ${blockRate > 80 ? 'text-emerald-400' : blockRate > 50 ? 'text-amber-400' : 'text-red-400'}`}>{blockRate}%</span>
                </span>
                <span className="text-[var(--hud-text-dim)]">
                    APM: <span className="text-[var(--hud-cyan)] font-bold">{attacksPerMinute}</span>
                </span>
            </div>

            {/* Severity distribution mini bar */}
            <div className="flex h-1 rounded-full overflow-hidden mb-2 border border-[var(--hud-border)]">
                {['critical', 'high', 'medium', 'low'].map(sev => {
                    const count = liveAttacks.filter(a => a.severity === sev).length;
                    const pct = liveAttacks.length > 0 ? (count / liveAttacks.length) * 100 : 0;
                    return <div key={sev} style={{ width: `${pct}%`, backgroundColor: SEVERITY_COLORS[sev] }} />;
                })}
            </div>

            {/* Scrollable attack list */}
            <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0">
                <AnimatePresence initial={false}>
                    {displayAttacks.length > 0 ? (
                        displayAttacks.map((attack, i) => (
                            <AttackRow key={attack.id} attack={attack} isNew={i === 0} />
                        ))
                    ) : (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex items-center justify-center h-full text-[var(--hud-text-dim)] text-[10px] font-mono"
                        >
                            SALDIRI BEKLENIYOR...
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
}
