import { useState } from 'react';
import { Activity, X, ChevronUp, ChevronDown } from 'lucide-react';
import { usePerformanceStore, usePerformanceMonitor } from '../../hooks/usePerformance';

function MetricRow({ label, value, unit, good, warn }) {
    const numVal = parseFloat(value);
    const color = numVal <= good ? 'var(--hud-emerald)' : numVal <= warn ? 'var(--hud-amber)' : 'var(--hud-red)';
    return (
        <div className="flex items-center justify-between py-0.5">
            <span className="text-[9px] font-mono text-[var(--hud-text-dim)] uppercase">{label}</span>
            <span className="text-[10px] font-mono font-bold tabular-nums" style={{ color }}>
                {value ?? '—'}{unit && <span className="text-[8px] ml-0.5 opacity-60">{unit}</span>}
            </span>
        </div>
    );
}

export default function PerformanceOverlay() {
    usePerformanceMonitor();
    const metrics = usePerformanceStore((s) => s.metrics);
    const [open, setOpen] = useState(false);
    const [minimized, setMinimized] = useState(false);

    if (!import.meta.env.DEV && !localStorage.getItem('cyberguard-perf-overlay')) return null;

    if (!open) {
        return (
            <button
                onClick={() => setOpen(true)}
                className="fixed bottom-4 right-4 z-[9999] p-2 rounded-full bg-[var(--hud-surface)] border border-[var(--hud-border)] text-[var(--hud-cyan)] hover:bg-[rgba(56,189,248,0.1)] transition-all shadow-lg"
                title="Performance Monitor"
            >
                <Activity className="w-4 h-4" />
            </button>
        );
    }

    return (
        <div className="fixed bottom-4 right-4 z-[9999] w-56 bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg shadow-2xl overflow-hidden"
            style={{ backdropFilter: 'blur(12px)' }}>
            <div className="flex items-center justify-between px-3 py-1.5 bg-[rgba(56,189,248,0.04)] border-b border-[var(--hud-border)]">
                <div className="flex items-center gap-1.5">
                    <Activity className="w-3 h-3 text-[var(--hud-cyan)]" />
                    <span className="text-[9px] font-mono text-[var(--hud-cyan)] tracking-wide">PERF MONITOR</span>
                </div>
                <div className="flex items-center gap-1">
                    <button onClick={() => setMinimized(!minimized)} className="text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]">
                        {minimized ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                    </button>
                    <button onClick={() => setOpen(false)} className="text-[var(--hud-text-dim)] hover:text-[var(--hud-red)]">
                        <X className="w-3 h-3" />
                    </button>
                </div>
            </div>

            {!minimized && (
                <div className="px-3 py-2 space-y-0.5">
                    <MetricRow label="FPS" value={metrics.fps} good={55} warn={30} />
                    <MetricRow label="FCP" value={metrics.fcp} unit="ms" good={1800} warn={3000} />
                    <MetricRow label="LCP" value={metrics.lcp} unit="ms" good={2500} warn={4000} />
                    <MetricRow label="CLS" value={metrics.cls} good={0.1} warn={0.25} />
                    <MetricRow label="TTFB" value={metrics.ttfb} unit="ms" good={800} warn={1800} />
                    <div className="border-t border-[var(--hud-border)] mt-1 pt-1">
                        <MetricRow label="DOM" value={metrics.domNodes} good={1500} warn={3000} />
                        {metrics.jsHeap && <MetricRow label="HEAP" value={metrics.jsHeap} unit="MB" good={50} warn={100} />}
                    </div>
                </div>
            )}
        </div>
    );
}
