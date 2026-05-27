import { useMemo } from 'react';
import { useRealtimeStore } from '../../hooks/useRealtimeMetrics';
import { Cpu, Server, Wifi, HardDrive, ArrowDownRight, ArrowUpRight, Activity, Zap } from 'lucide-react';

/* ── Tiny sparkline (pure SVG) ── */
function Sparkline({ data, color = 'var(--hud-cyan)', width = 120, height = 32 }) {
    const points = useMemo(() => {
        if (!data || data.length < 2) return '';
        const max = Math.max(...data.map(d => d.value), 1);
        const min = Math.min(...data.map(d => d.value), 0);
        const range = max - min || 1;
        return data
            .map((d, i) => {
                const x = (i / (data.length - 1)) * width;
                const y = height - ((d.value - min) / range) * (height - 4) - 2;
                return `${x},${y}`;
            })
            .join(' ');
    }, [data, width, height]);

    const areaPath = useMemo(() => {
        if (!data || data.length < 2) return '';
        const max = Math.max(...data.map(d => d.value), 1);
        const min = Math.min(...data.map(d => d.value), 0);
        const range = max - min || 1;
        const pts = data.map((d, i) => {
            const x = (i / (data.length - 1)) * width;
            const y = height - ((d.value - min) / range) * (height - 4) - 2;
            return `${x},${y}`;
        });
        return `M0,${height} L${pts.join(' L')} L${width},${height} Z`;
    }, [data, width, height]);

    if (!data || data.length < 2) {
        return <div style={{ width, height }} className="flex items-center justify-center text-[8px] text-[var(--hud-text-dim)]">...</div>;
    }

    return (
        <svg width={width} height={height} className="overflow-visible">
            <defs>
                <linearGradient id={`sg-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={color} stopOpacity={0.25} />
                    <stop offset="100%" stopColor={color} stopOpacity={0} />
                </linearGradient>
            </defs>
            <path d={areaPath} fill={`url(#sg-${color.replace('#', '')})`} />
            <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" />
            {/* Current value dot */}
            {data.length > 0 && (() => {
                const max = Math.max(...data.map(d => d.value), 1);
                const min = Math.min(...data.map(d => d.value), 0);
                const range = max - min || 1;
                const last = data[data.length - 1];
                const cx = width;
                const cy = height - ((last.value - min) / range) * (height - 4) - 2;
                return <circle cx={cx} cy={cy} r={2.5} fill={color} className="animate-pulse" />;
            })()}
        </svg>
    );
}

/* ── Metric Card ── */
function MetricCard({ label, value, unit, icon: Icon, color, data, warn }) {
    return (
        <div className={`hud-panel p-3 transition-all ${warn ? 'border-[rgba(239,68,68,0.25)]' : ''}`}>
            <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-1.5">
                    <Icon className="w-3.5 h-3.5" style={{ color }} />
                    <span className="text-[8px] font-mono text-[var(--hud-text-dim)] tracking-wide">{label}</span>
                </div>
                <span className={`text-sm font-mono font-bold tabular-nums ${warn ? 'animate-pulse' : ''}`} style={{ color }}>
                    {typeof value === 'number' ? value.toFixed(1) : value}{unit}
                </span>
            </div>
            <Sparkline data={data} color={color} width={180} height={28} />
        </div>
    );
}

/* ── Main LiveMetrics panel ── */
export default function LiveMetrics() {
    const cpu = useRealtimeStore(s => s.cpu);
    const memory = useRealtimeStore(s => s.memory);
    const network = useRealtimeStore(s => s.network);
    const disk = useRealtimeStore(s => s.disk);
    const currentCpu = useRealtimeStore(s => s.currentCpu);
    const currentMemory = useRealtimeStore(s => s.currentMemory);
    const currentNetwork = useRealtimeStore(s => s.currentNetwork);
    const currentDisk = useRealtimeStore(s => s.currentDisk);

    return (
        <div className="space-y-2">
            <div className="flex items-center gap-1.5 mb-1">
                <Activity className="w-3 h-3 text-[var(--hud-cyan)]" />
                <span className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide">Canlı Metrikler</span>
                <span className="ml-auto text-[8px] text-emerald-400 animate-pulse font-mono">● LIVE</span>
            </div>
            <MetricCard label="CPU" value={currentCpu} unit="%" icon={Cpu} color={currentCpu > 80 ? '#ef4444' : currentCpu > 60 ? '#ffab00' : 'var(--hud-cyan)'} data={cpu} warn={currentCpu > 85} />
            <MetricCard label="Bellek" value={currentMemory} unit="%" icon={Server} color={currentMemory > 85 ? '#ef4444' : currentMemory > 70 ? '#ffab00' : '#10b981'} data={memory} warn={currentMemory > 88} />
            <MetricCard label="Ağ Kullanımı" value={currentNetwork} unit="%" icon={Wifi} color="#448aff" data={network} />
            <MetricCard label="Depolama" value={currentDisk} unit="%" icon={HardDrive} color="#b388ff" data={disk} />
        </div>
    );
}

/* ── Bandwidth Chart (inbound/outbound) ── */
export function BandwidthChart() {
    const bwIn = useRealtimeStore(s => s.bandwidth.inbound);
    const bwOut = useRealtimeStore(s => s.bandwidth.outbound);
    const currentIn = useRealtimeStore(s => s.currentBandwidthIn);
    const currentOut = useRealtimeStore(s => s.currentBandwidthOut);

    return (
        <div className="space-y-2">
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                    <Activity className="w-3 h-3 text-[var(--hud-cyan)]" />
                    <span className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide">BANT GENISLIGI</span>
                </div>
                <span className="text-[8px] text-emerald-400 animate-pulse font-mono">● LIVE</span>
            </div>
            <div className="flex gap-4 text-[9px] font-mono">
                <span className="flex items-center gap-1 text-emerald-400"><ArrowDownRight className="w-3 h-3" /> GELEN: {Math.round(currentIn)} Mbps</span>
                <span className="flex items-center gap-1 text-amber-400"><ArrowUpRight className="w-3 h-3" /> GIDEN: {Math.round(currentOut)} Mbps</span>
            </div>
            <div className="hud-panel p-2">
                <Sparkline data={bwIn} color="#10b981" width={280} height={36} />
            </div>
            <div className="hud-panel p-2">
                <Sparkline data={bwOut} color="#ffab00" width={280} height={36} />
            </div>
        </div>
    );
}

/* ── Connection & RPS mini stats ── */
export function ConnectionStats() {
    const connections = useRealtimeStore(s => s.connections);
    const requests = useRealtimeStore(s => s.requests);
    const currentConns = useRealtimeStore(s => s.currentConnections);
    const currentRps = useRealtimeStore(s => s.currentRps);

    return (
        <div className="grid grid-cols-2 gap-2">
            <div className="hud-panel p-2">
                <div className="flex items-center gap-1 mb-1">
                    <Wifi className="w-3 h-3 text-[var(--hud-cyan)]" />
                    <span className="text-[8px] font-mono text-[var(--hud-text-dim)] tracking-wider">AKTIF BAGLANTI</span>
                </div>
                <div className="text-lg font-mono font-bold text-[var(--hud-cyan)] tabular-nums">{currentConns}</div>
                <Sparkline data={connections} color="var(--hud-cyan)" width={100} height={20} />
            </div>
            <div className="hud-panel p-2">
                <div className="flex items-center gap-1 mb-1">
                    <Zap className="w-3 h-3 text-amber-400" />
                    <span className="text-[8px] font-mono text-[var(--hud-text-dim)] tracking-wider">ISTEK/SN</span>
                </div>
                <div className="text-lg font-mono font-bold text-amber-400 tabular-nums">{currentRps}</div>
                <Sparkline data={requests} color="#ffab00" width={100} height={20} />
            </div>
        </div>
    );
}
