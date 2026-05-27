import { useState, useEffect } from 'react';
import { Activity, Cpu, HardDrive, Wifi } from 'lucide-react';

const MetricRow = ({ icon: Icon, label, value, unit, color = 'var(--hud-cyan)' }) => (
    <div className="flex items-center justify-between py-1 border-b border-[var(--hud-border)] last:border-0">
        <div className="flex items-center gap-1.5">
            <Icon className="w-3 h-3" style={{ color }} />
            <span className="text-[10px] text-[var(--hud-text-dim)] tracking-wider">{label}</span>
        </div>
        <div className="flex items-baseline gap-0.5">
            <span className="text-[11px] font-bold tabular-nums" style={{ color }}>{value}</span>
            <span className="text-[8px] text-[var(--hud-text-dim)]">{unit}</span>
        </div>
    </div>
);

export default function SystemStatus({ className = '' }) {
    const [metrics, setMetrics] = useState({
        cpu: 34,
        memory: 62,
        disk: 45,
        network: 128,
    });

    // Simulated live metrics
    useEffect(() => {
        const id = setInterval(() => {
            setMetrics(prev => ({
                cpu: Math.min(99, Math.max(5, prev.cpu + (Math.random() - 0.5) * 8)),
                memory: Math.min(95, Math.max(20, prev.memory + (Math.random() - 0.5) * 3)),
                disk: Math.min(90, Math.max(10, prev.disk + (Math.random() - 0.5) * 1)),
                network: Math.max(0, prev.network + (Math.random() - 0.5) * 40),
            }));
        }, 2000);
        return () => clearInterval(id);
    }, []);

    return (
        <div className={`font-mono ${className}`}>
            <div className="text-[9px] text-[var(--hud-text-dim)] tracking-[0.15em] mb-1.5">SYSTEM STATUS</div>
            <MetricRow icon={Cpu} label="CPU" value={metrics.cpu.toFixed(0)} unit="%" color={metrics.cpu > 80 ? 'var(--hud-red)' : 'var(--hud-cyan)'} />
            <MetricRow icon={Activity} label="MEM" value={metrics.memory.toFixed(0)} unit="%" color={metrics.memory > 85 ? 'var(--hud-amber)' : 'var(--hud-emerald)'} />
            <MetricRow icon={HardDrive} label="DISK" value={metrics.disk.toFixed(0)} unit="%" />
            <MetricRow icon={Wifi} label="NET" value={metrics.network.toFixed(0)} unit="Mb/s" color="var(--hud-emerald)" />
        </div>
    );
}
