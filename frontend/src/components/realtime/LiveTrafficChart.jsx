import { useMemo } from 'react';
import { useRealtimeStore } from '../../hooks/useRealtimeMetrics';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Activity } from 'lucide-react';

export default function LiveTrafficChart() {
    const bwIn = useRealtimeStore(s => s.bandwidth.inbound);
    const bwOut = useRealtimeStore(s => s.bandwidth.outbound);
    const currentIn = useRealtimeStore(s => s.currentBandwidthIn);
    const currentOut = useRealtimeStore(s => s.currentBandwidthOut);

    const chartData = useMemo(() => {
        const len = Math.max(bwIn.length, bwOut.length);
        return Array.from({ length: len }, (_, i) => ({
            time: bwIn[i]?.time || bwOut[i]?.time || '',
            inbound: bwIn[i]?.value || 0,
            outbound: bwOut[i]?.value || 0,
        }));
    }, [bwIn, bwOut]);

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                    <Activity className="w-3.5 h-3.5 text-[var(--hud-cyan)]" />
                    <span className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide">AG TRAFIGI (CANLI)</span>
                    <span className="text-[8px] text-emerald-400 animate-pulse font-mono ml-1">● LIVE</span>
                </div>
                <div className="flex gap-3 text-[9px] font-mono">
                    <span className="flex items-center gap-1">
                        <span className="w-2 h-1 bg-emerald-400 rounded-sm" /> GELEN: <span className="text-emerald-400 font-bold tabular-nums">{Math.round(currentIn)} Mbps</span>
                    </span>
                    <span className="flex items-center gap-1">
                        <span className="w-2 h-1 bg-amber-400 rounded-sm" /> GIDEN: <span className="text-amber-400 font-bold tabular-nums">{Math.round(currentOut)} Mbps</span>
                    </span>
                </div>
            </div>

            <div className="flex-1 min-h-0" style={{ minHeight: '100px' }}>
                {chartData.length > 2 ? (
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                            <defs>
                                <linearGradient id="liveGradIn" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                                    <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="liveGradOut" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#ffab00" stopOpacity={0.2} />
                                    <stop offset="95%" stopColor="#ffab00" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(56,189,248,0.06)" />
                            <XAxis
                                dataKey="time"
                                stroke="rgba(255,255,255,0.1)"
                                fontSize={8}
                                fontFamily="JetBrains Mono, monospace"
                                interval="preserveStartEnd"
                                tickCount={5}
                            />
                            <YAxis
                                stroke="rgba(255,255,255,0.1)"
                                fontSize={8}
                                fontFamily="JetBrains Mono, monospace"
                                width={35}
                                tickFormatter={v => `${v}`}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: 'var(--hud-bg)',
                                    border: '1px solid var(--hud-border)',
                                    borderRadius: '4px',
                                    fontFamily: 'JetBrains Mono, monospace',
                                    fontSize: '10px',
                                }}
                                labelStyle={{ color: 'var(--hud-text-dim)', fontSize: '9px' }}
                                formatter={(value, name) => [
                                    `${Math.round(value)} Mbps`,
                                    name === 'inbound' ? 'Gelen' : 'Giden'
                                ]}
                            />
                            <Area
                                type="monotone"
                                dataKey="inbound"
                                stroke="#10b981"
                                strokeWidth={1.5}
                                fillOpacity={1}
                                fill="url(#liveGradIn)"
                                isAnimationActive={false}
                            />
                            <Area
                                type="monotone"
                                dataKey="outbound"
                                stroke="#ffab00"
                                strokeWidth={1.5}
                                fillOpacity={1}
                                fill="url(#liveGradOut)"
                                isAnimationActive={false}
                            />
                        </AreaChart>
                    </ResponsiveContainer>
                ) : (
                    <div className="h-full flex items-center justify-center text-[var(--hud-text-dim)] font-mono text-xs">
                        VERI TOPLANYOR...
                    </div>
                )}
            </div>
        </div>
    );
}
