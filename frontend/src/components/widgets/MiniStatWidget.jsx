import { motion } from 'framer-motion';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

function MiniTrend({ value, inverted = false }) {
    const isUp = value > 0;
    const isDown = value < 0;
    const color = inverted
        ? (isUp ? 'text-[var(--hud-red)]' : isDown ? 'text-[var(--hud-emerald)]' : 'text-[var(--hud-text-dim)]')
        : (isUp ? 'text-[var(--hud-emerald)]' : isDown ? 'text-[var(--hud-red)]' : 'text-[var(--hud-text-dim)]');
    const Icon = isUp ? TrendingUp : isDown ? TrendingDown : Minus;

    return (
        <div className={`flex items-center gap-0.5 ${color}`}>
            <Icon className="w-3 h-3" />
            <span className="text-[10px] font-bold">{Math.abs(value)}%</span>
        </div>
    );
}

export default function MiniStatWidget({ icon: Icon, label, value, unit, trend, trendInverted, color, sparkData }) {
    return (
        <motion.div
            whileHover={{ scale: 1.02 }}
            className="h-full flex flex-col justify-between p-3"
        >
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                    <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ background: `${color}15`, border: `1px solid ${color}25` }}>
                        <Icon className="w-3.5 h-3.5" style={{ color }} />
                    </div>
                    <span className="text-[9px] text-[var(--hud-text-dim)] tracking-wide">{label}</span>
                </div>
                {trend !== undefined && <MiniTrend value={trend} inverted={trendInverted} />}
            </div>

            <div className="mt-2">
                <div className="flex items-baseline gap-1">
                    <span className="text-2xl font-bold tabular-nums" style={{ color }}>{value}</span>
                    {unit && <span className="text-[10px] text-[var(--hud-text-dim)]">{unit}</span>}
                </div>
            </div>

            {/* Mini sparkline bar */}
            {sparkData && (
                <div className="flex items-end gap-px mt-2 h-4">
                    {sparkData.map((v, i) => (
                        <div
                            key={i}
                            className="flex-1 rounded-t-sm transition-all"
                            style={{
                                height: `${Math.max(8, (v / Math.max(...sparkData)) * 100)}%`,
                                background: `${color}${i === sparkData.length - 1 ? '80' : '30'}`,
                            }}
                        />
                    ))}
                </div>
            )}
        </motion.div>
    );
}
