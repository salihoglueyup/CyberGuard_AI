import { useMemo, memo } from 'react';
import { Network } from 'lucide-react';
import { HudSankey } from '../charts';

const ATTACK_FLOW_DATA = {
    nodes: [
        { name: 'Internet' },
        { name: 'Firewall' },
        { name: 'IDS/IPS' },
        { name: 'WAF' },
        { name: 'Engellendi' },
        { name: 'Karantina' },
        { name: 'Geçti' },
        { name: 'Analiz' },
    ],
    links: [
        { source: 0, target: 1, value: 850 },
        { source: 1, target: 4, value: 320 },
        { source: 1, target: 2, value: 530 },
        { source: 2, target: 4, value: 180 },
        { source: 2, target: 3, value: 350 },
        { source: 3, target: 4, value: 120 },
        { source: 3, target: 5, value: 80 },
        { source: 3, target: 6, value: 150 },
        { source: 5, target: 7, value: 80 },
        { source: 6, target: 7, value: 50 },
    ],
};

export default memo(function AttackFlowWidget() {
    const blockRate = useMemo(() => {
        const totalBlocked = 320 + 180 + 120;
        return Math.round((totalBlocked / 850) * 100);
    }, []);

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between px-3 pt-3 pb-1">
                <div className="flex items-center gap-2">
                    <Network className="w-4 h-4 text-[var(--hud-emerald)]" />
                    <span className="text-[11px] font-bold text-[var(--hud-emerald)] tracking-wider">SALDIRI AKIS DIAGRAMI</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[9px] text-[var(--hud-text-dim)]">ENGEL:</span>
                    <span className="text-[10px] font-bold text-[var(--hud-emerald)]">{blockRate}%</span>
                </div>
            </div>
            <div className="flex-1 min-h-0 px-1">
                <HudSankey data={ATTACK_FLOW_DATA} />
            </div>
            <div className="px-3 pb-2 grid grid-cols-3 gap-2">
                {[
                    { label: 'GELEN', value: '850', color: 'var(--hud-cyan)' },
                    { label: 'ENGEL', value: '620', color: 'var(--hud-emerald)' },
                    { label: 'GECEN', value: '150', color: 'var(--hud-amber)' },
                ].map(s => (
                    <div key={s.label} className="text-center">
                        <div className="text-[8px] text-[var(--hud-text-dim)] tracking-wider">{s.label}</div>
                        <div className="text-[11px] font-bold" style={{ color: s.color }}>{s.value}</div>
                    </div>
                ))}
            </div>
        </div>
    );
})
