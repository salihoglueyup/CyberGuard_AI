import { useMemo, memo } from 'react';
import { Shield, Target, AlertTriangle, Wifi, Bug, Lock } from 'lucide-react';
import { HudRadarChart } from '../charts';

const THREAT_CATEGORIES = [
    { subject: 'DDoS', fullMark: 100 },
    { subject: 'Malware', fullMark: 100 },
    { subject: 'Phishing', fullMark: 100 },
    { subject: 'Brute Force', fullMark: 100 },
    { subject: 'XSS/SQLi', fullMark: 100 },
    { subject: 'Zero-Day', fullMark: 100 },
    { subject: 'APT', fullMark: 100 },
    { subject: 'Ransomware', fullMark: 100 },
];

function randomScore(base, variance = 20) {
    return Math.min(100, Math.max(5, base + (Math.random() - 0.5) * variance));
}

export default memo(function ThreatRadarWidget() {
    const data = useMemo(() => THREAT_CATEGORIES.map(c => ({
        ...c,
        current: Math.round(randomScore(55)),
        previous: Math.round(randomScore(45)),
    })), []);

    const topThreat = useMemo(() => {
        const sorted = [...data].sort((a, b) => b.current - a.current);
        return sorted[0];
    }, [data]);

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between px-3 pt-3 pb-1">
                <div className="flex items-center gap-2">
                    <Target className="w-4 h-4 text-[var(--hud-cyan)]" />
                    <span className="text-[11px] font-bold text-[var(--hud-cyan)] tracking-wider">TEHDIT RADAR</span>
                </div>
                <span className="text-[9px] text-[var(--hud-text-dim)] border border-[var(--hud-border)] px-1.5 py-0.5 rounded">CANLI</span>
            </div>
            <div className="flex-1 min-h-0 px-2">
                <HudRadarChart
                    data={data}
                    dataKeys={[
                        { key: 'current', name: 'Güncel', color: 'var(--hud-cyan)' },
                        { key: 'previous', name: 'Önceki', color: '#b388ff' },
                    ]}
                    categoryKey="subject"
                />
            </div>
            <div className="px-3 pb-2 flex items-center gap-2">
                <AlertTriangle className="w-3 h-3 text-[var(--hud-amber)]" />
                <span className="text-[9px] text-[var(--hud-text-dim)]">EN YÜKSEK:</span>
                <span className="text-[10px] font-bold text-[var(--hud-amber)]">{topThreat?.subject}</span>
                <span className="text-[9px] text-[var(--hud-text-dim)] ml-auto">{topThreat?.current}%</span>
            </div>
        </div>
    );
})
