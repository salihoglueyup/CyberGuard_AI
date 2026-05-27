import { useMemo, memo } from 'react';
import { HardDrive } from 'lucide-react';
import { HudTreemap } from '../charts';

const ASSET_DATA = [
    { name: 'Web Sunucu', size: 340, severity: 'high', color: '#ff6d00' },
    { name: 'Veritabani', size: 280, severity: 'critical', color: '#ef4444' },
    { name: 'DNS', size: 150, severity: 'medium', color: '#ffab00' },
    { name: 'Mail', size: 120, severity: 'high', color: '#ff6d00' },
    { name: 'VPN', size: 95, severity: 'low', color: 'var(--hud-cyan)' },
    { name: 'Firewall', size: 200, severity: 'medium', color: '#ffab00' },
    { name: 'Load Balancer', size: 85, severity: 'low', color: 'var(--hud-cyan)' },
    { name: 'API Gateway', size: 170, severity: 'high', color: '#ff6d00' },
    { name: 'Cache', size: 60, severity: 'low', color: '#10b981' },
    { name: 'CDN', size: 110, severity: 'medium', color: '#ffab00' },
    { name: 'SIEM', size: 75, severity: 'low', color: '#10b981' },
    { name: 'Backup', size: 45, severity: 'low', color: '#10b981' },
];

export default memo(function AssetTreemapWidget() {
    const treemapData = useMemo(() => [{
        name: 'Varliklar',
        children: ASSET_DATA.map(a => ({
            name: a.name,
            size: a.size,
            color: a.color,
        })),
    }], []);

    const critCount = ASSET_DATA.filter(a => a.severity === 'critical' || a.severity === 'high').length;

    return (
        <div className="h-full flex flex-col">
            <div className="flex items-center justify-between px-3 pt-3 pb-1">
                <div className="flex items-center gap-2">
                    <HardDrive className="w-4 h-4 text-[var(--hud-purple)]" />
                    <span className="text-[11px] font-bold text-[var(--hud-purple)] tracking-wider">VARLIK HARITASI</span>
                </div>
                <div className="flex items-center gap-2">
                    <span className="text-[9px] text-[var(--hud-text-dim)]">RISKLI:</span>
                    <span className="text-[10px] font-bold text-[var(--hud-red)]">{critCount}</span>
                </div>
            </div>
            <div className="flex-1 min-h-0 px-1">
                <HudTreemap data={treemapData} />
            </div>
            <div className="px-3 pb-2 flex items-center gap-3 text-[9px]">
                {['critical', 'high', 'medium', 'low'].map(sev => {
                    const color = sev === 'critical' ? '#ef4444' : sev === 'high' ? '#ff6d00' : sev === 'medium' ? '#ffab00' : '#10b981';
                    const count = ASSET_DATA.filter(a => a.severity === sev).length;
                    return (
                        <div key={sev} className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-sm" style={{ background: color }} />
                            <span className="text-[var(--hud-text-dim)] uppercase">{sev}</span>
                            <span className="font-bold" style={{ color }}>{count}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
})
