import { useState, useEffect } from 'react';
import { Shield, Wifi, Database, Lock, Cloud, Server, CheckCircle, AlertTriangle } from 'lucide-react';

const STATUS_ITEMS = [
    { key: 'firewall', icon: Shield, label: 'FIREWALL' },
    { key: 'ids', icon: Wifi, label: 'IDS/IPS' },
    { key: 'database', icon: Database, label: 'DATABASE' },
    { key: 'encryption', icon: Lock, label: 'ENCRYPT' },
    { key: 'cloud', icon: Cloud, label: 'CLOUD' },
    { key: 'server', icon: Server, label: 'SERVERS' },
];

export default function StatusBar({ className = '' }) {
    const [statuses, setStatuses] = useState(() =>
        STATUS_ITEMS.reduce((acc, item) => {
            acc[item.key] = Math.random() > 0.15 ? 'ok' : 'warn';
            return acc;
        }, {})
    );

    useEffect(() => {
        const id = setInterval(() => {
            setStatuses(prev => {
                const next = { ...prev };
                const keys = Object.keys(next);
                const k = keys[Math.floor(Math.random() * keys.length)];
                next[k] = Math.random() > 0.12 ? 'ok' : 'warn';
                return next;
            });
        }, 4000);
        return () => clearInterval(id);
    }, []);

    return (
        <div className={`flex items-center gap-3 font-mono ${className}`}>
            {STATUS_ITEMS.map(({ key, icon: Icon, label }) => {
                const isOk = statuses[key] === 'ok';
                return (
                    <div
                        key={key}
                        className="flex items-center gap-1"
                        title={`${label}: ${isOk ? 'Online' : 'Warning'}`}
                    >
                        <Icon className="w-3 h-3" style={{ color: isOk ? 'var(--hud-emerald)' : 'var(--hud-amber)' }} />
                        <span className="text-[8px] tracking-wider" style={{ color: isOk ? 'var(--hud-text-dim)' : 'var(--hud-amber)' }}>
                            {label}
                        </span>
                        <div
                            className="w-1.5 h-1.5 rounded-full"
                            style={{
                                backgroundColor: isOk ? 'var(--hud-emerald)' : 'var(--hud-amber)',
                                boxShadow: `0 0 4px ${isOk ? 'rgba(0,230,118,0.5)' : 'rgba(255,171,0,0.5)'}`,
                            }}
                        />
                    </div>
                );
            })}
        </div>
    );
}
