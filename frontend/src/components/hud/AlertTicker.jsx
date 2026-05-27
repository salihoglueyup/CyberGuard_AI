import { useState, useEffect, useRef, useCallback } from 'react';
import { AlertTriangle, ShieldAlert, Info, CheckCircle } from 'lucide-react';

const SEVERITY_CONFIG = {
    critical: { icon: ShieldAlert, color: 'var(--hud-red)', bg: 'rgba(255,0,60,0.06)' },
    high: { icon: AlertTriangle, color: '#ff6b35', bg: 'rgba(255,107,53,0.06)' },
    medium: { icon: Info, color: 'var(--hud-amber)', bg: 'rgba(255,171,0,0.06)' },
    low: { icon: CheckCircle, color: 'var(--hud-emerald)', bg: 'rgba(0,230,118,0.06)' },
};

const SAMPLE_ALERTS = [
    { id: 1, severity: 'critical', text: 'Brute force tespit edildi — 192.168.1.45', time: '2s' },
    { id: 2, severity: 'high', text: 'Port scan: 10.0.0.12 uzerinde 1024 port', time: '15s' },
    { id: 3, severity: 'medium', text: 'Anormal DNS trafigi: query spike +340%', time: '32s' },
    { id: 4, severity: 'critical', text: 'C2 baglanti denemesi: 203.0.113.50', time: '1m' },
    { id: 5, severity: 'low', text: 'SSL sertifika yenileme basarili', time: '2m' },
    { id: 6, severity: 'high', text: 'Privilege escalation: kullanici www-data', time: '3m' },
    { id: 7, severity: 'medium', text: 'Anomali: gece 03:00 login aktivitesi', time: '5m' },
    { id: 8, severity: 'critical', text: 'Ransomware imzasi algılandi: /tmp/enc', time: '6m' },
];

export default function AlertTicker({ maxVisible = 5, className = '' }) {
    const [alerts, setAlerts] = useState(SAMPLE_ALERTS.slice(0, maxVisible));
    const containerRef = useRef(null);

    const rotateAlert = useCallback(() => {
        setAlerts(prev => {
            const pool = SAMPLE_ALERTS;
            const next = pool[Math.floor(Math.random() * pool.length)];
            return [{ ...next, id: Date.now(), time: '0s' }, ...prev.slice(0, maxVisible - 1)];
        });
    }, [maxVisible]);

    useEffect(() => {
        const id = setInterval(rotateAlert, 5000);
        return () => clearInterval(id);
    }, [rotateAlert]);

    return (
        <div className={`font-mono ${className}`} ref={containerRef}>
            <div className="text-[9px] text-[var(--hud-text-dim)] tracking-[0.15em] mb-1.5">ALERT FEED</div>
            <div className="space-y-1">
                {alerts.map((alert, i) => {
                    const cfg = SEVERITY_CONFIG[alert.severity];
                    const Icon = cfg.icon;
                    return (
                        <div
                            key={alert.id}
                            className="flex items-start gap-1.5 px-1.5 py-1 rounded border border-transparent transition-all duration-300"
                            style={{
                                opacity: 1 - i * 0.12,
                                backgroundColor: i === 0 ? cfg.bg : 'transparent',
                                borderColor: i === 0 ? `${cfg.color}30` : 'transparent',
                            }}
                        >
                            <Icon className="w-3 h-3 flex-shrink-0 mt-0.5" style={{ color: cfg.color }} />
                            <span className="text-[10px] text-[var(--hud-text-muted)] leading-tight flex-1 truncate">
                                {alert.text}
                            </span>
                            <span className="text-[8px] text-[var(--hud-text-dim)] flex-shrink-0">{alert.time}</span>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
