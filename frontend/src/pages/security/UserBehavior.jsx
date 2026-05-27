import { useState, useMemo, useEffect } from 'react';
import { Users, Activity, Clock, AlertTriangle, Shield, MapPin, Monitor, TrendingUp, Search, Filter, Eye } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const RISK_LEVELS = { critical: '#ef4444', high: '#ff6d00', medium: '#ffab00', low: '#10b981', normal: 'var(--hud-cyan)' };

const USERS_DATA = [
    { id: 'USR-001', name: 'admin_ops', role: 'Admin', dept: 'IT Ops', risk: 'medium', score: 65, sessions: 12, lastActive: '2m ago', anomalies: 3, location: 'Ankara, TR', device: 'WS-042' },
    { id: 'USR-002', name: 'j.smith', role: 'Analyst', dept: 'SOC', risk: 'low', score: 22, sessions: 8, lastActive: '5m ago', anomalies: 0, location: 'Istanbul, TR', device: 'WS-087' },
    { id: 'USR-003', name: 'db_admin', role: 'DBA', dept: 'Data', risk: 'high', score: 82, sessions: 5, lastActive: '1h ago', anomalies: 7, location: 'Izmir, TR', device: 'SRV-DB-01' },
    { id: 'USR-004', name: 'svc_backup', role: 'Service', dept: 'IT Ops', risk: 'critical', score: 95, sessions: 45, lastActive: '30s ago', anomalies: 12, location: 'Multiple', device: 'Multiple' },
    { id: 'USR-005', name: 'a.yilmaz', role: 'Developer', dept: 'Engineering', risk: 'normal', score: 15, sessions: 3, lastActive: '10m ago', anomalies: 0, location: 'Ankara, TR', device: 'WS-023' },
    { id: 'USR-006', name: 'sec_scanner', role: 'Service', dept: 'Security', risk: 'low', score: 28, sessions: 120, lastActive: '1s ago', anomalies: 1, location: 'DC-Primary', device: 'SRV-SEC-01' },
    { id: 'USR-007', name: 'm.demir', role: 'Manager', dept: 'Management', risk: 'normal', score: 10, sessions: 2, lastActive: '3h ago', anomalies: 0, location: 'Istanbul, TR', device: 'WS-101' },
    { id: 'USR-008', name: 'e.kaya', role: 'Analyst', dept: 'SOC', risk: 'medium', score: 55, sessions: 15, lastActive: '1m ago', anomalies: 2, location: 'Ankara, TR', device: 'WS-044' },
    { id: 'USR-009', name: 'net_monitor', role: 'Service', dept: 'Network', risk: 'low', score: 20, sessions: 200, lastActive: '0s ago', anomalies: 0, location: 'DC-Primary', device: 'SRV-NET-01' },
    { id: 'USR-010', name: 'guest_audit', role: 'Guest', dept: 'External', risk: 'high', score: 78, sessions: 1, lastActive: '20m ago', anomalies: 5, location: 'Unknown VPN', device: 'Unknown' },
];

const BEHAVIOR_TIMELINE = Array.from({ length: 24 }, (_, i) => ({
    hour: `${String(i).padStart(2, '0')}:00`,
    logins: Math.floor(Math.random() * 50) + 5,
    anomalies: Math.floor(Math.random() * 8),
    blocked: Math.floor(Math.random() * 3),
}));

const ANOMALY_TYPES = [
    { type: 'Olağandışı Saat Erişimi', count: 23, color: '#ef4444' },
    { type: 'Yeni Konum', count: 18, color: '#ff6d00' },
    { type: 'Toplu Veri İndirme', count: 12, color: '#ffab00' },
    { type: 'Başarısız Giriş', count: 45, color: '#b388ff' },
    { type: 'Yetki Yükseltme', count: 8, color: 'var(--hud-cyan)' },
    { type: 'Bilinmeyen Cihaz', count: 15, color: '#10b981' },
];

export default function UserBehavior() {
    const [selectedUser, setSelectedUser] = useState(null);
    const [riskFilter, setRiskFilter] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');

    const filteredUsers = useMemo(() => {
        return USERS_DATA.filter(u => {
            if (riskFilter !== 'all' && u.risk !== riskFilter) return false;
            if (searchQuery && !u.name.toLowerCase().includes(searchQuery.toLowerCase()) && !u.dept.toLowerCase().includes(searchQuery.toLowerCase())) return false;
            return true;
        });
    }, [riskFilter, searchQuery]);

    const stats = useMemo(() => ({
        total: USERS_DATA.length,
        critical: USERS_DATA.filter(u => u.risk === 'critical').length,
        high: USERS_DATA.filter(u => u.risk === 'high').length,
        totalAnomalies: USERS_DATA.reduce((s, u) => s + u.anomalies, 0),
        avgScore: Math.round(USERS_DATA.reduce((s, u) => s + u.score, 0) / USERS_DATA.length),
    }), []);

    return (
        <div className="min-h-screen bg-[var(--hud-bg)] relative">
            <div className="border-b border-[var(--hud-border)] px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Users className="w-5 h-5 text-[var(--hud-cyan)]" />
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">User Behavior Analytics</h1>
                    <span className="text-[9px] text-[var(--hud-text-dim)] bg-[rgba(56,189,248,0.08)] border border-[var(--hud-border)] px-2 py-0.5 rounded">UEBA</span>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-5 gap-3 px-6 py-3 border-b border-[var(--hud-border)]">
                {[
                    { label: 'Kullanıcı', value: stats.total, color: 'var(--hud-cyan)' },
                    { label: 'Kritik Risk', value: stats.critical, color: '#ef4444' },
                    { label: 'Yüksek Risk', value: stats.high, color: '#ff6d00' },
                    { label: 'Anomali', value: stats.totalAnomalies, color: '#ffab00' },
                    { label: 'Ort. Skor', value: stats.avgScore, color: stats.avgScore > 50 ? '#ff6d00' : '#10b981' },
                ].map(s => (
                    <div key={s.label} className="text-center">
                        <div className="text-lg font-bold tabular-nums" style={{ color: s.color }}>{s.value}</div>
                        <div className="text-[8px] text-[var(--hud-text-dim)] tracking-wider">{s.label}</div>
                    </div>
                ))}
            </div>

            <div className="p-6 space-y-6">
                {/* Charts row */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
                    <div className="lg:col-span-2 hud-panel">
                        <h3 className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mb-2">24 Saatlik Aktivite</h3>
                        <div className="h-40">
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={BEHAVIOR_TIMELINE}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(56,189,248,0.06)" />
                                    <XAxis dataKey="hour" stroke="rgba(255,255,255,0.15)" fontSize={8} />
                                    <YAxis stroke="rgba(255,255,255,0.15)" fontSize={8} />
                                    <Tooltip contentStyle={{ backgroundColor: 'var(--hud-bg)', border: '1px solid var(--hud-border)', fontSize: '10px', fontFamily: 'monospace' }} />
                                    <Area type="monotone" dataKey="logins" stroke="var(--hud-cyan)" fill="rgba(56,189,248,0.1)" />
                                    <Area type="monotone" dataKey="anomalies" stroke="#ef4444" fill="rgba(239,68,68,0.1)" />
                                    <Area type="monotone" dataKey="blocked" stroke="#ffab00" fill="rgba(255,171,0,0.1)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                    <div className="hud-panel">
                        <h3 className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mb-2">Anomali Tipleri</h3>
                        <div className="h-40">
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie data={ANOMALY_TYPES} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={55} innerRadius={25} strokeWidth={0}>
                                        {ANOMALY_TYPES.map((a, i) => <Cell key={i} fill={a.color} />)}
                                    </Pie>
                                    <Tooltip contentStyle={{ backgroundColor: 'var(--hud-bg)', border: '1px solid var(--hud-border)', fontSize: '10px', fontFamily: 'monospace' }} />
                                </PieChart>
                            </ResponsiveContainer>
                        </div>
                    </div>
                </div>

                {/* Filters */}
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 flex-1 hud-panel py-1.5 px-3">
                        <Search className="w-3 h-3 text-[var(--hud-text-dim)]" />
                        <input value={searchQuery} onChange={e => setSearchQuery(e.target.value)} placeholder="Kullanıcı ara..." className="bg-transparent text-[10px] text-[var(--hud-text)] outline-none flex-1 placeholder:text-[var(--hud-text-dim)]" />
                    </div>
                    {['all', 'critical', 'high', 'medium', 'low', 'normal'].map(r => (
                        <button key={r} onClick={() => setRiskFilter(r)}
                            className={`px-2 py-1 rounded text-[9px] uppercase transition-all border ${riskFilter === r ? 'border-[var(--hud-cyan)]/30 text-[var(--hud-cyan)] bg-cyan-500/10' : 'border-[var(--hud-border)] text-[var(--hud-text-dim)]'}`}>
                            {r === 'all' ? 'Tümü' : r}
                        </button>
                    ))}
                </div>

                {/* Users table */}
                <div className="hud-panel p-0 overflow-hidden">
                    <table className="w-full text-[10px]">
                        <thead>
                            <tr className="border-b border-[var(--hud-border)]">
                                {['Kullanıcı', 'Rol', 'Departman', 'Risk', 'Skor', 'Oturum', 'Anomali', 'Konum', 'Son Aktif'].map(h => (
                                    <th key={h} className="text-left px-3 py-2 text-[var(--hud-text-dim)] tracking-wider font-medium">{h}</th>
                                ))}
                            </tr>
                        </thead>
                        <tbody>
                            {filteredUsers.map(user => (
                                <tr key={user.id} className="border-b border-[rgba(255,255,255,0.02)] hover:bg-[rgba(56,189,248,0.02)] cursor-pointer transition-colors"
                                    onClick={() => setSelectedUser(selectedUser === user.id ? null : user.id)}>
                                    <td className="px-3 py-2 text-[var(--hud-cyan)] font-bold">{user.name}</td>
                                    <td className="px-3 py-2 text-[var(--hud-text-muted)]">{user.role}</td>
                                    <td className="px-3 py-2 text-[var(--hud-text-muted)]">{user.dept}</td>
                                    <td className="px-3 py-2"><span className="px-1.5 py-0.5 rounded uppercase text-[9px]" style={{ color: RISK_LEVELS[user.risk], backgroundColor: `${RISK_LEVELS[user.risk]}15` }}>{user.risk}</span></td>
                                    <td className="px-3 py-2 font-bold tabular-nums" style={{ color: user.score > 70 ? '#ef4444' : user.score > 40 ? '#ffab00' : '#10b981' }}>{user.score}</td>
                                    <td className="px-3 py-2 text-[var(--hud-text-muted)] tabular-nums">{user.sessions}</td>
                                    <td className="px-3 py-2 tabular-nums" style={{ color: user.anomalies > 5 ? '#ef4444' : user.anomalies > 0 ? '#ffab00' : '#10b981' }}>{user.anomalies}</td>
                                    <td className="px-3 py-2 text-[var(--hud-text-dim)]">{user.location}</td>
                                    <td className="px-3 py-2 text-[var(--hud-text-dim)]">{user.lastActive}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
