import { useState, useMemo } from 'react';
import { Shield, Lock, Unlock, AlertTriangle, CheckCircle2, XCircle, Globe, Key, FileText, Clock, Filter, Search, ChevronDown, ChevronRight } from 'lucide-react';

const API_ENDPOINTS = [
    { path: '/api/v1/auth/login', method: 'POST', auth: 'public', rateLimit: '10/min', status: 'secure', lastCall: '2s ago', calls24h: 4521, latency: 45 },
    { path: '/api/v1/auth/register', method: 'POST', auth: 'public', rateLimit: '5/min', status: 'secure', lastCall: '15s ago', calls24h: 892, latency: 120 },
    { path: '/api/v1/auth/token/refresh', method: 'POST', auth: 'bearer', rateLimit: '20/min', status: 'secure', lastCall: '1s ago', calls24h: 12453, latency: 22 },
    { path: '/api/v1/models', method: 'GET', auth: 'bearer', rateLimit: '60/min', status: 'secure', lastCall: '3s ago', calls24h: 8934, latency: 88 },
    { path: '/api/v1/models/{id}/predict', method: 'POST', auth: 'bearer+scope', rateLimit: '30/min', status: 'warning', lastCall: '8s ago', calls24h: 3421, latency: 250 },
    { path: '/api/v1/threats', method: 'GET', auth: 'bearer', rateLimit: '60/min', status: 'secure', lastCall: '1s ago', calls24h: 15678, latency: 35 },
    { path: '/api/v1/threats/{id}', method: 'GET', auth: 'bearer', rateLimit: '60/min', status: 'secure', lastCall: '5s ago', calls24h: 6543, latency: 42 },
    { path: '/api/v1/scan', method: 'POST', auth: 'bearer+admin', rateLimit: '5/min', status: 'secure', lastCall: '2m ago', calls24h: 234, latency: 1800 },
    { path: '/api/v1/reports', method: 'GET', auth: 'bearer', rateLimit: '30/min', status: 'secure', lastCall: '30s ago', calls24h: 2345, latency: 150 },
    { path: '/api/v1/users', method: 'GET', auth: 'bearer+admin', rateLimit: '20/min', status: 'secure', lastCall: '1m ago', calls24h: 567, latency: 65 },
    { path: '/api/v1/config', method: 'PUT', auth: 'bearer+admin', rateLimit: '10/min', status: 'critical', lastCall: '5m ago', calls24h: 23, latency: 95 },
    { path: '/api/v1/websocket', method: 'WS', auth: 'bearer', rateLimit: 'N/A', status: 'secure', lastCall: '0s ago', calls24h: 892, latency: 5 },
    { path: '/api/v1/upload', method: 'POST', auth: 'bearer+scope', rateLimit: '10/min', status: 'warning', lastCall: '15m ago', calls24h: 89, latency: 3500 },
    { path: '/api/v1/export', method: 'GET', auth: 'bearer', rateLimit: '5/min', status: 'secure', lastCall: '3m ago', calls24h: 156, latency: 2200 },
];

const VULN_FINDINGS = [
    { severity: 'critical', endpoint: '/api/v1/config', issue: 'Missing CSRF protection on state-changing endpoint', recommendation: 'Add CSRF token validation' },
    { severity: 'high', endpoint: '/api/v1/models/{id}/predict', issue: 'Input validation insufficient — potential injection', recommendation: 'Add schema validation middleware' },
    { severity: 'high', endpoint: '/api/v1/upload', issue: 'File type validation bypass possible', recommendation: 'Validate file magic bytes server-side' },
    { severity: 'medium', endpoint: '/api/v1/auth/login', issue: 'User enumeration via timing side-channel', recommendation: 'Constant-time comparison for auth' },
    { severity: 'medium', endpoint: '/api/v1/reports', issue: 'Verbose error messages in production', recommendation: 'Sanitize error responses' },
    { severity: 'low', endpoint: '/api/v1/threats', issue: 'Missing rate-limit headers in response', recommendation: 'Add X-RateLimit-* headers' },
];

const STATUS_MAP = {
    secure: { color: '#10b981', icon: CheckCircle2, label: 'Güvenli' },
    warning: { color: '#ffab00', icon: AlertTriangle, label: 'Uyarı' },
    critical: { color: '#ef4444', icon: XCircle, label: 'Kritik' },
};

const METHOD_COLORS = { GET: 'var(--hud-cyan)', POST: '#10b981', PUT: '#ffab00', DELETE: '#ef4444', WS: '#b388ff', PATCH: '#ff6d00' };

export default function ApiSecurity() {
    const [tab, setTab] = useState('endpoints');
    const [search, setSearch] = useState('');
    const [statusFilter, setStatusFilter] = useState('all');
    const [expandedEndpoint, setExpandedEndpoint] = useState(null);

    const filteredEndpoints = useMemo(() => {
        return API_ENDPOINTS.filter(ep => {
            if (statusFilter !== 'all' && ep.status !== statusFilter) return false;
            if (search && !ep.path.toLowerCase().includes(search.toLowerCase())) return false;
            return true;
        });
    }, [statusFilter, search]);

    const overallStats = useMemo(() => ({
        total: API_ENDPOINTS.length,
        secure: API_ENDPOINTS.filter(e => e.status === 'secure').length,
        warning: API_ENDPOINTS.filter(e => e.status === 'warning').length,
        critical: API_ENDPOINTS.filter(e => e.status === 'critical').length,
        totalCalls: API_ENDPOINTS.reduce((s, e) => s + e.calls24h, 0),
        avgLatency: Math.round(API_ENDPOINTS.reduce((s, e) => s + e.latency, 0) / API_ENDPOINTS.length),
    }), []);

    return (
        <div className="min-h-screen bg-[var(--hud-bg)] relative">
            <div className="border-b border-[var(--hud-border)] px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5 text-[var(--hud-cyan)]" />
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">API Security</h1>
                    <span className="text-[9px] text-[var(--hud-text-dim)] bg-[rgba(56,189,248,0.08)] border border-[var(--hud-border)] px-2 py-0.5 rounded">{API_ENDPOINTS.length} ENDPOINTS</span>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-6 gap-3 px-6 py-3 border-b border-[var(--hud-border)]">
                {[
                    { label: 'Toplam', value: overallStats.total, color: 'var(--hud-cyan)' },
                    { label: 'Güvenli', value: overallStats.secure, color: '#10b981' },
                    { label: 'Uyarı', value: overallStats.warning, color: '#ffab00' },
                    { label: 'Kritik', value: overallStats.critical, color: '#ef4444' },
                    { label: '24s Çağrı', value: overallStats.totalCalls.toLocaleString(), color: 'var(--hud-purple)' },
                    { label: 'Ort. Gecikme', value: `${overallStats.avgLatency}ms`, color: 'var(--hud-cyan)' },
                ].map(s => (
                    <div key={s.label} className="text-center">
                        <div className="text-lg font-bold tabular-nums" style={{ color: s.color }}>{s.value}</div>
                        <div className="text-[8px] text-[var(--hud-text-dim)] tracking-wider">{s.label}</div>
                    </div>
                ))}
            </div>

            {/* Tabs */}
            <div className="flex gap-1 px-6 py-2 border-b border-[var(--hud-border)]">
                {['endpoints', 'vulnerabilities'].map(t => (
                    <button key={t} onClick={() => setTab(t)}
                        className={`px-3 py-1.5 rounded text-[10px] uppercase tracking-wide transition-all ${tab === t ? 'bg-cyan-500/10 text-[var(--hud-cyan)] border border-[var(--hud-cyan)]/30' : 'text-[var(--hud-text-dim)]'}`}>
                        {t === 'endpoints' ? 'Endpoints' : 'Zaafiyetler'}
                    </button>
                ))}
            </div>

            <div className="p-6">
                {tab === 'endpoints' && (
                    <div className="space-y-3">
                        <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2 flex-1 hud-panel py-1.5 px-3">
                                <Search className="w-3 h-3 text-[var(--hud-text-dim)]" />
                                <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Endpoint ara..." className="bg-transparent text-[10px] text-[var(--hud-text)] outline-none flex-1 placeholder:text-[var(--hud-text-dim)]" />
                            </div>
                            {['all', 'secure', 'warning', 'critical'].map(s => (
                                <button key={s} onClick={() => setStatusFilter(s)}
                                    className={`px-2 py-1 rounded text-[9px] uppercase transition-all border ${statusFilter === s ? 'border-[var(--hud-cyan)]/30 text-[var(--hud-cyan)] bg-cyan-500/10' : 'border-[var(--hud-border)] text-[var(--hud-text-dim)]'}`}>
                                    {s === 'all' ? 'TUMU' : STATUS_MAP[s]?.label || s}
                                </button>
                            ))}
                        </div>
                        {filteredEndpoints.map((ep, i) => {
                            const S = STATUS_MAP[ep.status];
                            return (
                                <div key={i} className="hud-panel p-0 overflow-hidden">
                                    <button onClick={() => setExpandedEndpoint(expandedEndpoint === i ? null : i)} className="w-full flex items-center gap-4 px-4 py-2.5 hover:bg-[rgba(56,189,248,0.02)] transition-colors">
                                        {expandedEndpoint === i ? <ChevronDown className="w-3 h-3 text-[var(--hud-cyan)]" /> : <ChevronRight className="w-3 h-3 text-[var(--hud-text-dim)]" />}
                                        <span className="text-[10px] font-bold px-2 py-0.5 rounded" style={{ color: METHOD_COLORS[ep.method], backgroundColor: `${METHOD_COLORS[ep.method]}15` }}>{ep.method}</span>
                                        <span className="text-[10px] text-[var(--hud-text)] flex-1 text-left">{ep.path}</span>
                                        <span className="text-[9px] text-[var(--hud-text-dim)]">{ep.calls24h.toLocaleString()} calls</span>
                                        <span className="text-[9px] text-[var(--hud-text-dim)]">{ep.latency}ms</span>
                                        <S.icon className="w-3.5 h-3.5" style={{ color: S.color }} />
                                    </button>
                                    {expandedEndpoint === i && (
                                        <div className="border-t border-[var(--hud-border)] px-4 py-3 bg-[rgba(0,0,0,0.2)] grid grid-cols-2 md:grid-cols-4 gap-3 text-[10px]">
                                            <div><span className="text-[var(--hud-text-dim)]">Auth: </span><span className="text-[var(--hud-cyan)]">{ep.auth}</span></div>
                                            <div><span className="text-[var(--hud-text-dim)]">Rate Limit: </span><span className="text-[var(--hud-text)]">{ep.rateLimit}</span></div>
                                            <div><span className="text-[var(--hud-text-dim)]">Son Cagri: </span><span className="text-[var(--hud-text)]">{ep.lastCall}</span></div>
                                            <div><span className="text-[var(--hud-text-dim)]">Durum: </span><span style={{ color: S.color }}>{S.label}</span></div>
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                    </div>
                )}

                {tab === 'vulnerabilities' && (
                    <div className="space-y-3">
                        <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider">{VULN_FINDINGS.length} ZAFIYET TESPIT EDILDI</div>
                        {VULN_FINDINGS.map((v, i) => {
                            const color = v.severity === 'critical' ? '#ef4444' : v.severity === 'high' ? '#ff6d00' : v.severity === 'medium' ? '#ffab00' : 'var(--hud-cyan)';
                            return (
                                <div key={i} className="hud-panel">
                                    <div className="flex items-center gap-3 mb-2">
                                        <span className="text-[9px] px-1.5 py-0.5 rounded uppercase font-bold" style={{ color, backgroundColor: `${color}15` }}>{v.severity}</span>
                                        <span className="text-[10px] text-[var(--hud-cyan)]">{v.endpoint}</span>
                                    </div>
                                    <div className="text-[10px] text-[var(--hud-text)]">{v.issue}</div>
                                    <div className="text-[9px] text-emerald-400/80 mt-1">→ {v.recommendation}</div>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
}
