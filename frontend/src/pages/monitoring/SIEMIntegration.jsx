import React, { useState, useEffect, useCallback } from 'react';
import api from '../../services/api';
import { useToast } from '../../components/ui/Toast';

const SIEMIntegration = () => {
    const [platforms, setPlatforms] = useState([]);
    const [connections, setConnections] = useState([]);
    const [rules, setRules] = useState([]);
    const [stats, setStats] = useState(null);
    const [showConnect, setShowConnect] = useState(false);
    const toast = useToast();
    const [connectForm, setConnectForm] = useState({
        siem_type: 'splunk',
        host: '',
        port: 443,
        api_key: ''
    });

    const loadData = useCallback(async () => {
        try {
            const [platformsRes, connectionsRes, rulesRes, statsRes] = await Promise.all([
                api.get('/siem/platforms'),
                api.get('/siem/connections'),
                api.get('/siem/rules'),
                api.get('/siem/stats')
            ]);
            setPlatforms(platformsRes.data.data.platforms);
            setConnections(connectionsRes.data.data.connections);
            setRules(rulesRes.data.data.rules);
            setStats(statsRes.data.data);
        } catch (error) {
            console.error('Error loading SIEM data:', error);
        }
    }, []);

    useEffect(() => {
        loadData();
    }, [loadData]);

    const handleConnect = async () => {
        try {
            await api.post('/siem/connect', connectForm);
            setShowConnect(false);
            loadData();
        } catch (error) {
            console.error('Error connecting SIEM:', error);
        }
    };

    const testConnection = async (connectionId) => {
        try {
            const response = await api.post('/siem/test', { connection_id: connectionId });
            toast[response.data.data.test_result === 'passed' ? 'success' : 'error'](response.data.data.test_result === 'passed' ? 'Bağlantı başarılı!' : 'Bağlantı başarısız!');
        } catch (error) {
            console.error('Error testing connection:', error);
        }
    };

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] p-6">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">SIEM Entegrasyonu</h1>
                    <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">Splunk, Elastic, QRadar bağlantıları</p>
                </div>
                <button
                    onClick={() => setShowConnect(!showConnect)}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium"
                >
                    + Bağlantı Ekle
                </button>
            </div>

            {/* Stats */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Aktif Bağlantı</p>
                        <p className="text-2xl font-bold text-blue-400">{stats.active_connections}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">İletilen Event</p>
                        <p className="text-2xl font-bold text-green-400">{stats.events_forwarded_24h?.toLocaleString()}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Başarısız</p>
                        <p className="text-2xl font-bold text-red-400">{stats.failed_forwards_24h}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Ort. Gecikme</p>
                        <p className="text-2xl font-bold text-yellow-400">{stats.avg_latency_ms}ms</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Aktif Kurallar</p>
                        <p className="text-2xl font-bold text-purple-400">{stats.rules_active}</p>
                    </div>
                </div>
            )}

            {/* Connect Form */}
            {showConnect && (
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-blue-500/50 mb-6">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">Yeni SIEM Bağlantısı</h3>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <select
                            value={connectForm.siem_type}
                            onChange={(e) => setConnectForm({ ...connectForm, siem_type: e.target.value })}
                            className="px-4 py-2 bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg"
                        >
                            {platforms.map((p) => (
                                <option key={p.id} value={p.id}>{p.name}</option>
                            ))}
                        </select>
                        <input
                            type="text"
                            placeholder="Host (örn: siem.company.com)"
                            value={connectForm.host}
                            onChange={(e) => setConnectForm({ ...connectForm, host: e.target.value })}
                            className="px-4 py-2 bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg"
                        />
                        <input
                            type="password"
                            placeholder="API Key"
                            value={connectForm.api_key}
                            onChange={(e) => setConnectForm({ ...connectForm, api_key: e.target.value })}
                            className="px-4 py-2 bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg"
                        />
                        <button
                            onClick={handleConnect}
                            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg font-medium"
                        >
                            Bağlan
                        </button>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Connections */}
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">🔌 Bağlantılar</h3>
                    <div className="space-y-3">
                        {connections.map((conn) => (
                            <div key={conn.id} className="p-4 bg-[var(--hud-panel)]/50 rounded-lg">
                                <div className="flex justify-between items-center">
                                    <div>
                                        <p className="font-medium">{conn.name}</p>
                                        <p className="text-sm text-[var(--hud-text-muted)]">{conn.siem_type}</p>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className={`px-2 py-1 rounded text-xs ${conn.status === 'connected' ? 'bg-green-600' : 'bg-red-600'
                                            }`}>{conn.status}</span>
                                        <button
                                            onClick={() => testConnection(conn.id)}
                                            className="px-3 py-1 bg-[var(--hud-border)] hover:bg-[var(--hud-panel)] rounded text-sm"
                                        >
                                            Test
                                        </button>
                                    </div>
                                </div>
                                <div className="mt-2 text-xs text-[var(--hud-text-dim)]">
                                    <span>{conn.events_forwarded_24h?.toLocaleString()} event/24s</span>
                                    <span className="mx-2">•</span>
                                    <span>Son: {new Date(conn.last_event).toLocaleString()}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Rules */}
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">📋 Forwarding Kuralları</h3>
                    <div className="space-y-3">
                        {rules.map((rule) => (
                            <div key={rule.id} className="p-4 bg-[var(--hud-panel)]/50 rounded-lg">
                                <div className="flex justify-between items-center">
                                    <span className="font-medium">{rule.name}</span>
                                    <span className={`px-2 py-1 rounded text-xs ${rule.enabled ? 'bg-green-600' : 'bg-[var(--hud-border)]'}`}>
                                        {rule.enabled ? 'Aktif' : 'Pasif'}
                                    </span>
                                </div>
                                <div className="flex gap-2 mt-2">
                                    {rule.event_types?.map((type) => (
                                        <span key={type} className="px-2 py-0.5 bg-blue-600/30 rounded text-xs">{type}</span>
                                    ))}
                                    <span className="px-2 py-0.5 bg-yellow-600/30 rounded text-xs">≥ {rule.severity}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Platforms */}
                <div className="lg:col-span-2 bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">🌐 Desteklenen Platformlar</h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                        {platforms.map((platform) => (
                            <div key={platform.id} className="p-4 bg-[var(--hud-panel)]/50 rounded-lg text-center">
                                <p className="font-medium">{platform.name}</p>
                                <p className="text-xs text-[var(--hud-text-muted)]">{platform.version}</p>
                                <span className={`mt-2 inline-block px-2 py-0.5 rounded text-xs ${platform.status === 'supported' ? 'bg-green-600' :
                                    platform.status === 'beta' ? 'bg-yellow-600' : 'bg-[var(--hud-border)]'
                                    }`}>{platform.status}</span>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SIEMIntegration;
