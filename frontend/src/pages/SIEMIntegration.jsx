import React, { useState, useEffect } from 'react';
import api from '../services/api';

const SIEMIntegration = () => {
    const [platforms, setPlatforms] = useState([]);
    const [connections, setConnections] = useState([]);
    const [rules, setRules] = useState([]);
    const [stats, setStats] = useState(null);
    const [showConnect, setShowConnect] = useState(false);
    const [connectForm, setConnectForm] = useState({
        siem_type: 'splunk',
        host: '',
        port: 443,
        api_key: ''
    });

    const loadData = async () => {
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
    };

    // eslint-disable-next-line react-hooks/exhaustive-deps
    useEffect(() => {
        loadData();
    }, []);

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
            alert(response.data.data.test_result === 'passed' ? '✅ Bağlantı başarılı!' : '❌ Bağlantı başarısız!');
        } catch (error) {
            console.error('Error testing connection:', error);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-3xl font-bold text-blue-400">🔗 SIEM Entegrasyonu</h1>
                    <p className="text-gray-400">Splunk, Elastic, QRadar bağlantıları</p>
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
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Aktif Bağlantı</p>
                        <p className="text-2xl font-bold text-blue-400">{stats.active_connections}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">İletilen Event</p>
                        <p className="text-2xl font-bold text-green-400">{stats.events_forwarded_24h?.toLocaleString()}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Başarısız</p>
                        <p className="text-2xl font-bold text-red-400">{stats.failed_forwards_24h}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Ort. Gecikme</p>
                        <p className="text-2xl font-bold text-yellow-400">{stats.avg_latency_ms}ms</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Aktif Kurallar</p>
                        <p className="text-2xl font-bold text-purple-400">{stats.rules_active}</p>
                    </div>
                </div>
            )}

            {/* Connect Form */}
            {showConnect && (
                <div className="bg-gray-800 rounded-lg p-4 border border-blue-500/50 mb-6">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">Yeni SIEM Bağlantısı</h3>
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <select
                            value={connectForm.siem_type}
                            onChange={(e) => setConnectForm({ ...connectForm, siem_type: e.target.value })}
                            className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg"
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
                            className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg"
                        />
                        <input
                            type="password"
                            placeholder="API Key"
                            value={connectForm.api_key}
                            onChange={(e) => setConnectForm({ ...connectForm, api_key: e.target.value })}
                            className="px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg"
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
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">🔌 Bağlantılar</h3>
                    <div className="space-y-3">
                        {connections.map((conn) => (
                            <div key={conn.id} className="p-4 bg-gray-700/50 rounded-lg">
                                <div className="flex justify-between items-center">
                                    <div>
                                        <p className="font-medium">{conn.name}</p>
                                        <p className="text-sm text-gray-400">{conn.siem_type}</p>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className={`px-2 py-1 rounded text-xs ${conn.status === 'connected' ? 'bg-green-600' : 'bg-red-600'
                                            }`}>{conn.status}</span>
                                        <button
                                            onClick={() => testConnection(conn.id)}
                                            className="px-3 py-1 bg-gray-600 hover:bg-gray-500 rounded text-sm"
                                        >
                                            Test
                                        </button>
                                    </div>
                                </div>
                                <div className="mt-2 text-xs text-gray-500">
                                    <span>{conn.events_forwarded_24h?.toLocaleString()} event/24s</span>
                                    <span className="mx-2">•</span>
                                    <span>Son: {new Date(conn.last_event).toLocaleString()}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Rules */}
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">📋 Forwarding Kuralları</h3>
                    <div className="space-y-3">
                        {rules.map((rule) => (
                            <div key={rule.id} className="p-4 bg-gray-700/50 rounded-lg">
                                <div className="flex justify-between items-center">
                                    <span className="font-medium">{rule.name}</span>
                                    <span className={`px-2 py-1 rounded text-xs ${rule.enabled ? 'bg-green-600' : 'bg-gray-600'}`}>
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
                <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <h3 className="text-lg font-semibold text-blue-400 mb-4">🌐 Desteklenen Platformlar</h3>
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
                        {platforms.map((platform) => (
                            <div key={platform.id} className="p-4 bg-gray-700/50 rounded-lg text-center">
                                <p className="font-medium">{platform.name}</p>
                                <p className="text-xs text-gray-400">{platform.version}</p>
                                <span className={`mt-2 inline-block px-2 py-0.5 rounded text-xs ${platform.status === 'supported' ? 'bg-green-600' :
                                    platform.status === 'beta' ? 'bg-yellow-600' : 'bg-gray-600'
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
