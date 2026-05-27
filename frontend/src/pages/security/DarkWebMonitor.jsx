import React, { useState, useEffect } from 'react';
import api from '../../services/api';

const DarkWebMonitor = () => {
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState(null);
    const [breachCheck, setBreachCheck] = useState(null);
    const [alerts, setAlerts] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(false);
    const [activeTab, setActiveTab] = useState('search');

    useEffect(() => {
        loadAlerts();
        loadStats();
    }, []);

    const loadAlerts = async () => {
        try {
            const response = await api.get('/darkweb/alerts');
            setAlerts(response.data?.data?.alerts || []);
        } catch (error) {
            console.error('Error loading alerts:', error);
        }
    };

    const loadStats = async () => {
        try {
            const response = await api.get('/darkweb/stats');
            setStats(response.data.data);
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    };

    const handleSearch = async () => {
        if (!searchQuery) return;
        setLoading(true);
        try {
            const response = await api.post('/darkweb/search', {
                query: searchQuery,
                search_type: 'all'
            });
            setSearchResults(response.data.data);
        } catch (error) {
            console.error('Error searching:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleBreachCheck = async () => {
        if (!searchQuery) return;
        setLoading(true);
        try {
            const response = await api.post('/darkweb/credentials/check', {
                email: searchQuery.includes('@') ? searchQuery : null,
                domain: searchQuery.includes('.') && !searchQuery.includes('@') ? searchQuery : null,
                username: !searchQuery.includes('@') && !searchQuery.includes('.') ? searchQuery : null
            });
            setBreachCheck(response.data.data);
        } catch (error) {
            console.error('Error checking credentials:', error);
        } finally {
            setLoading(false);
        }
    };

    const getRiskColor = (level) => {
        switch (level) {
            case 'critical': return 'text-red-500 bg-red-500/20';
            case 'high': return 'text-orange-500 bg-orange-500/20';
            case 'medium': return 'text-yellow-500 bg-yellow-500/20';
            case 'safe': return 'text-green-500 bg-green-500/20';
            default: return 'text-[var(--hud-text-dim)] bg-gray-500/20';
        }
    };

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] p-6">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-xl font-semibold text-[var(--hud-text)]">Dark Web Monitor</h1>
                <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">Monitor dark web for leaked credentials and threat intelligence</p>
            </div>

            {/* Stats Cards */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Scans (24h)</p>
                        <p className="text-2xl font-bold text-purple-400">{stats.total_scans_24h}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Threats Found</p>
                        <p className="text-2xl font-bold text-red-400">{stats.threats_detected_24h}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Credentials Found</p>
                        <p className="text-2xl font-bold text-orange-400">{stats.credentials_found_24h}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Active Threats</p>
                        <p className="text-2xl font-bold text-yellow-400">{stats.active_threats}</p>
                    </div>
                </div>
            )}

            {/* Tabs */}
            <div className="flex gap-2 mb-6">
                {['search', 'breach', 'alerts'].map((tab) => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-4 py-2 rounded-lg font-medium transition ${activeTab === tab
                                ? 'bg-purple-600 text-[var(--hud-text)]'
                                : 'bg-[var(--hud-panel)] text-[var(--hud-text)] hover:bg-[var(--hud-border)]'
                            }`}
                    >
                        {tab === 'search' && '🔍 Dark Web Search'}
                        {tab === 'breach' && '🔑 Breach Check'}
                        {tab === 'alerts' && '🚨 Alerts'}
                    </button>
                ))}
            </div>

            {/* Search Bar */}
            {(activeTab === 'search' || activeTab === 'breach') && (
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)] mb-6">
                    <div className="flex gap-4">
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder={activeTab === 'search' ? "Enter email, domain, or keyword..." : "Enter email, username, or domain..."}
                            className="flex-1 px-4 py-3 bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg focus:border-purple-500 focus:outline-none"
                        />
                        <button
                            onClick={activeTab === 'search' ? handleSearch : handleBreachCheck}
                            disabled={loading || !searchQuery}
                            className="px-6 py-3 bg-purple-600 hover:bg-purple-700 disabled:opacity-50 rounded-lg font-medium"
                        >
                            {loading ? '⏳ Searching...' : activeTab === 'search' ? '🔍 Search' : '🔑 Check'}
                        </button>
                    </div>
                </div>
            )}

            {/* Search Results */}
            {activeTab === 'search' && searchResults && (
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-semibold text-purple-400">Search Results</h3>
                        <span className={`px-3 py-1 rounded-lg ${getRiskColor(searchResults.risk_score > 50 ? 'high' : 'medium')}`}>
                            Risk Score: {searchResults.risk_score}
                        </span>
                    </div>

                    <p className="text-[var(--hud-text-muted)] mb-4">
                        Found {searchResults.total_mentions} mentions across {searchResults.sources_checked} sources
                    </p>

                    <div className="space-y-3">
                        {searchResults.mentions?.map((mention, idx) => (
                            <div key={idx} className="bg-[var(--hud-panel)]/50 rounded-lg p-4 border-l-4 border-purple-500">
                                <div className="flex justify-between">
                                    <div>
                                        <p className="font-medium">{mention.source}</p>
                                        <p className="text-sm text-[var(--hud-text-muted)]">{mention.mention_type} • {mention.source_type}</p>
                                    </div>
                                    <span className={`px-2 py-1 h-fit rounded text-xs ${getRiskColor(mention.risk_level)}`}>
                                        {mention.risk_level?.toUpperCase()}
                                    </span>
                                </div>
                                <p className="text-[var(--hud-text)] mt-2 font-mono text-sm">{mention.snippet}</p>
                                <p className="text-xs text-[var(--hud-text-dim)] mt-2">
                                    Detected: {new Date(mention.detected_at).toLocaleString()} • Confidence: {(mention.confidence * 100).toFixed(0)}%
                                </p>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Breach Check Results */}
            {activeTab === 'breach' && breachCheck && (
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                    <div className="flex justify-between items-center mb-4">
                        <h3 className="text-lg font-semibold text-purple-400">Breach Check Results</h3>
                        <span className={`px-3 py-1 rounded-lg ${getRiskColor(breachCheck.risk_level)}`}>
                            {breachCheck.risk_level?.toUpperCase()}
                        </span>
                    </div>

                    {breachCheck.is_compromised ? (
                        <div className="bg-red-900/30 border border-red-500/50 rounded-lg p-4 mb-4">
                            <p className="text-red-400 font-semibold">⚠️ Compromised in {breachCheck.total_breaches} breach(es)</p>
                        </div>
                    ) : (
                        <div className="bg-green-900/30 border border-green-500/50 rounded-lg p-4 mb-4">
                            <p className="text-green-400 font-semibold">✅ No known breaches found</p>
                        </div>
                    )}

                    {breachCheck.breaches?.length > 0 && (
                        <div className="space-y-3 mt-4">
                            <h4 className="font-medium">Breach Details:</h4>
                            {breachCheck.breaches.map((breach, idx) => (
                                <div key={idx} className="bg-[var(--hud-panel)]/50 rounded-lg p-4">
                                    <p className="font-medium">{breach.breach_name}</p>
                                    <p className="text-sm text-[var(--hud-text-muted)]">Date: {breach.breach_date}</p>
                                    <div className="flex flex-wrap gap-2 mt-2">
                                        {breach.data_types?.map((type) => (
                                            <span key={type} className="px-2 py-1 bg-[var(--hud-border)] rounded text-xs">{type}</span>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}

                    {breachCheck.recommendations?.length > 0 && (
                        <div className="mt-4 bg-[var(--hud-panel)]/30 rounded-lg p-4">
                            <h4 className="font-medium mb-2">Recommendations:</h4>
                            <ul className="list-disc list-inside text-[var(--hud-text)]">
                                {breachCheck.recommendations.map((rec, idx) => (
                                    <li key={idx}>{rec}</li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}

            {/* Alerts */}
            {activeTab === 'alerts' && (
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                    <h3 className="text-lg font-semibold text-purple-400 mb-4">Active Alerts</h3>

                    {alerts.length === 0 ? (
                        <p className="text-[var(--hud-text-muted)] text-center py-8">No active alerts</p>
                    ) : (
                        <div className="space-y-3">
                            {alerts.map((alert) => (
                                <div key={alert.id} className="bg-[var(--hud-panel)]/50 rounded-lg p-4 border-l-4 border-red-500">
                                    <div className="flex justify-between">
                                        <div>
                                            <p className="font-medium">{alert.type}</p>
                                            <p className="text-sm text-[var(--hud-text-muted)]">Source: {alert.source}</p>
                                        </div>
                                        <span className={`px-2 py-1 h-fit rounded text-xs ${alert.severity === 'critical' ? 'bg-red-600' :
                                                alert.severity === 'high' ? 'bg-orange-600' : 'bg-yellow-600'
                                            }`}>
                                            {alert.severity?.toUpperCase()}
                                        </span>
                                    </div>
                                    <div className="flex justify-between mt-2 text-xs text-[var(--hud-text-dim)]">
                                        <span>Detected: {new Date(alert.detected_at).toLocaleString()}</span>
                                        <span>Status: {alert.status}</span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default DarkWebMonitor;
