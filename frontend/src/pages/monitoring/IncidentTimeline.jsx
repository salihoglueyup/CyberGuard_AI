import React, { useState, useEffect } from 'react';
import api from '../../services/api';

const IncidentTimeline = () => {
    const [incidents, setIncidents] = useState([]);
    // eslint-disable-next-line no-unused-vars
    const [stats, setStats] = useState(null);
    const [users, setUsers] = useState([]);
    const [activeTab, setActiveTab] = useState('timeline');
    const [loading, setLoading] = useState(false);
    const [filter, setFilter] = useState({ severity: '', status: '' });

    useEffect(() => {
        loadIncidents();
        loadUsers();
    }, [filter]);

    const loadIncidents = async () => {
        setLoading(true);
        try {
            let url = '/incidents/timeline?limit=50';
            if (filter.severity) url += `&severity=${filter.severity}`;
            if (filter.status) url += `&status=${filter.status}`;

            const response = await api.get(url);
            if (response.data.success) {
                setIncidents(response.data.data?.timeline || []);
            }
        } catch (error) {
            console.error('Error loading incidents:', error);
            setIncidents([]);
        } finally {
            setLoading(false);
        }
    };

    const loadUsers = async () => {
        try {
            const response = await api.get('/incidents/behavior/users?limit=20');
            if (response.data.success) {
                setUsers(response.data.data?.users || []);
            }
        } catch (error) {
            console.error('Error loading users:', error);
            setUsers([]);
        }
    };

    const getSeverityColor = (severity) => {
        const colors = {
            critical: 'bg-red-500',
            high: 'bg-orange-500',
            medium: 'bg-yellow-500',
            low: 'bg-green-500'
        };
        return colors[severity] || 'bg-gray-500';
    };

    const getStatusColor = (status) => {
        const colors = {
            open: 'text-red-400',
            investigating: 'text-[var(--hud-amber)]',
            resolved: 'text-green-400',
            closed: 'text-[var(--hud-text-muted)]'
        };
        return colors[status] || 'text-[var(--hud-text-muted)]';
    };

    const getRiskColor = (score) => {
        if (score > 70) return 'text-red-400';
        if (score > 40) return 'text-[var(--hud-amber)]';
        return 'text-green-400';
    };

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">Incident Timeline</h1>
                    <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">
                        Saldırı olayları ve kullanıcı davranış analizi
                    </p>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 mb-6">
                    <button
                        onClick={() => setActiveTab('timeline')}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === 'timeline' ? 'bg-red-600' : 'bg-[var(--hud-surface)] hover:bg-[var(--hud-panel)]'
                            }`}
                    >
                        ⏱️ Timeline
                    </button>
                    <button
                        onClick={() => setActiveTab('users')}
                        className={`px-4 py-2 rounded-lg font-medium transition-colors ${activeTab === 'users' ? 'bg-red-600' : 'bg-[var(--hud-surface)] hover:bg-[var(--hud-panel)]'
                            }`}
                    >
                        👥 User Behavior
                    </button>
                </div>

                {/* Stats */}
                {stats && activeTab === 'timeline' && (
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                        <div className="bg-[var(--hud-surface)] rounded-xl p-5 text-center">
                            <div className="text-3xl font-bold text-red-400">{stats.open}</div>
                            <div className="text-[var(--hud-text-muted)] text-sm">Open</div>
                        </div>
                        <div className="bg-[var(--hud-surface)] rounded-xl p-5 text-center">
                            <div className="text-3xl font-bold text-[var(--hud-amber)]">{stats.investigating}</div>
                            <div className="text-[var(--hud-text-muted)] text-sm">Investigating</div>
                        </div>
                        <div className="bg-[var(--hud-surface)] rounded-xl p-5 text-center">
                            <div className="text-3xl font-bold text-green-400">{stats.resolved}</div>
                            <div className="text-[var(--hud-text-muted)] text-sm">Resolved</div>
                        </div>
                        <div className="bg-[var(--hud-surface)] rounded-xl p-5 text-center">
                            <div className="text-3xl font-bold text-red-500">{stats.by_severity?.critical || 0}</div>
                            <div className="text-[var(--hud-text-muted)] text-sm">Critical</div>
                        </div>
                    </div>
                )}

                {/* Filters */}
                {activeTab === 'timeline' && (
                    <div className="bg-[var(--hud-surface)] rounded-xl p-4 mb-6 flex gap-4">
                        <select
                            value={filter.severity}
                            onChange={(e) => setFilter({ ...filter, severity: e.target.value })}
                            className="bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg px-3 py-2"
                        >
                            <option value="">All Severities</option>
                            <option value="critical">Critical</option>
                            <option value="high">High</option>
                            <option value="medium">Medium</option>
                            <option value="low">Low</option>
                        </select>
                        <select
                            value={filter.status}
                            onChange={(e) => setFilter({ ...filter, status: e.target.value })}
                            className="bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg px-3 py-2"
                        >
                            <option value="">All Status</option>
                            <option value="open">Open</option>
                            <option value="investigating">Investigating</option>
                            <option value="resolved">Resolved</option>
                            <option value="closed">Closed</option>
                        </select>
                    </div>
                )}

                {loading && (
                    <div className="flex justify-center py-12">
                        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-red-500"></div>
                    </div>
                )}

                {activeTab === 'timeline' && !loading && (
                    <div className="space-y-4">
                        {incidents?.length > 0 ? incidents.map((incident) => (
                            <div key={incident.id} className="bg-[var(--hud-surface)] rounded-xl p-5 relative">
                                <div className="absolute left-0 top-0 bottom-0 w-1 rounded-l-xl"
                                    style={{ backgroundColor: getSeverityColor(incident.severity).replace('bg-', '#') }}></div>

                                <div className="flex items-start justify-between">
                                    <div className="ml-4">
                                        <div className="flex items-center gap-3">
                                            <span className={`px-2 py-1 rounded text-xs ${getSeverityColor(incident.severity)}`}>
                                                {incident.severity?.toUpperCase()}
                                            </span>
                                            <span className={`font-bold text-lg ${getStatusColor(incident.status)}`}>
                                                {incident.title}
                                            </span>
                                        </div>
                                        <div className="text-[var(--hud-text-muted)] text-sm mt-1">
                                            ID: {incident.id} | Source: {incident.source_ip} → Target: {incident.target_ip}
                                        </div>
                                    </div>
                                    <div className="text-right">
                                        <div className="text-sm text-[var(--hud-text-muted)]">
                                            {new Date(incident.timestamp).toLocaleString()}
                                        </div>
                                        <div className={`text-sm ${getStatusColor(incident.status)}`}>
                                            {incident.status}
                                        </div>
                                    </div>
                                </div>

                                {/* Events */}
                                {incident.events && incident.events.length > 0 && (
                                    <div className="mt-4 ml-4 border-l-2 border-[var(--hud-border)] pl-4">
                                        {incident.events.slice(0, 3).map((event, i) => (
                                            <div key={i} className="flex items-center gap-2 text-sm mb-2">
                                                <div className="w-2 h-2 rounded-full bg-gray-500"></div>
                                                <span className="text-[var(--hud-text-muted)]">{event.action}</span>
                                                <span className="text-gray-600">by {event.user}</span>
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )) : (
                            <div className="text-center py-8 text-[var(--hud-text-dim)]">
                                <p>Olay bulunamadı</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Users Tab */}
                {activeTab === 'users' && !loading && (
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {users?.length > 0 ? users.map((user, index) => (
                            <div key={user.user || index} className="bg-[var(--hud-surface)] rounded-xl p-5">
                                <div className="flex items-center justify-between mb-4">
                                    <div className="flex items-center gap-3">
                                        <div className="w-10 h-10 rounded-full bg-[var(--hud-panel)] flex items-center justify-center text-lg">
                                            👤
                                        </div>
                                        <span className="font-bold">{user.user || user.user_id || 'Bilinmiyor'}</span>
                                    </div>
                                    <span className={`text-2xl font-bold ${getRiskColor(user.risk_score)}`}>
                                        {user.risk_score?.toFixed(0)}
                                    </span>
                                </div>

                                <div className="space-y-2 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-[var(--hud-text-muted)]">Logins (24h)</span>
                                        <span>{user.login_count_24h}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-[var(--hud-text-muted)]">Failed Logins</span>
                                        <span className={user.failed_logins_24h > 5 ? 'text-red-400' : ''}>
                                            {user.failed_logins_24h}
                                        </span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-[var(--hud-text-muted)]">Unique IPs</span>
                                        <span>{user.unique_ips}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-[var(--hud-text-muted)]">Unusual Hours</span>
                                        <span className={user.unusual_hours_access ? 'text-[var(--hud-amber)]' : 'text-green-400'}>
                                            {user.unusual_hours_access ? 'Yes' : 'No'}
                                        </span>
                                    </div>
                                </div>

                                {user.anomalies && user.anomalies.length > 0 && (
                                    <div className="mt-4 pt-4 border-t border-[var(--hud-border)]">
                                        <div className="text-xs text-[var(--hud-text-dim)] mb-2">Anomalies</div>
                                        {user.anomalies.slice(0, 2).map((a, i) => (
                                            <span key={i} className="inline-block bg-red-900/50 text-red-400 px-2 py-1 rounded text-xs mr-1 mb-1">
                                                {a.type}
                                            </span>
                                        ))}
                                    </div>
                                )}
                            </div>
                        )) : (
                            <div className="col-span-3 text-center py-8 text-[var(--hud-text-dim)]">
                                <p>Kullanıcı verisi bulunamadı</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default IncidentTimeline;
