import React, { useState, useEffect } from 'react';
import api from '../../services/api';

const ThreatHunting = () => {
    const [query, setQuery] = useState('');
    const [timeRange, setTimeRange] = useState('24h');
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [templates, setTemplates] = useState([]);
    const [investigations, setInvestigations] = useState([]);

    useEffect(() => {
        loadTemplates();
        loadInvestigations();
    }, []);

    const loadTemplates = async () => {
        try {
            const response = await api.get('/threat-hunting/templates');
            setTemplates(response.data.data.templates);
        } catch (error) {
            console.error('Error loading templates:', error);
        }
    };

    const loadInvestigations = async () => {
        try {
            const response = await api.get('/threat-hunting/investigations');
            setInvestigations(response.data.data.investigations);
        } catch (error) {
            console.error('Error loading investigations:', error);
        }
    };

    const executeHunt = async () => {
        if (!query) return;
        setLoading(true);
        try {
            const response = await api.post('/threat-hunting/query', {
                query,
                time_range: timeRange,
                data_sources: ['network', 'endpoint', 'logs']
            });
            setResults(response.data.data);
        } catch (error) {
            console.error('Error executing hunt:', error);
        } finally {
            setLoading(false);
        }
    };

    const applyTemplate = (template) => {
        setQuery(template.query);
    };

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] p-6">
            <div className="mb-6">
                <h1 className="text-xl font-semibold text-[var(--hud-text)]">Threat Hunting</h1>
                <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">Proaktif tehdit arama ve soruşturma</p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Query Builder */}
                <div className="lg:col-span-2 space-y-6">
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <h3 className="text-lg font-semibold text-green-400 mb-4">🎯 Hunt Query</h3>

                        <textarea
                            value={query}
                            onChange={(e) => setQuery(e.target.value)}
                            placeholder="event_type:authentication AND status:failed AND count > 5"
                            className="w-full h-32 px-4 py-3 bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg focus:border-green-500 focus:outline-none font-mono text-sm"
                        />

                        <div className="flex gap-4 mt-4">
                            <select
                                value={timeRange}
                                onChange={(e) => setTimeRange(e.target.value)}
                                className="px-4 py-2 bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg"
                            >
                                <option value="1h">Son 1 saat</option>
                                <option value="24h">Son 24 saat</option>
                                <option value="7d">Son 7 gün</option>
                                <option value="30d">Son 30 gün</option>
                            </select>

                            <button
                                onClick={executeHunt}
                                disabled={loading || !query}
                                className="flex-1 px-4 py-2 bg-green-600 hover:bg-green-700 disabled:opacity-50 rounded-lg font-medium"
                            >
                                {loading ? '⏳ Aranıyor...' : '🔍 Hunt Başlat'}
                            </button>
                        </div>
                    </div>

                    {/* Results */}
                    {results && (
                        <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                            <div className="flex justify-between items-center mb-4">
                                <h3 className="text-lg font-semibold text-green-400">📊 Sonuçlar</h3>
                                <span className="text-[var(--hud-text-muted)]">{results.hunt?.total_matches || 0} kayıt bulundu</span>
                            </div>

                            <div className="grid grid-cols-2 gap-3 mb-4">
                                <div className="bg-blue-900/30 rounded-lg p-3 text-center">
                                    <p className="text-2xl font-bold text-blue-400">{results.results?.logs?.length || 0}</p>
                                    <p className="text-xs text-[var(--hud-text-muted)]">Log Eşleşmesi</p>
                                </div>
                                <div className="bg-purple-900/30 rounded-lg p-3 text-center">
                                    <p className="text-2xl font-bold text-purple-400">{results.results?.data?.length || 0}</p>
                                    <p className="text-xs text-[var(--hud-text-muted)]">Veri Eşleşmesi</p>
                                </div>
                            </div>

                            <div className="space-y-2 max-h-[300px] overflow-y-auto">
                                {/* Log Results */}
                                {results.results?.logs?.slice(0, 10).map((r, i) => (
                                    <div key={`log-${i}`} className="p-3 rounded-lg bg-[var(--hud-panel)]/50 border-l-4 border-blue-500">
                                        <div className="flex justify-between">
                                            <span className="font-medium text-blue-400">{r.file}</span>
                                            <span className="text-xs text-[var(--hud-text-muted)]">Satır: {r.line_number}</span>
                                        </div>
                                        <p className="text-sm text-[var(--hud-text-muted)] font-mono truncate">{r.content}</p>
                                    </div>
                                ))}
                                {/* Data Results */}
                                {results.results?.data?.slice(0, 10).map((r, i) => (
                                    <div key={`data-${i}`} className="p-3 rounded-lg bg-[var(--hud-panel)]/50 border-l-4 border-purple-500">
                                        <div className="flex justify-between">
                                            <span className="font-medium text-purple-400">{r.file}</span>
                                            <span className="text-xs text-[var(--hud-text-muted)]">{r.match_type}</span>
                                        </div>
                                        <p className="text-sm text-[var(--hud-text-muted)] font-mono truncate">{r.preview}</p>
                                    </div>
                                ))}
                                {(!results.results?.logs?.length && !results.results?.data?.length) && (
                                    <div className="text-center py-4 text-[var(--hud-text-dim)]">
                                        Eşleşme bulunamadı
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>

                {/* Sidebar */}
                <div className="space-y-6">
                    {/* Templates */}
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <h3 className="text-lg font-semibold text-green-400 mb-4">📋 Şablonlar</h3>
                        <div className="space-y-2">
                            {templates.map((t) => (
                                <button
                                    key={t.id}
                                    onClick={() => applyTemplate(t)}
                                    className="w-full text-left p-3 bg-[var(--hud-panel)]/50 hover:bg-[var(--hud-panel)] rounded-lg transition"
                                >
                                    <p className="font-medium">{t.name}</p>
                                    <p className="text-xs text-[var(--hud-text-muted)]">{t.description}</p>
                                    <div className="flex gap-1 mt-1">
                                        {t.mitre?.map((m) => (
                                            <span key={m} className="px-1 py-0.5 bg-green-600/30 rounded text-xs">{m}</span>
                                        ))}
                                    </div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Investigations */}
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <h3 className="text-lg font-semibold text-green-400 mb-4">🔬 Soruşturmalar</h3>
                        <div className="space-y-2">
                            {investigations?.map((inv) => (
                                <div key={inv.id} className="p-3 bg-[var(--hud-panel)]/50 rounded-lg">
                                    <div className="flex justify-between">
                                        <span className="font-medium">{inv.name}</span>
                                        <span className={`px-2 py-0.5 rounded text-xs ${inv.status === 'active' ? 'bg-green-600' :
                                            inv.status === 'pending' ? 'bg-yellow-600' : 'bg-[var(--hud-border)]'
                                            }`}>{inv.status}</span>
                                    </div>
                                    <p className="text-xs text-[var(--hud-text-muted)] mt-1">{inv.matches || 0} bulgu • {inv.assignee || 'Atanmadı'}</p>
                                </div>
                            ))}
                            {(!investigations || investigations.length === 0) && (
                                <p className="text-[var(--hud-text-dim)] text-center py-2">Soruşturma bulunamadı</p>
                            )}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ThreatHunting;
