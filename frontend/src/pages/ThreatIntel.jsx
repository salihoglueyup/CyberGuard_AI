import { useState, useEffect } from 'react';
import {
    Radar, Globe, Shield, AlertTriangle, RefreshCw, ExternalLink,
    TrendingUp, Clock, MapPin, Flag, Activity, Zap, Brain, Target,
    BarChart3, PieChart, Filter, Search, Radio, Plus, Trash2, Eye,
    Hash, Link, Server, CheckCircle, XCircle
} from 'lucide-react';
import { Card, Button, Badge } from '../components/ui';
import { attacksApi, modelsApi } from '../services/api';
import api from '../services/api';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart as RechartsPie, Pie, Cell, Legend
} from 'recharts';

const API_BASE = 'http://localhost:8000/api';

// Renk paleti
const COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];

const SEVERITY_COLORS = {
    critical: 'bg-red-500/20 text-red-400 border-red-500/50',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
    low: 'bg-green-500/20 text-green-400 border-green-500/50',
};

export default function ThreatIntel() {
    const [activeTab, setActiveTab] = useState('overview');
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState(null);
    const [attacks, setAttacks] = useState([]);
    const [stats, setStats] = useState(null);
    const [typeData, setTypeData] = useState([]);
    const [severityData, setSeverityData] = useState([]);
    const [topIps, setTopIps] = useState([]);
    const [loading, setLoading] = useState(true);
    const [analyzing, setAnalyzing] = useState(false);
    const [analysisResults, setAnalysisResults] = useState([]);
    const [hours, setHours] = useState(24);

    // MITRE ATT&CK
    const [mitreTactics, setMitreTactics] = useState([]);
    const [mitreMapping, setMitreMapping] = useState(null);

    // IOC
    const [iocs, setIocs] = useState([]);
    const [showAddIoc, setShowAddIoc] = useState(false);
    const [newIoc, setNewIoc] = useState({ type: 'ip', value: '', severity: 'medium', description: '' });

    // Real-time IDS
    const [idsStatus, setIdsStatus] = useState(null);
    const [idsAlerts, setIdsAlerts] = useState([]);

    // IP Reputation
    const [ipLookup, setIpLookup] = useState('');
    const [ipReputation, setIpReputation] = useState(null);

    useEffect(() => {
        loadModels();
        loadData();
        loadMitre();
        loadIocs();
        loadIdsStatus();
    }, [hours]);

    useEffect(() => {
        // IDS status polling
        let interval;
        if (idsStatus?.status === 'running') {
            interval = setInterval(loadIdsStatus, 3000);
        }
        return () => clearInterval(interval);
    }, [idsStatus?.status]);

    const loadModels = async () => {
        try {
            const res = await modelsApi.getAll();
            if (res.data.success) {
                setModels(res.data.data);
                const deployed = res.data.data.find(m => m.status === 'deployed');
                if (deployed) setSelectedModel(deployed);
                else if (res.data.data.length > 0) setSelectedModel(res.data.data[0]);
            }
        } catch (error) {
            console.error('Models load error:', error);
        }
    };

    const loadData = async () => {
        setLoading(true);
        try {
            const modelSamples = selectedModel
                ? (selectedModel.train_samples || 0) + (selectedModel.test_samples || 0)
                : 1000;
            const fetchLimit = Math.min(modelSamples || 1000, 200000);

            const [attacksRes, statsRes, typeRes, sevRes, ipsRes] = await Promise.all([
                attacksApi.getAll(1, fetchLimit, hours),
                attacksApi.getStats(hours),
                attacksApi.getByType(hours),
                attacksApi.getBySeverity(hours),
                attacksApi.getTopIps(10, hours)
            ]);

            if (attacksRes.data.success) setAttacks(attacksRes.data.data || []);
            if (statsRes.data.success) setStats(statsRes.data.data);

            if (typeRes.data.success) {
                const typeDataRaw = typeRes.data.data || {};
                setTypeData(Object.entries(typeDataRaw).map(([name, value], idx) => ({
                    name: String(name),
                    value: typeof value === 'number' ? value : (typeof value === 'object' ? (value?.value || value?.count || 0) : Number(value) || 0),
                    color: COLORS[idx % COLORS.length]
                })));
            }

            if (sevRes.data.success) {
                const sevDataRaw = sevRes.data.data || {};
                setSeverityData(Object.entries(sevDataRaw).map(([name, value]) => ({
                    name: String(name),
                    value: typeof value === 'number' ? value : (typeof value === 'object' ? (value?.value || value?.count || 0) : Number(value) || 0)
                })));
            }

            if (ipsRes.data.success) setTopIps(ipsRes.data.data || []);
        } catch (error) {
            console.error('Data load error:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadMitre = async () => {
        try {
            const [tacticsRes, mappingRes] = await Promise.all([
                api.get('/threat-analysis/mitre/tactics'),
                api.get(`/threat-analysis/mitre/mapping?hours=${hours}`)
            ]);
            if (tacticsRes.data.success) setMitreTactics(tacticsRes.data.data);
            if (mappingRes.data.success) setMitreMapping(mappingRes.data.data);
        } catch (error) {
            console.error('MITRE load error:', error);
        }
    };

    const loadIocs = async () => {
        try {
            const res = await api.get('/threat-analysis/ioc');
            if (res.data.success) setIocs(res.data.data);
        } catch (error) {
            console.error('IOC load error:', error);
        }
    };

    const loadIdsStatus = async () => {
        try {
            const [statusRes, alertsRes] = await Promise.all([
                api.get('/threat-analysis/realtime-ids/status'),
                api.get('/threat-analysis/realtime-ids/alerts?limit=10')
            ]);
            if (statusRes.data.success) setIdsStatus(statusRes.data.data);
            if (alertsRes.data.success) setIdsAlerts(alertsRes.data.data);
        } catch (error) {
            console.error('IDS status error:', error);
        }
    };

    const addIoc = async () => {
        try {
            const res = await api.post('/threat-analysis/ioc', newIoc);
            if (res.data.success) {
                setIocs([...iocs, res.data.data]);
                setShowAddIoc(false);
                setNewIoc({ type: 'ip', value: '', severity: 'medium', description: '' });
            }
        } catch (error) {
            console.error('IOC add error:', error);
        }
    };

    const deleteIoc = async (iocId) => {
        try {
            await api.delete(`/threat-analysis/ioc/${iocId}`);
            setIocs(iocs.filter(i => i.id !== iocId));
        } catch (error) {
            console.error('IOC delete error:', error);
        }
    };

    const lookupIp = async () => {
        if (!ipLookup) return;
        try {
            const res = await api.get(`/threat-analysis/ip-reputation/${ipLookup}`);
            if (res.data.success) setIpReputation(res.data.data);
        } catch (error) {
            console.error('IP lookup error:', error);
        }
    };

    const analyzeThreats = async () => {
        if (!selectedModel || attacks.length === 0) return;
        setAnalyzing(true);
        setAnalysisResults([]);

        try {
            const res = await fetch(`${API_BASE}/prediction/batch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    data: attacks.slice(0, 50).map(a => ({
                        source_ip: a.source_ip,
                        destination_ip: a.destination_ip || '10.0.0.1',
                        source_port: a.source_port || 0,
                        destination_port: a.destination_port || 80,
                        protocol: a.protocol || 'TCP',
                        packet_size: a.packet_size || 1024,
                        severity: a.severity,
                        attack_type: a.attack_type
                    })),
                    model_id: selectedModel.id
                })
            });
            const data = await res.json();
            if (data.success) {
                setAnalysisResults(data.data.predictions || []);
            }
        } catch (error) {
            console.error('Analysis error:', error);
        } finally {
            setAnalyzing(false);
        }
    };

    const criticalCount = attacks.filter(a => a.severity === 'critical').length;
    const highCount = attacks.filter(a => a.severity === 'high').length;
    const blockedCount = attacks.filter(a => a.blocked).length;

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    // Tab Components
    const OverviewTab = () => (
        <div className="space-y-6">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="bg-gradient-to-br from-red-600/20 to-red-900/10 border-red-500/30">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-red-400 text-sm">Toplam Saldırı</p>
                            <p className="text-3xl font-bold text-white">{stats?.total || 0}</p>
                        </div>
                        <AlertTriangle className="w-8 h-8 text-red-400" />
                    </div>
                </Card>

                <Card className="bg-gradient-to-br from-orange-600/20 to-orange-900/10 border-orange-500/30">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-orange-400 text-sm">Kritik + Yüksek</p>
                            <p className="text-3xl font-bold text-white">{criticalCount + highCount}</p>
                        </div>
                        <Zap className="w-8 h-8 text-orange-400" />
                    </div>
                </Card>

                <Card className="bg-gradient-to-br from-green-600/20 to-green-900/10 border-green-500/30">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-green-400 text-sm">Engellenen</p>
                            <p className="text-3xl font-bold text-white">{blockedCount}</p>
                        </div>
                        <Shield className="w-8 h-8 text-green-400" />
                    </div>
                </Card>

                <Card className="bg-gradient-to-br from-blue-600/20 to-blue-900/10 border-blue-500/30">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-blue-400 text-sm">Benzersiz IP</p>
                            <p className="text-3xl font-bold text-white">{topIps.length}</p>
                        </div>
                        <Globe className="w-8 h-8 text-blue-400" />
                    </div>
                </Card>
            </div>

            {/* Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <PieChart className="w-5 h-5 text-blue-400" />
                        Saldırı Tipleri
                    </h3>
                    <div className="h-64">
                        {typeData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <RechartsPie>
                                    <Pie
                                        data={typeData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={50}
                                        outerRadius={80}
                                        dataKey="value"
                                        label={({ name, value }) => `${name}: ${value}`}
                                    >
                                        {typeData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                    <Tooltip />
                                    <Legend />
                                </RechartsPie>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex items-center justify-center h-full text-slate-400">Veri yok</div>
                        )}
                    </div>
                </Card>

                <Card>
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                        <BarChart3 className="w-5 h-5 text-orange-400" />
                        Şiddet Dağılımı
                    </h3>
                    <div className="h-64">
                        {severityData.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={severityData}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                    <XAxis dataKey="name" stroke="#94a3b8" />
                                    <YAxis stroke="#94a3b8" />
                                    <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155' }} />
                                    <Bar dataKey="value" fill="#f97316" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex items-center justify-center h-full text-slate-400">Veri yok</div>
                        )}
                    </div>
                </Card>
            </div>

            {/* Top Attackers */}
            <Card>
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Flag className="w-5 h-5 text-red-400" />
                    En Aktif Saldırganlar
                </h3>
                <div className="space-y-2">
                    {topIps.slice(0, 5).map((ip, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                            <div className="flex items-center gap-3">
                                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${idx === 0 ? 'bg-red-500' : idx === 1 ? 'bg-orange-500' : 'bg-slate-600'} text-white`}>
                                    {idx + 1}
                                </span>
                                <span className="text-white font-mono">{ip.source_ip}</span>
                            </div>
                            <Badge variant="danger">{ip.attack_count} saldırı</Badge>
                        </div>
                    ))}
                </div>
            </Card>
        </div>
    );

    const MitreTab = () => (
        <div className="space-y-6">
            <Card>
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Target className="w-5 h-5 text-purple-400" />
                    MITRE ATT&CK Mapping
                </h3>
                {mitreMapping && (
                    <div className="mb-4 text-sm text-gray-400">
                        {mitreMapping.total_mapped} / {mitreMapping.total_attacks} saldırı eşleştirildi
                    </div>
                )}
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                    {mitreTactics.map(tactic => {
                        const mapping = mitreMapping?.mapping?.find(m => m.name === tactic.name);
                        const count = mapping?.count || 0;
                        return (
                            <div
                                key={tactic.id}
                                className={`p-3 rounded-lg border transition-all cursor-pointer hover:scale-105 ${count > 0 ? 'border-purple-500 bg-purple-500/10' : 'border-gray-700 bg-gray-800/30'}`}
                                style={{ borderLeftColor: tactic.color, borderLeftWidth: '4px' }}
                            >
                                <h4 className="font-medium text-white text-sm">{tactic.name_tr}</h4>
                                <p className="text-xs text-gray-400">{tactic.id}</p>
                                <p className="text-lg font-bold mt-1" style={{ color: tactic.color }}>{count}</p>
                            </div>
                        );
                    })}
                </div>
            </Card>
        </div>
    );

    const IocTab = () => (
        <div className="space-y-6">
            <Card>
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                        <Hash className="w-5 h-5 text-red-400" />
                        IOC (Indicator of Compromise)
                    </h3>
                    <Button onClick={() => setShowAddIoc(true)} size="sm">
                        <Plus className="w-4 h-4 mr-1" />
                        IOC Ekle
                    </Button>
                </div>

                {showAddIoc && (
                    <div className="mb-4 p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                        <div className="grid grid-cols-4 gap-3 mb-3">
                            <select
                                value={newIoc.type}
                                onChange={(e) => setNewIoc({ ...newIoc, type: e.target.value })}
                                className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                            >
                                <option value="ip">IP</option>
                                <option value="domain">Domain</option>
                                <option value="hash">Hash</option>
                                <option value="url">URL</option>
                            </select>
                            <input
                                type="text"
                                placeholder="Değer..."
                                value={newIoc.value}
                                onChange={(e) => setNewIoc({ ...newIoc, value: e.target.value })}
                                className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                            />
                            <select
                                value={newIoc.severity}
                                onChange={(e) => setNewIoc({ ...newIoc, severity: e.target.value })}
                                className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                            >
                                <option value="low">Low</option>
                                <option value="medium">Medium</option>
                                <option value="high">High</option>
                                <option value="critical">Critical</option>
                            </select>
                            <input
                                type="text"
                                placeholder="Açıklama..."
                                value={newIoc.description}
                                onChange={(e) => setNewIoc({ ...newIoc, description: e.target.value })}
                                className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                            />
                        </div>
                        <div className="flex gap-2">
                            <Button onClick={addIoc} size="sm">Ekle</Button>
                            <Button onClick={() => setShowAddIoc(false)} variant="outline" size="sm">İptal</Button>
                        </div>
                    </div>
                )}

                <div className="space-y-2">
                    {iocs.length > 0 ? iocs.map(ioc => (
                        <div key={ioc.id} className="flex items-center justify-between p-3 bg-gray-800/30 rounded-lg">
                            <div className="flex items-center gap-3">
                                {ioc.type === 'ip' && <Server className="w-4 h-4 text-blue-400" />}
                                {ioc.type === 'domain' && <Globe className="w-4 h-4 text-green-400" />}
                                {ioc.type === 'hash' && <Hash className="w-4 h-4 text-purple-400" />}
                                {ioc.type === 'url' && <Link className="w-4 h-4 text-orange-400" />}
                                <span className="text-white font-mono">{ioc.value}</span>
                                <Badge variant={ioc.severity === 'critical' ? 'danger' : ioc.severity === 'high' ? 'warning' : 'default'} size="sm">
                                    {ioc.severity}
                                </Badge>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="text-xs text-gray-400">{ioc.hits || 0} hit</span>
                                <button onClick={() => deleteIoc(ioc.id)} className="p-1 hover:bg-red-500/20 rounded">
                                    <Trash2 className="w-4 h-4 text-red-400" />
                                </button>
                            </div>
                        </div>
                    )) : (
                        <p className="text-center text-gray-500 py-4">Henüz IOC eklenmedi</p>
                    )}
                </div>
            </Card>

            {/* IP Reputation Lookup */}
            <Card>
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Eye className="w-5 h-5 text-blue-400" />
                    IP Reputation Lookup
                </h3>
                <div className="flex gap-2 mb-4">
                    <input
                        type="text"
                        placeholder="IP adresi girin..."
                        value={ipLookup}
                        onChange={(e) => setIpLookup(e.target.value)}
                        onKeyDown={(e) => e.key === 'Enter' && lookupIp()}
                        className="flex-1 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                    />
                    <Button onClick={lookupIp}>Sorgula</Button>
                </div>

                {ipReputation && (
                    <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
                        <div className="grid grid-cols-4 gap-4">
                            <div>
                                <p className="text-sm text-gray-400">Risk Score</p>
                                <p className={`text-2xl font-bold ${ipReputation.risk_score > 70 ? 'text-red-400' : ipReputation.risk_score > 40 ? 'text-yellow-400' : 'text-green-400'}`}>
                                    {ipReputation.risk_score}/100
                                </p>
                            </div>
                            <div>
                                <p className="text-sm text-gray-400">Ülke</p>
                                <p className="text-xl font-bold text-white">{ipReputation.country}</p>
                            </div>
                            <div>
                                <p className="text-sm text-gray-400">ASN</p>
                                <p className="text-lg text-white">{ipReputation.asn}</p>
                            </div>
                            <div>
                                <p className="text-sm text-gray-400">Raporlar</p>
                                <p className="text-xl font-bold text-orange-400">{ipReputation.reports}</p>
                            </div>
                        </div>
                        {ipReputation.categories.length > 0 && (
                            <div className="mt-3 flex gap-2">
                                {ipReputation.categories.map((cat, i) => (
                                    <Badge key={i} variant="danger" size="sm">{cat}</Badge>
                                ))}
                            </div>
                        )}
                    </div>
                )}
            </Card>
        </div>
    );

    const RealTimeTab = () => (
        <div className="space-y-6">
            {/* IDS Status */}
            <Card className="bg-gradient-to-r from-green-900/30 to-blue-900/30 border-green-500/30">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <Radio className={`w-10 h-10 ${idsStatus?.status === 'running' ? 'text-green-400 animate-pulse' : 'text-gray-400'}`} />
                        <div>
                            <h3 className="text-lg font-semibold text-white">Real-time IDS</h3>
                            <p className="text-sm text-gray-400">
                                Durum: {idsStatus?.status === 'running' ? '🟢 Çalışıyor' : idsStatus?.status === 'stopped' ? '🔴 Durdu' : '⚪ Başlatılmadı'}
                            </p>
                        </div>
                    </div>
                    <Button onClick={loadIdsStatus} variant="outline" size="sm">
                        <RefreshCw className="w-4 h-4" />
                    </Button>
                </div>

                {idsStatus?.metrics && (
                    <div className="grid grid-cols-4 gap-4 mt-4 pt-4 border-t border-gray-700">
                        <div className="text-center">
                            <p className="text-2xl font-bold text-white">{idsStatus.metrics.total_packets?.toLocaleString()}</p>
                            <p className="text-xs text-gray-400">Toplam Paket</p>
                        </div>
                        <div className="text-center">
                            <p className="text-2xl font-bold text-red-400">{idsStatus.metrics.attack_packets}</p>
                            <p className="text-xs text-gray-400">Saldırı Tespit</p>
                        </div>
                        <div className="text-center">
                            <p className="text-2xl font-bold text-blue-400">{idsStatus.metrics.packets_per_second?.toFixed(1)}</p>
                            <p className="text-xs text-gray-400">Paket/Saniye</p>
                        </div>
                        <div className="text-center">
                            <p className="text-2xl font-bold text-green-400">{idsStatus.metrics.processing_time_avg_ms?.toFixed(2)} ms</p>
                            <p className="text-xs text-gray-400">İşlem Süresi</p>
                        </div>
                    </div>
                )}
            </Card>

            {/* Recent Alerts */}
            <Card>
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    Son Alert'ler
                </h3>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                    {idsAlerts.length > 0 ? idsAlerts.map((alert, idx) => (
                        <div key={idx} className={`p-3 rounded-lg border ${SEVERITY_COLORS[alert.severity] || SEVERITY_COLORS.medium}`}>
                            <div className="flex justify-between">
                                <span className="font-medium text-white">{alert.attack_type}</span>
                                <Badge variant={alert.severity === 'critical' ? 'danger' : 'warning'} size="sm">
                                    {alert.severity}
                                </Badge>
                            </div>
                            <p className="text-sm text-gray-400 mt-1">{alert.description}</p>
                            <p className="text-xs text-gray-500 mt-1">{alert.timestamp}</p>
                        </div>
                    )) : (
                        <p className="text-center text-gray-500 py-4">Henüz alert yok veya IDS çalışmıyor</p>
                    )}
                </div>
            </Card>
        </div>
    );

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <Radar className="w-7 h-7 text-blue-400" />
                        Tehdit İstihbaratı
                    </h1>
                    <p className="text-slate-400 mt-1">MITRE ATT&CK, IOC, Real-time IDS</p>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={hours === null ? 'all' : hours}
                        onChange={(e) => setHours(e.target.value === 'all' ? null : Number(e.target.value))}
                        className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm"
                    >
                        <option value={6}>Son 6 saat</option>
                        <option value={24}>Son 24 saat</option>
                        <option value={72}>Son 3 gün</option>
                        <option value={168}>Son 1 hafta</option>
                        <option value="all">Tüm Zamanlar</option>
                    </select>
                    <Button onClick={loadData} variant="outline" size="sm">
                        <RefreshCw className="w-4 h-4 mr-2" />
                        Yenile
                    </Button>
                </div>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 border-b border-gray-700 pb-2">
                {[
                    { id: 'overview', label: '📊 Genel Bakış', icon: BarChart3 },
                    { id: 'mitre', label: '🎯 MITRE ATT&CK', icon: Target },
                    { id: 'ioc', label: '🔍 IOC Yönetimi', icon: Hash },
                    { id: 'realtime', label: '📡 Real-time IDS', icon: Radio },
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-t-lg transition-colors ${activeTab === tab.id
                            ? 'bg-blue-600 text-white'
                            : 'text-gray-400 hover:bg-gray-800'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {activeTab === 'overview' && <OverviewTab />}
            {activeTab === 'mitre' && <MitreTab />}
            {activeTab === 'ioc' && <IocTab />}
            {activeTab === 'realtime' && <RealTimeTab />}
        </div>
    );
}
