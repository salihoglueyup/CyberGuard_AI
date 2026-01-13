import React, { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
    Controls,
    Background,
    useNodesState,
    useEdgesState,
    MarkerType
} from 'reactflow';
import 'reactflow/dist/style.css';
import api from '../services/api';

// Gauge Component - Yuvarlak Skor Göstergesi
const GaugeChart = ({ value, grade, size = 200 }) => {
    const radius = (size - 20) / 2;
    const circumference = radius * 2 * Math.PI;
    const offset = circumference - (value / 100) * circumference;

    const getGradeColor = (g) => {
        const colors = { A: '#22c55e', B: '#3b82f6', C: '#eab308', D: '#f97316', F: '#ef4444' };
        return colors[g] || '#6b7280';
    };

    return (
        <div className="relative" style={{ width: size, height: size }}>
            <svg className="transform -rotate-90" width={size} height={size}>
                {/* Arka plan çemberi */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke="#374151"
                    strokeWidth="12"
                    fill="none"
                />
                {/* Değer çemberi */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    stroke={getGradeColor(grade)}
                    strokeWidth="12"
                    fill="none"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                    className="transition-all duration-1000 ease-out"
                />
            </svg>
            <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className={`text-5xl font-bold`} style={{ color: getGradeColor(grade) }}>
                    {grade}
                </span>
                <span className="text-2xl font-semibold text-white">{value}%</span>
            </div>
        </div>
    );
};

// Custom Node for React Flow
const DeviceNode = ({ data }) => {
    const icons = {
        router: '🌐',
        firewall: '🔥',
        switch: '🔀',
        server: '🖥️',
        workstation: '💻',
        default: '📱'
    };

    const statusColors = {
        online: 'border-green-500 shadow-green-500/30',
        warning: 'border-yellow-500 shadow-yellow-500/30',
        offline: 'border-red-500 shadow-red-500/30'
    };

    return (
        <div className={`px-4 py-3 bg-gray-800 rounded-xl border-2 shadow-lg ${statusColors[data.status] || statusColors.online}`}>
            <div className="text-center">
                <div className="text-3xl mb-1">{icons[data.type] || icons.default}</div>
                <div className="text-sm font-medium text-white">{data.label}</div>
                <div className="text-xs text-gray-400">{data.ip}</div>
            </div>
        </div>
    );
};

const nodeTypes = { device: DeviceNode };

const SecurityHub = () => {
    const [securityScore, setSecurityScore] = useState(null);
    const [honeypot, setHoneypot] = useState(null);
    const [compliance, setCompliance] = useState(null);
    const [topology, setTopology] = useState(null);
    const [heatmap, setHeatmap] = useState(null);
    const [activeTab, setActiveTab] = useState('score');
    const [loading, setLoading] = useState(false);
    const [lastUpdate, setLastUpdate] = useState(null);
    const [autoRefresh, setAutoRefresh] = useState(false);

    // React Flow states
    const [nodes, setNodes, onNodesChange] = useNodesState([]);
    const [edges, setEdges, onEdgesChange] = useEdgesState([]);

    useEffect(() => {
        loadSecurityScore();
    }, []);

    // Auto refresh
    useEffect(() => {
        if (autoRefresh) {
            const interval = setInterval(() => {
                handleTabChange(activeTab);
            }, 30000);
            return () => clearInterval(interval);
        }
    }, [autoRefresh, activeTab]);

    const loadSecurityScore = async () => {
        setLoading(true);
        try {
            const response = await api.get('/security/score');
            if (response.data.success) {
                setSecurityScore(response.data.data);
                setLastUpdate(new Date());
            }
        } catch (error) {
            console.error('Güvenlik skoru yüklenirken hata:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadHoneypot = async () => {
        setLoading(true);
        try {
            const response = await api.get('/security/honeypot');
            if (response.data.success) {
                setHoneypot(response.data.data);
                setLastUpdate(new Date());
            }
        } catch (error) {
            console.error('Bal küpü yüklenirken hata:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadCompliance = async () => {
        setLoading(true);
        try {
            const response = await api.get('/security/compliance');
            if (response.data.success) {
                setCompliance(response.data.data);
                setLastUpdate(new Date());
            }
        } catch (error) {
            console.error('Uyumluluk yüklenirken hata:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadTopology = async () => {
        setLoading(true);
        try {
            const response = await api.get('/security/topology');
            if (response.data.success) {
                const data = response.data.data;
                setTopology(data);

                // Convert to React Flow format
                const flowNodes = data.nodes?.map((node, index) => ({
                    id: node.id,
                    type: 'device',
                    position: {
                        x: 150 + (index % 4) * 200,
                        y: 100 + Math.floor(index / 4) * 150
                    },
                    data: {
                        label: node.label,
                        ip: node.ip,
                        type: node.type,
                        status: node.status || 'online'
                    }
                })) || [];

                const flowEdges = data.edges?.map((edge, index) => ({
                    id: `e${index}`,
                    source: edge.source,
                    target: edge.target,
                    animated: edge.traffic === 'high',
                    style: { stroke: edge.traffic === 'high' ? '#22c55e' : '#6b7280' },
                    markerEnd: { type: MarkerType.ArrowClosed }
                })) || [];

                setNodes(flowNodes);
                setEdges(flowEdges);
                setLastUpdate(new Date());
            }
        } catch (error) {
            console.error('Topoloji yüklenirken hata:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadHeatmap = async () => {
        setLoading(true);
        try {
            const response = await api.get('/security/heatmap');
            if (response.data.success) {
                setHeatmap(response.data.data);
                setLastUpdate(new Date());
            }
        } catch (error) {
            console.error('Tehdit haritası yüklenirken hata:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleTabChange = (tab) => {
        setActiveTab(tab);
        switch (tab) {
            case 'score': loadSecurityScore(); break;
            case 'honeypot': loadHoneypot(); break;
            case 'compliance': loadCompliance(); break;
            case 'topology': loadTopology(); break;
            case 'heatmap': loadHeatmap(); break;
        }
    };

    const getGradeColor = (grade) => {
        const colors = { A: 'text-green-400', B: 'text-blue-400', C: 'text-yellow-400', D: 'text-orange-400', F: 'text-red-400' };
        return colors[grade] || 'text-gray-400';
    };

    const getStatusColor = (status) => {
        const colors = {
            compliant: 'bg-green-500',
            partial: 'bg-yellow-500',
            'non-compliant': 'bg-red-500',
            active: 'bg-green-500',
            inactive: 'bg-gray-500'
        };
        return colors[status] || 'bg-gray-500';
    };

    const translatePriority = (priority) => {
        const map = { high: 'YÜKSEK', medium: 'ORTA', low: 'DÜŞÜK' };
        return map[priority] || priority?.toUpperCase();
    };

    const translateStatus = (status) => {
        const map = {
            compliant: 'Uyumlu',
            partial: 'Kısmi',
            'non-compliant': 'Uyumsuz',
            active: 'Aktif',
            inactive: 'Pasif',
            improving: 'İyileşiyor',
            declining: 'Düşüyor',
            stable: 'Stabil'
        };
        return map[status] || status;
    };

    const translateComponent = (key) => {
        const map = {
            firewall: 'Güvenlik Duvarı',
            antivirus: 'Antivirüs',
            updates: 'Güncellemeler',
            encryption: 'Şifreleme',
            network: 'Ağ Güvenliği',
            access_control: 'Erişim Kontrolü',
            backup: 'Yedekleme',
            monitoring: 'İzleme'
        };
        return map[key] || key.replace(/_/g, ' ');
    };

    const tabs = [
        { id: 'score', label: '🛡️ Güvenlik Skoru' },
        { id: 'honeypot', label: '🍯 Bal Küpü' },
        { id: 'compliance', label: '✅ Uyumluluk' },
        { id: 'topology', label: '🌐 Ağ Topolojisi' },
        { id: 'heatmap', label: '🗺️ Tehdit Haritası' }
    ];

    const countryFlags = {
        RU: '🇷🇺', CN: '🇨🇳', US: '🇺🇸', BR: '🇧🇷', IN: '🇮🇳',
        DE: '🇩🇪', NL: '🇳🇱', FR: '🇫🇷', KR: '🇰🇷', GB: '🇬🇧',
        JP: '🇯🇵', TR: '🇹🇷', UA: '🇺🇦', IR: '🇮🇷'
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8 flex justify-between items-start">
                    <div>
                        <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                            🛡️ Güvenlik Merkezi
                        </h1>
                        <p className="text-gray-400 mt-2">
                            Kapsamlı güvenlik izleme ve analiz merkezi
                        </p>
                    </div>
                    <div className="flex items-center gap-4">
                        {lastUpdate && (
                            <span className="text-xs text-gray-500">
                                Son güncelleme: {lastUpdate.toLocaleTimeString('tr-TR')}
                            </span>
                        )}
                        <button
                            onClick={() => setAutoRefresh(!autoRefresh)}
                            className={`px-3 py-1.5 rounded-lg text-sm transition ${autoRefresh
                                    ? 'bg-green-600 text-white'
                                    : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                                }`}
                        >
                            {autoRefresh ? '🔄 Otomatik' : '⏸️ Manuel'}
                        </button>
                    </div>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => handleTabChange(tab.id)}
                            className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${activeTab === tab.id
                                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-500/20'
                                    : 'bg-gray-800/80 text-gray-400 hover:bg-gray-700 hover:text-white'
                                }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {loading && (
                    <div className="flex justify-center py-12">
                        <div className="text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500 mx-auto"></div>
                            <p className="text-gray-400 mt-4">Yükleniyor...</p>
                        </div>
                    </div>
                )}

                {/* ==================== GÜVENLIK SKORU ==================== */}
                {activeTab === 'score' && securityScore && !loading && (
                    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                        {/* Ana Skor - Gauge */}
                        <div className="lg:col-span-1 bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 text-center border border-gray-700">
                            <h2 className="text-lg font-medium text-gray-400 mb-6">Genel Güvenlik Skoru</h2>
                            <div className="flex justify-center mb-4">
                                <GaugeChart
                                    value={securityScore.overall_score}
                                    grade={securityScore.grade}
                                    size={180}
                                />
                            </div>
                            <div className="text-gray-400 mt-2">{translateStatus(securityScore.status)}</div>
                            <div className={`mt-4 inline-block px-4 py-2 rounded-full text-sm font-medium ${securityScore.trend === 'improving' ? 'bg-green-900/50 text-green-400 border border-green-500/30' :
                                    securityScore.trend === 'declining' ? 'bg-red-900/50 text-red-400 border border-red-500/30' :
                                        'bg-gray-700 text-gray-300 border border-gray-600'
                                }`}>
                                {securityScore.trend === 'improving' ? '📈' : securityScore.trend === 'declining' ? '📉' : '➡️'}
                                {' '}{securityScore.change_from_last_week > 0 ? '+' : ''}{securityScore.change_from_last_week}% bu hafta
                            </div>
                        </div>

                        {/* Bileşen Skorları */}
                        <div className="lg:col-span-2 bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                            <h2 className="text-lg font-medium text-gray-400 mb-4">📊 Bileşen Skorları</h2>
                            <div className="space-y-4">
                                {Object.entries(securityScore.components || {}).map(([key, value]) => (
                                    <div key={key} className="group">
                                        <div className="flex justify-between text-sm mb-1">
                                            <span className="capitalize font-medium">{translateComponent(key)}</span>
                                            <span className={`font-bold ${value >= 80 ? 'text-green-400' : value >= 60 ? 'text-yellow-400' : 'text-red-400'}`}>
                                                {value}%
                                            </span>
                                        </div>
                                        <div className="bg-gray-700 rounded-full h-3 overflow-hidden">
                                            <div
                                                className={`h-3 rounded-full transition-all duration-1000 ease-out ${value >= 80 ? 'bg-gradient-to-r from-green-500 to-emerald-400' :
                                                        value >= 60 ? 'bg-gradient-to-r from-yellow-500 to-amber-400' :
                                                            'bg-gradient-to-r from-red-500 to-rose-400'
                                                    }`}
                                                style={{ width: `${value}%` }}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Öneriler */}
                        <div className="lg:col-span-3 bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                            <h2 className="text-lg font-medium text-gray-400 mb-4">🎯 Öneriler</h2>
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                {securityScore.recommendations?.map((rec, i) => (
                                    <div key={i} className={`p-4 rounded-xl border transition-all hover:scale-[1.02] ${rec.priority === 'high' ? 'border-red-500/50 bg-red-900/20 hover:border-red-500' :
                                            rec.priority === 'medium' ? 'border-yellow-500/50 bg-yellow-900/20 hover:border-yellow-500' :
                                                'border-gray-600 bg-gray-700/30 hover:border-gray-500'
                                        }`}>
                                        <div className={`text-xs font-bold mb-2 ${rec.priority === 'high' ? 'text-red-400' :
                                                rec.priority === 'medium' ? 'text-yellow-400' : 'text-gray-400'
                                            }`}>
                                            {translatePriority(rec.priority)}
                                        </div>
                                        <div className="font-medium text-white">{rec.action}</div>
                                        <div className="text-green-400 text-sm mt-2">✨ {rec.impact}</div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}

                {/* ==================== BAL KÜPÜ ==================== */}
                {activeTab === 'honeypot' && honeypot && !loading && (
                    <div className="space-y-6">
                        {/* İstatistikler */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-gradient-to-br from-yellow-900/30 to-gray-900 rounded-xl p-5 text-center border border-yellow-500/30">
                                <div className="text-3xl font-bold text-yellow-400">{honeypot.honeypots?.length || 0}</div>
                                <div className="text-gray-400 text-sm">Aktif Bal Küpü</div>
                            </div>
                            <div className="bg-gradient-to-br from-red-900/30 to-gray-900 rounded-xl p-5 text-center border border-red-500/30">
                                <div className="text-3xl font-bold text-red-400">{honeypot.total_attacks_today || 0}</div>
                                <div className="text-gray-400 text-sm">Bugünkü Saldırı</div>
                            </div>
                            <div className="bg-gradient-to-br from-orange-900/30 to-gray-900 rounded-xl p-5 text-center border border-orange-500/30">
                                <div className="text-3xl font-bold text-orange-400">{honeypot.unique_attackers || 0}</div>
                                <div className="text-gray-400 text-sm">Benzersiz Saldırgan</div>
                            </div>
                            <div className="bg-gradient-to-br from-green-900/30 to-gray-900 rounded-xl p-5 text-center border border-green-500/30">
                                <div className="text-3xl font-bold text-green-400">{translateStatus(honeypot.status)}</div>
                                <div className="text-gray-400 text-sm">Durum</div>
                            </div>
                        </div>

                        {/* Bal Küpü Listesi */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                            <h2 className="text-lg font-medium mb-4">🍯 Aktif Bal Küpleri</h2>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                                {honeypot.honeypots?.map(hp => (
                                    <div key={hp.id} className="bg-gray-700/50 rounded-xl p-4 hover:bg-gray-700/70 transition border border-gray-600">
                                        <div className="flex items-center gap-2 mb-2">
                                            <div className={`w-3 h-3 rounded-full ${getStatusColor(hp.status)} animate-pulse`}></div>
                                            <span className="font-medium text-white">{hp.type}</span>
                                        </div>
                                        <div className="text-sm text-gray-400">Port: <span className="text-cyan-400">{hp.port}</span></div>
                                        <div className="text-sm text-gray-400">IP: <span className="text-cyan-400">{hp.ip}</span></div>
                                        <div className="text-lg font-bold text-yellow-400 mt-2">{hp.attacks_captured} yakalama</div>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* Son Yakalamalar */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                            <h2 className="text-lg font-medium mb-4">📥 Son Yakalamalar</h2>
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                                            <th className="pb-3">Saldırgan IP</th>
                                            <th className="pb-3">Bal Küpü</th>
                                            <th className="pb-3">Saldırı Tipi</th>
                                            <th className="pb-3">Yakalanan</th>
                                            <th className="pb-3">Zaman</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {honeypot.recent_captures?.map((capture, i) => (
                                            <tr key={i} className="border-t border-gray-700/50 hover:bg-gray-700/30">
                                                <td className="py-3 text-red-400 font-mono">{capture.attacker_ip}</td>
                                                <td className="py-3">{capture.honeypot}</td>
                                                <td className="py-3">
                                                    <span className="px-2 py-1 bg-orange-900/30 text-orange-400 rounded text-xs">
                                                        {capture.attack_type}
                                                    </span>
                                                </td>
                                                <td className="py-3 text-green-400">{capture.captured_data}</td>
                                                <td className="py-3 text-gray-400 text-sm">{new Date(capture.timestamp).toLocaleString('tr-TR')}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {/* ==================== UYUMLULUK ==================== */}
                {activeTab === 'compliance' && compliance && !loading && (
                    <div className="space-y-6">
                        {/* Genel Uyumluluk */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 text-center border border-gray-700">
                            <h2 className="text-lg text-gray-400 mb-4">Genel Uyumluluk</h2>
                            <div className="flex justify-center mb-4">
                                <GaugeChart value={compliance.overall_compliance} grade={compliance.overall_compliance >= 80 ? 'A' : compliance.overall_compliance >= 60 ? 'B' : 'C'} size={150} />
                            </div>
                            <div className={`mt-2 inline-block px-4 py-2 rounded-full text-sm font-medium ${compliance.status === 'compliant' ? 'bg-green-900/50 text-green-400 border border-green-500/30' :
                                    'bg-yellow-900/50 text-yellow-400 border border-yellow-500/30'
                                }`}>
                                {translateStatus(compliance.status)}
                            </div>
                        </div>

                        {/* Standartlar */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {compliance.standards?.map(std => (
                                <div key={std.standard} className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-5 border border-gray-700 hover:border-gray-600 transition">
                                    <div className="flex justify-between items-center mb-3">
                                        <span className="font-bold text-lg text-white">{std.standard}</span>
                                        <span className={`px-2 py-1 rounded text-xs font-medium ${std.status === 'compliant' ? 'bg-green-900/50 text-green-400' :
                                                std.status === 'partial' ? 'bg-yellow-900/50 text-yellow-400' :
                                                    'bg-red-900/50 text-red-400'
                                            }`}>
                                            {translateStatus(std.status)}
                                        </span>
                                    </div>
                                    <div className="text-4xl font-bold mb-2 text-white">{std.score}%</div>
                                    <div className="text-sm text-gray-400 mb-3">
                                        {std.controls_passed}/{std.controls_total} kontrol geçti
                                    </div>
                                    <div className="bg-gray-700 rounded-full h-2 overflow-hidden">
                                        <div
                                            className={`h-2 rounded-full transition-all duration-1000 ${std.score >= 80 ? 'bg-gradient-to-r from-green-500 to-emerald-400' :
                                                    std.score >= 60 ? 'bg-gradient-to-r from-yellow-500 to-amber-400' :
                                                        'bg-gradient-to-r from-red-500 to-rose-400'
                                                }`}
                                            style={{ width: `${std.score}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* ==================== AĞ TOPOLOJİSİ (React Flow) ==================== */}
                {activeTab === 'topology' && topology && !loading && (
                    <div className="space-y-6">
                        {/* İstatistikler */}
                        <div className="grid grid-cols-3 gap-4">
                            <div className="bg-gradient-to-br from-cyan-900/30 to-gray-900 rounded-xl p-4 text-center border border-cyan-500/30">
                                <div className="text-2xl font-bold text-cyan-400">{topology.stats?.total_devices}</div>
                                <div className="text-gray-400 text-sm">Toplam Cihaz</div>
                            </div>
                            <div className="bg-gradient-to-br from-green-900/30 to-gray-900 rounded-xl p-4 text-center border border-green-500/30">
                                <div className="text-2xl font-bold text-green-400">{topology.stats?.active_connections}</div>
                                <div className="text-gray-400 text-sm">Aktif Bağlantı</div>
                            </div>
                            <div className="bg-gradient-to-br from-yellow-900/30 to-gray-900 rounded-xl p-4 text-center border border-yellow-500/30">
                                <div className="text-2xl font-bold text-yellow-400">{topology.stats?.warnings}</div>
                                <div className="text-gray-400 text-sm">Uyarı</div>
                            </div>
                        </div>

                        {/* Interactive Network Graph */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl border border-gray-700 overflow-hidden" style={{ height: 500 }}>
                            <ReactFlow
                                nodes={nodes}
                                edges={edges}
                                onNodesChange={onNodesChange}
                                onEdgesChange={onEdgesChange}
                                nodeTypes={nodeTypes}
                                fitView
                                attributionPosition="bottom-left"
                            >
                                <Controls className="!bg-gray-800 !border-gray-700" />
                                <Background color="#374151" gap={20} />
                            </ReactFlow>
                        </div>

                        <div className="text-center text-sm text-gray-500">
                            💡 İpucu: Cihazları sürükleyerek konumlandırabilirsiniz
                        </div>
                    </div>
                )}

                {/* ==================== TEHDİT HARİTASI ==================== */}
                {activeTab === 'heatmap' && heatmap && !loading && (
                    <div className="space-y-6">
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                            <h2 className="text-lg font-medium mb-4">🗺️ Ülkelere Göre Tehdit Haritası</h2>
                            <div className="text-center mb-6">
                                <span className="text-4xl font-bold text-red-400">{heatmap.total_attacks?.toLocaleString()}</span>
                                <span className="text-gray-400 ml-2">Toplam Saldırı ({heatmap.period})</span>
                            </div>
                            <div className="space-y-3">
                                {heatmap.heatmap?.map((country, index) => (
                                    <div key={country.code} className="flex items-center gap-4 p-2 rounded-lg hover:bg-gray-700/30 transition">
                                        <span className="w-6 text-center font-bold text-gray-500">{index + 1}</span>
                                        <span className="w-10 text-2xl">{countryFlags[country.code] || '🌍'}</span>
                                        <span className="w-36 font-medium">{country.name}</span>
                                        <div className="flex-1 bg-gray-700 rounded-full h-4 overflow-hidden">
                                            <div
                                                className="h-4 rounded-full transition-all duration-1000"
                                                style={{
                                                    width: `${country.intensity * 100}%`,
                                                    background: `linear-gradient(to right, #f59e0b, #ef4444)`
                                                }}
                                            />
                                        </div>
                                        <span className="w-28 text-right font-mono text-red-400">{country.attacks.toLocaleString()}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default SecurityHub;
