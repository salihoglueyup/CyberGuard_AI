import { useState, useEffect, useCallback } from 'react';
import {
    BarChart3, TrendingUp, Users, Clock, Activity, Cpu,
    Server, HardDrive, Wifi, Shield, AlertTriangle, Brain, RefreshCw, Zap, Flame
} from 'lucide-react';
import {
    AreaChart, Area, BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
    XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, RadialBarChart, RadialBar
} from 'recharts';
import { Card, Badge, Button } from '../components/ui';
import { ProgressBar } from '../components/ui/Progress';
import { attacksApi, modelsApi } from '../services/api';
import { useWebSocket } from '../hooks/useWebSocket';

const API_BASE = 'http://localhost:8000/api';

// Premium renk paleti
const COLORS = {
    primary: '#3b82f6',
    secondary: '#8b5cf6',
    success: '#22c55e',
    warning: '#f59e0b',
    danger: '#ef4444',
    info: '#06b6d4',
    pink: '#ec4899',
    gradient: ['#667eea', '#764ba2']
};

const PIE_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899'];

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="bg-slate-900/95 backdrop-blur-xl border border-slate-700/50 rounded-xl p-4 shadow-2xl">
                <p className="text-slate-400 text-sm mb-2">{label}</p>
                {payload.map((entry, index) => (
                    <div key={index} className="flex items-center gap-2">
                        <div
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: entry.color }}
                        />
                        <span className="text-white font-medium">{entry.name}:</span>
                        <span className="text-slate-300">{entry.value}</span>
                    </div>
                ))}
            </div>
        );
    }
    return null;
};

// Radial Progress Component
const RadialProgress = ({ value, color, label, icon: Icon }) => {
    const data = [{ value, fill: color }];

    return (
        <div className="relative flex flex-col items-center">
            <div className="relative w-24 h-24">
                <ResponsiveContainer width="100%" height="100%">
                    <RadialBarChart
                        cx="50%"
                        cy="50%"
                        innerRadius="70%"
                        outerRadius="100%"
                        data={data}
                        startAngle={90}
                        endAngle={-270}
                    >
                        <RadialBar
                            background={{ fill: '#1e293b' }}
                            dataKey="value"
                            cornerRadius={10}
                        />
                    </RadialBarChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex items-center justify-center">
                    {Icon && <Icon className="w-6 h-6" style={{ color }} />}
                </div>
            </div>
            <div className="mt-2 text-center">
                <p className="text-2xl font-bold text-white">{Math.round(value)}%</p>
                <p className="text-xs text-slate-400">{label}</p>
            </div>
        </div>
    );
};

export default function Analytics() {
    const { isConnected, systemStats, analytics, threats, requestAnalytics } = useWebSocket();

    const [initialStats, setInitialStats] = useState(null);
    const [hourlyData, setHourlyData] = useState([]);
    const [typeData, setTypeData] = useState([]);
    const [severityData, setSeverityData] = useState([]);
    const [modelStats, setModelStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [hours, setHours] = useState(24);

    useEffect(() => {
        if (analytics) {
            if (analytics.by_type) {
                const total = Object.values(analytics.by_type).reduce((a, b) => a + b, 0) || 1;
                setTypeData(Object.entries(analytics.by_type).map(([name, value]) => ({
                    name,
                    value: Math.round((value / total) * 100),
                    count: value
                })));
            }
            if (analytics.by_severity) {
                setSeverityData(Object.entries(analytics.by_severity).map(([name, value]) => ({
                    name,
                    value
                })));
            }
        }
    }, [analytics]);

    useEffect(() => {
        if (threats.length > 0) {
            const hourlyMap = {};
            threats.slice(0, 50).forEach(t => {
                const hour = new Date(t.timestamp).getHours() + ':00';
                if (!hourlyMap[hour]) hourlyMap[hour] = { threats: 0, blocked: 0 };
                hourlyMap[hour].threats++;
                if (t.blocked) hourlyMap[hour].blocked++;
            });

            const data = Object.entries(hourlyMap).map(([hour, data]) => ({
                hour,
                threats: data.threats,
                blocked: data.blocked
            }));

            if (data.length > 0) {
                setHourlyData(prev => [...data, ...prev].slice(0, 24));
            }
        }
    }, [threats]);

    const loadInitialData = useCallback(async () => {
        setLoading(true);
        try {
            const [statsRes, hourlyRes, typeRes, sevRes, modelsRes] = await Promise.all([
                attacksApi.getStats(hours),
                attacksApi.getTimeline(hours),
                attacksApi.getByType(hours),
                attacksApi.getBySeverity(hours),
                modelsApi.getStats()
            ]);

            if (statsRes.data.success) setInitialStats(statsRes.data.data);

            if (hourlyRes.data.success) {
                const timeline = hourlyRes.data.data || [];
                setHourlyData(timeline.map(item => ({
                    hour: item.hour || item.time,
                    threats: item.count || item.attacks || 0,
                    blocked: Math.floor((item.count || 0) * 0.85)
                })));
            }

            if (typeRes.data.success) {
                const types = typeRes.data.data || {};
                const total = Object.values(types).reduce((a, b) => a + b, 0) || 1;
                setTypeData(Object.entries(types).map(([name, value]) => ({
                    name,
                    value: Math.round((value / total) * 100),
                    count: value
                })));
            }

            if (sevRes.data.success) {
                setSeverityData(Object.entries(sevRes.data.data || {}).map(([name, value]) => ({
                    name,
                    value
                })));
            }

            if (modelsRes.data.success) setModelStats(modelsRes.data.data);

        } catch (error) {
            console.error('Analytics load error:', error);
        } finally {
            setLoading(false);
        }
    }, [hours]);

    useEffect(() => {
        loadInitialData();
    }, [loadInitialData]);

    const stats = analytics?.stats || initialStats || {};
    const totalThreats = stats.total_attacks || stats.total || 0;
    const blockedCount = stats.blocked || Math.floor(totalThreats * 0.85);
    const blockRate = stats.block_rate || (totalThreats > 0 ? ((blockedCount / totalThreats) * 100).toFixed(1) : 0);
    const activeModels = modelStats?.total_models || 0;

    const cpu = systemStats?.cpu || 0;
    const memory = systemStats?.memory || 0;
    const disk = systemStats?.disk || 0;

    if (loading && !initialStats) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="relative">
                    <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-500/20 border-t-blue-500"></div>
                    <div className="absolute inset-0 flex items-center justify-center">
                        <BarChart3 className="w-6 h-6 text-blue-400 animate-pulse" />
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl">
                            <BarChart3 className="w-6 h-6 text-white" />
                        </div>
                        Analitik Dashboard
                    </h1>
                    <p className="text-slate-400 mt-1">
                        {isConnected ? '🔴 WebSocket ile gerçek zamanlı veri akışı' : 'Performans metrikleri'}
                    </p>
                </div>
                <div className="flex items-center gap-3">
                    <select
                        value={hours}
                        onChange={(e) => setHours(Number(e.target.value))}
                        className="bg-slate-800/50 backdrop-blur border border-slate-700/50 rounded-xl px-4 py-2 text-white text-sm focus:border-blue-500 transition-colors"
                    >
                        <option value={6}>Son 6 saat</option>
                        <option value={24}>Son 24 saat</option>
                        <option value={72}>Son 3 gün</option>
                        <option value={168}>Son 1 hafta</option>
                    </select>
                    <Button variant="ghost" size="sm" icon={RefreshCw} onClick={() => { loadInitialData(); requestAnalytics(); }}>
                        Yenile
                    </Button>
                    <div className={`flex items-center gap-2 px-3 py-2 rounded-xl ${isConnected ? 'bg-green-500/20 border border-green-500/30' : 'bg-red-500/20 border border-red-500/30'}`}>
                        <span className={`relative flex h-2 w-2`}>
                            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></span>
                            <span className={`relative inline-flex rounded-full h-2 w-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
                        </span>
                        <span className={`text-sm font-medium ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
                            {isConnected ? 'Canlı' : 'Offline'}
                        </span>
                    </div>
                </div>
            </div>

            {/* Summary Stats - Premium Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Total Threats */}
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-red-500 to-orange-500 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">Toplam Saldırı</p>
                                <p className="text-4xl font-bold bg-gradient-to-r from-red-400 to-orange-400 bg-clip-text text-transparent mt-1">
                                    {totalThreats.toLocaleString()}
                                </p>
                                <p className="text-xs text-slate-500 mt-2 flex items-center gap-1">
                                    <Flame className="w-3 h-3 text-orange-400" />
                                    Son {hours} saat
                                </p>
                            </div>
                            <div className="p-3 bg-gradient-to-br from-red-500/20 to-orange-500/20 rounded-xl">
                                <AlertTriangle className="w-6 h-6 text-red-400" />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Block Rate */}
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-green-500 to-emerald-500 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">Engelleme Oranı</p>
                                <p className="text-4xl font-bold bg-gradient-to-r from-green-400 to-emerald-400 bg-clip-text text-transparent mt-1">
                                    {blockRate}%
                                </p>
                                <p className="text-xs text-slate-500 mt-2">
                                    ✅ {blockedCount.toLocaleString()} engellendi
                                </p>
                            </div>
                            <div className="p-3 bg-gradient-to-br from-green-500/20 to-emerald-500/20 rounded-xl">
                                <Shield className="w-6 h-6 text-green-400" />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Active Models */}
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">Aktif Model</p>
                                <p className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent mt-1">
                                    {activeModels}
                                </p>
                                <p className="text-xs text-slate-500 mt-2">
                                    🧠 {modelStats?.deployed_models || 0} deploy edildi
                                </p>
                            </div>
                            <div className="p-3 bg-gradient-to-br from-purple-500/20 to-pink-500/20 rounded-xl">
                                <Brain className="w-6 h-6 text-purple-400" />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* CPU Usage */}
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-2xl blur opacity-30 group-hover:opacity-50 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">CPU Kullanımı</p>
                                <p className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent mt-1">
                                    {Math.round(cpu)}%
                                </p>
                                <p className="text-xs text-slate-500 mt-2 flex items-center gap-1">
                                    {isConnected && <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>}
                                    {isConnected ? 'Gerçek zamanlı' : '-'}
                                </p>
                            </div>
                            <div className="p-3 bg-gradient-to-br from-blue-500/20 to-cyan-500/20 rounded-xl">
                                <Cpu className="w-6 h-6 text-blue-400" />
                            </div>
                        </div>
                    </Card>
                </div>
            </div>

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Hourly Trend - Premium */}
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500/50 to-purple-500/50 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-gradient-to-br from-blue-500 to-purple-500 rounded-lg">
                                    <TrendingUp className="w-4 h-4 text-white" />
                                </div>
                                <h3 className="text-lg font-semibold text-white">Saatlik Trend</h3>
                            </div>
                            {isConnected && (
                                <div className="flex items-center gap-2 px-2 py-1 bg-green-500/20 rounded-lg">
                                    <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                                    <span className="text-xs text-green-400">Canlı</span>
                                </div>
                            )}
                        </div>
                        <div className="h-72">
                            {hourlyData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={hourlyData}>
                                        <defs>
                                            <linearGradient id="threatGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#ef4444" stopOpacity={0.4} />
                                                <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                                            </linearGradient>
                                            <linearGradient id="blockedGradient" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#22c55e" stopOpacity={0.4} />
                                                <stop offset="100%" stopColor="#22c55e" stopOpacity={0} />
                                            </linearGradient>
                                            <filter id="glow">
                                                <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                                                <feMerge>
                                                    <feMergeNode in="coloredBlur" />
                                                    <feMergeNode in="SourceGraphic" />
                                                </feMerge>
                                            </filter>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                        <XAxis dataKey="hour" stroke="#64748b" fontSize={10} />
                                        <YAxis stroke="#64748b" fontSize={10} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Legend
                                            wrapperStyle={{ paddingTop: '20px' }}
                                            formatter={(value) => <span className="text-slate-300">{value}</span>}
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="threats"
                                            name="Tehdit"
                                            stroke="#ef4444"
                                            strokeWidth={2}
                                            fill="url(#threatGradient)"
                                            filter="url(#glow)"
                                        />
                                        <Area
                                            type="monotone"
                                            dataKey="blocked"
                                            name="Engellenen"
                                            stroke="#22c55e"
                                            strokeWidth={2}
                                            fill="url(#blockedGradient)"
                                        />
                                    </AreaChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-slate-400">
                                    <Activity className="w-12 h-12 mb-3 opacity-50 animate-pulse" />
                                    <p>Veri bekleniyor...</p>
                                </div>
                            )}
                        </div>
                    </Card>
                </div>

                {/* Attack Types - Premium Pie */}
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-orange-500/50 to-red-500/50 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-gradient-to-br from-orange-500 to-red-500 rounded-lg">
                                    <AlertTriangle className="w-4 h-4 text-white" />
                                </div>
                                <h3 className="text-lg font-semibold text-white">Saldırı Türleri</h3>
                            </div>
                        </div>
                        <div className="h-72 flex items-center">
                            {typeData.length > 0 ? (
                                <>
                                    <ResponsiveContainer width="55%" height="100%">
                                        <PieChart>
                                            <defs>
                                                {PIE_COLORS.map((color, index) => (
                                                    <linearGradient key={index} id={`pieGradient${index}`} x1="0" y1="0" x2="1" y2="1">
                                                        <stop offset="0%" stopColor={color} stopOpacity={1} />
                                                        <stop offset="100%" stopColor={color} stopOpacity={0.6} />
                                                    </linearGradient>
                                                ))}
                                                <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                                                    <feDropShadow dx="0" dy="4" stdDeviation="8" floodOpacity="0.3" />
                                                </filter>
                                            </defs>
                                            <Pie
                                                data={typeData}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={55}
                                                outerRadius={85}
                                                paddingAngle={4}
                                                dataKey="value"
                                                filter="url(#shadow)"
                                            >
                                                {typeData.map((entry, index) => (
                                                    <Cell
                                                        key={`cell-${index}`}
                                                        fill={`url(#pieGradient${index % PIE_COLORS.length})`}
                                                        stroke={PIE_COLORS[index % PIE_COLORS.length]}
                                                        strokeWidth={2}
                                                    />
                                                ))}
                                            </Pie>
                                            <Tooltip content={<CustomTooltip />} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                    <div className="flex-1 space-y-3">
                                        {typeData.slice(0, 6).map((item, idx) => (
                                            <div key={idx} className="flex items-center justify-between p-2 rounded-lg hover:bg-slate-800/50 transition-colors">
                                                <div className="flex items-center gap-3">
                                                    <div
                                                        className="w-3 h-3 rounded-full shadow-lg"
                                                        style={{
                                                            backgroundColor: PIE_COLORS[idx],
                                                            boxShadow: `0 0 10px ${PIE_COLORS[idx]}50`
                                                        }}
                                                    />
                                                    <span className="text-slate-300 text-sm">{item.name}</span>
                                                </div>
                                                <span className="text-white font-bold">{item.value}%</span>
                                            </div>
                                        ))}
                                    </div>
                                </>
                            ) : (
                                <div className="flex flex-col items-center justify-center w-full text-slate-400">
                                    <Activity className="w-12 h-12 mb-3 opacity-50 animate-pulse" />
                                    <p>Veri bekleniyor...</p>
                                </div>
                            )}
                        </div>
                    </Card>
                </div>
            </div>

            {/* Charts Row 2 */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Severity Bar Chart - Premium */}
                <div className="lg:col-span-2 relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-yellow-500/50 to-green-500/50 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-center gap-3 mb-6">
                            <div className="p-2 bg-gradient-to-br from-yellow-500 to-green-500 rounded-lg">
                                <BarChart3 className="w-4 h-4 text-white" />
                            </div>
                            <h3 className="text-lg font-semibold text-white">Şiddet Dağılımı</h3>
                        </div>
                        <div className="h-64">
                            {severityData.length > 0 ? (
                                <ResponsiveContainer width="100%" height="100%">
                                    <BarChart data={severityData} barGap={8}>
                                        <defs>
                                            <linearGradient id="criticalBar" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#ef4444" stopOpacity={1} />
                                                <stop offset="100%" stopColor="#ef4444" stopOpacity={0.6} />
                                            </linearGradient>
                                            <linearGradient id="highBar" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#f97316" stopOpacity={1} />
                                                <stop offset="100%" stopColor="#f97316" stopOpacity={0.6} />
                                            </linearGradient>
                                            <linearGradient id="mediumBar" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#eab308" stopOpacity={1} />
                                                <stop offset="100%" stopColor="#eab308" stopOpacity={0.6} />
                                            </linearGradient>
                                            <linearGradient id="lowBar" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="0%" stopColor="#22c55e" stopOpacity={1} />
                                                <stop offset="100%" stopColor="#22c55e" stopOpacity={0.6} />
                                            </linearGradient>
                                        </defs>
                                        <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                        <XAxis dataKey="name" stroke="#64748b" fontSize={12} />
                                        <YAxis stroke="#64748b" fontSize={10} />
                                        <Tooltip content={<CustomTooltip />} />
                                        <Bar
                                            dataKey="value"
                                            name="Saldırı Sayısı"
                                            radius={[8, 8, 0, 0]}
                                        >
                                            {severityData.map((entry, index) => (
                                                <Cell
                                                    key={`cell-${index}`}
                                                    fill={
                                                        entry.name === 'critical' ? 'url(#criticalBar)' :
                                                            entry.name === 'high' ? 'url(#highBar)' :
                                                                entry.name === 'medium' ? 'url(#mediumBar)' : 'url(#lowBar)'
                                                    }
                                                />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            ) : (
                                <div className="flex flex-col items-center justify-center h-full text-slate-400">
                                    <Activity className="w-12 h-12 mb-3 opacity-50 animate-pulse" />
                                    <p>Veri bekleniyor...</p>
                                </div>
                            )}
                        </div>
                    </Card>
                </div>

                {/* System Resources - Radial Gauges */}
                <div className="relative group">
                    <div className="absolute -inset-0.5 bg-gradient-to-r from-cyan-500/50 to-blue-500/50 rounded-2xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
                    <Card className="relative bg-slate-900/80 backdrop-blur-xl border-slate-800/50">
                        <div className="flex items-center justify-between mb-6">
                            <div className="flex items-center gap-3">
                                <div className="p-2 bg-gradient-to-br from-cyan-500 to-blue-500 rounded-lg">
                                    <Server className="w-4 h-4 text-white" />
                                </div>
                                <h3 className="text-lg font-semibold text-white">Sistem</h3>
                            </div>
                            {isConnected && (
                                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                            )}
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <RadialProgress value={cpu} color="#3b82f6" label="CPU" icon={Cpu} />
                            <RadialProgress value={memory} color="#8b5cf6" label="RAM" icon={Server} />
                            <RadialProgress value={disk} color="#22c55e" label="Disk" icon={HardDrive} />
                            <RadialProgress
                                value={Math.min(100, (systemStats?.network_recv || 0) / 10000000)}
                                color="#06b6d4"
                                label="Ağ"
                                icon={Wifi}
                            />
                        </div>

                        <div className="mt-6 pt-4 border-t border-slate-700/50">
                            <div className="flex items-center justify-between">
                                <span className="text-slate-400 text-sm">WebSocket</span>
                                <Badge variant={isConnected ? 'success' : 'danger'} size="sm">
                                    {isConnected ? '🟢 Bağlı' : '🔴 Offline'}
                                </Badge>
                            </div>
                            <div className="flex items-center justify-between mt-2">
                                <span className="text-slate-400 text-sm">Aktif Tehditler</span>
                                <span className="text-orange-400 font-bold">{threats.length}</span>
                            </div>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
}
