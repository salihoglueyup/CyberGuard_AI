import { useEffect, useState } from 'react';
import {
    Shield, AlertTriangle, Activity, Brain, Zap,
    TrendingUp, Server, Cpu, HardDrive, Wifi, RefreshCw,
    ArrowUpRight, Clock, Target, FileText, Download,
    Radio, MapPin
} from 'lucide-react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell
} from 'recharts';
import { Card, Badge, Button, Skeleton } from '../components/ui';
import { ProgressBar } from '../components/ui/Progress';
import { useToast } from '../components/ui/Toast';
import { dashboardApi, attacksApi, modelsApi } from '../services/api';
import { useNotificationStore } from '../components/NotificationBell';
import ThreatMap from '../components/ThreatMap';
import { useWebSocket } from '../hooks/useWebSocket';
import { generateDashboardReport, generateThreatReport } from '../utils/generateReport';

const COLORS = ['#3b82f6', '#8b5cf6', '#22c55e', '#f59e0b', '#ef4444'];

export default function Dashboard() {
    const [stats, setStats] = useState(null);
    const [recentAttacks, setRecentAttacks] = useState([]);
    const [attacksByType, setAttacksByType] = useState([]);
    const [hourlyTrend, setHourlyTrend] = useState([]);
    const [loading, setLoading] = useState(true);
    const [generatingPdf, setGeneratingPdf] = useState(false);

    const addNotification = useNotificationStore((s) => s.addNotification);
    const toast = useToast();

    // WebSocket bağlantısı
    const { isConnected, threats, systemStats, clearThreats } = useWebSocket();

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        try {
            setLoading(true);

            const [statsRes, recentRes, typeRes, trendRes] = await Promise.all([
                dashboardApi.getStats(24),
                dashboardApi.getRecentAttacks(5),
                attacksApi.getByType(24),
                attacksApi.getTimeline(24),
            ]);

            if (statsRes.data.success) setStats(statsRes.data.data);
            if (recentRes.data.success) setRecentAttacks(recentRes.data.data);
            if (typeRes.data.success) setAttacksByType(typeRes.data.data);
            if (trendRes.data.success) setHourlyTrend(trendRes.data.data);
        } catch (error) {
            console.error('Dashboard yükleme hatası:', error);
        } finally {
            setLoading(false);
        }
    };

    // PDF Rapor oluştur
    const handleGenerateReport = async () => {
        setGeneratingPdf(true);
        try {
            const reportData = {
                stats,
                recentModels: recentAttacks,
                threatStats: {
                    total: threats.length,
                    blocked: threats.filter(t => t.blocked).length,
                    blockRate: threats.length > 0
                        ? Math.round((threats.filter(t => t.blocked).length / threats.length) * 100)
                        : 0
                },
                systemHealth: systemStats || { cpu: 34, memory: 67, storage: 45, network: 89 }
            };

            const filename = generateDashboardReport(reportData);
            toast.success(`Rapor indirildi: ${filename}`);
            addNotification({ type: 'success', title: 'Rapor Oluşturuldu', message: filename });
        } catch (error) {
            toast.error('Rapor oluşturulurken hata oluştu');
            console.error(error);
        } finally {
            setGeneratingPdf(false);
        }
    };

    // Tehdit raporu
    const handleThreatReport = () => {
        if (threats.length === 0) {
            toast.warning('Henüz tehdit verisi yok');
            return;
        }
        const filename = generateThreatReport(threats);
        toast.success(`Tehdit raporu indirildi: ${filename}`);
    };

    // Hızlı İşlemler
    const quickActions = [
        { icon: Shield, label: 'Tarama Başlat', color: 'blue', action: () => addNotification({ type: 'info', title: 'Tarama Başladı', message: 'Tam sistem taraması başlatıldı' }) },
        { icon: Brain, label: 'Model Eğit', color: 'purple', action: () => window.location.href = '/models' },
        { icon: Target, label: 'Tehditleri Gör', color: 'red', action: () => window.location.href = '/network' },
        { icon: Zap, label: 'AI Analiz', color: 'yellow', action: () => window.location.href = '/assistant' },
    ];

    // Sistem Sağlığı (WebSocket'ten veya varsayılan)
    const health = systemStats || { cpu: 34, memory: 67, disk: 45, network: 89 };
    const systemHealth = [
        { label: 'CPU Kullanımı', value: health.cpu, icon: Cpu },
        { label: 'Bellek', value: health.memory, icon: Server },
        { label: 'Depolama', value: health.disk || 45, icon: HardDrive },
        { label: 'Ağ', value: health.network, icon: Wifi },
    ];

    if (loading) {
        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {[1, 2, 3, 4].map(i => <Skeleton key={i} height="120px" />)}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Skeleton height="300px" />
                    <Skeleton height="300px" />
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6 fade-in">
            {/* Üst Bar - Bağlantı Durumu ve Aksiyon Butonları */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                    {/* WebSocket Durumu */}
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${isConnected
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                        : 'bg-red-500/20 text-red-400 border border-red-500/30'
                        }`}>
                        <Radio className={`w-4 h-4 ${isConnected ? 'animate-pulse' : ''}`} />
                        {isConnected ? 'Canlı Bağlantı' : 'Bağlantı Yok'}
                    </div>

                    {threats.length > 0 && (
                        <Badge variant="danger" dot>
                            {threats.length} aktif tehdit
                        </Badge>
                    )}
                </div>

                <div className="flex items-center gap-2">
                    <Button variant="secondary" size="sm" icon={RefreshCw} onClick={loadDashboardData}>
                        Yenile
                    </Button>
                    <Button variant="secondary" size="sm" icon={FileText} onClick={handleThreatReport}>
                        Tehdit Raporu
                    </Button>
                    <Button variant="primary" size="sm" icon={Download} onClick={handleGenerateReport} loading={generatingPdf}>
                        PDF İndir
                    </Button>
                </div>
            </div>

            {/* Hızlı İstatistikler */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {/* Toplam Model */}
                <Card className="relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Toplam Model</p>
                            <p className="text-3xl font-bold text-white mt-1">{stats?.total_models || 0}</p>
                            <div className="flex items-center gap-1 mt-2 text-emerald-400 text-sm">
                                <ArrowUpRight className="w-4 h-4" />
                                <span>+2 bu hafta</span>
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-blue-500/20">
                            <Brain className="w-6 h-6 text-blue-400" />
                        </div>
                    </div>
                </Card>

                {/* Dağıtılan */}
                <Card className="relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-emerald-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Dağıtılan</p>
                            <p className="text-3xl font-bold text-white mt-1">{stats?.deployed_models || stats?.blocked || 0}</p>
                            <div className="flex items-center gap-1 mt-2 text-emerald-400 text-sm">
                                <Shield className="w-4 h-4" />
                                <span>Aktif Koruma</span>
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-emerald-500/20">
                            <Shield className="w-6 h-6 text-emerald-400" />
                        </div>
                    </div>
                </Card>

                {/* Canlı Tehditler */}
                <Card className="relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-red-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Canlı Tehdit</p>
                            <p className="text-3xl font-bold text-white mt-1">{threats.length}</p>
                            <div className="flex items-center gap-1 mt-2 text-red-400 text-sm">
                                <Radio className="w-4 h-4 animate-pulse" />
                                <span>Gerçek Zamanlı</span>
                            </div>
                        </div>
                        <div className="p-3 rounded-xl bg-red-500/20">
                            <AlertTriangle className="w-6 h-6 text-red-400" />
                        </div>
                    </div>
                </Card>

                {/* En İyi Doğruluk */}
                <Card className="relative overflow-hidden group">
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-600/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">En İyi Doğruluk</p>
                            <p className="text-3xl font-bold text-white mt-1">
                                {((stats?.best_accuracy || 0) * 100).toFixed(1)}%
                            </p>
                            <p className="text-xs text-slate-500 mt-2 truncate max-w-[120px]">
                                {stats?.best_model || 'Model yok'}
                            </p>
                        </div>
                        <div className="p-3 rounded-xl bg-purple-500/20">
                            <TrendingUp className="w-6 h-6 text-purple-400" />
                        </div>
                    </div>
                </Card>
            </div>

            {/* Tehdit Haritası */}
            <Card className="p-0 overflow-hidden">
                <ThreatMap threats={threats} height="350px" />
            </Card>

            {/* Ana İçerik */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Trend Grafiği */}
                <Card className="lg:col-span-2">
                    <div className="flex items-center justify-between mb-4">
                        <div>
                            <h3 className="text-lg font-semibold text-white">Performans Trendi</h3>
                            <p className="text-sm text-slate-400">Zaman içinde model doğruluğu</p>
                        </div>
                    </div>
                    <div className="h-64" style={{ minHeight: '256px' }}>
                        {hourlyTrend.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={200}>
                                <AreaChart data={hourlyTrend}>
                                    <defs>
                                        <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                    <XAxis dataKey="hour" stroke="#64748b" fontSize={12} />
                                    <YAxis stroke="#64748b" fontSize={12} />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#0f172a',
                                            border: '1px solid #1e293b',
                                            borderRadius: '12px',
                                        }}
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="count"
                                        stroke="#3b82f6"
                                        strokeWidth={2}
                                        fillOpacity={1}
                                        fill="url(#colorCount)"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="flex items-center justify-center h-full text-slate-400">Veri yükleniyor...</div>
                        )}
                    </div>
                </Card>

                {/* Sağ Panel */}
                <div className="space-y-6">
                    {/* Hızlı İşlemler */}
                    <Card>
                        <h3 className="text-lg font-semibold text-white mb-4">⚡ Hızlı İşlemler</h3>
                        <div className="grid grid-cols-2 gap-3">
                            {quickActions.map((action, idx) => (
                                <button
                                    key={idx}
                                    onClick={action.action}
                                    className="p-4 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 hover:border-slate-600 transition-all group"
                                >
                                    <action.icon className={`w-6 h-6 mb-2 text-${action.color}-400 group-hover:scale-110 transition-transform`} />
                                    <p className="text-sm font-medium text-slate-300">{action.label}</p>
                                </button>
                            ))}
                        </div>
                    </Card>

                    {/* Sistem Sağlığı */}
                    <Card>
                        <div className="flex items-center justify-between mb-4">
                            <h3 className="text-lg font-semibold text-white">💻 Sistem Sağlığı</h3>
                            {isConnected && <Badge variant="success" size="sm">Canlı</Badge>}
                        </div>
                        <div className="space-y-4">
                            {systemHealth.map((item, idx) => (
                                <div key={idx} className="flex items-center gap-3">
                                    <div className="p-2 rounded-lg bg-slate-800">
                                        <item.icon className="w-4 h-4 text-slate-400" />
                                    </div>
                                    <div className="flex-1">
                                        <div className="flex justify-between text-sm mb-1">
                                            <span className="text-slate-400">{item.label}</span>
                                            <span className="text-white font-medium">{item.value}%</span>
                                        </div>
                                        <ProgressBar
                                            value={item.value}
                                            variant={item.value > 80 ? 'danger' : item.value > 60 ? 'warning' : 'success'}
                                            size="sm"
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </Card>
                </div>
            </div>

            {/* Alt Panel */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Framework Dağılımı */}
                <Card>
                    <h3 className="text-lg font-semibold text-white mb-4">📊 Model Framework'leri</h3>
                    <div className="h-48 flex items-center justify-center" style={{ minHeight: '192px' }}>
                        {attacksByType.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={150}>
                                <PieChart>
                                    <Pie
                                        data={attacksByType}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={50}
                                        outerRadius={70}
                                        paddingAngle={5}
                                        dataKey="value"
                                    >
                                        {attacksByType.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: '#0f172a',
                                            border: '1px solid #1e293b',
                                            borderRadius: '8px'
                                        }}
                                    />
                                </PieChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="text-slate-400">Veri yükleniyor...</div>
                        )}
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2 justify-center">
                        {attacksByType.slice(0, 4).map((item, idx) => (
                            <span key={idx} className="flex items-center gap-1.5 text-xs text-slate-400">
                                <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[idx % COLORS.length] }} />
                                {item.name}
                            </span>
                        ))}
                    </div>
                </Card>

                {/* Son Tehditler (Real-time) */}
                <Card className="lg:col-span-2">
                    <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                            <h3 className="text-lg font-semibold text-white">🔴 Canlı Tehditler</h3>
                            {isConnected && <Radio className="w-4 h-4 text-red-400 animate-pulse" />}
                        </div>
                        {threats.length > 0 && (
                            <Button variant="ghost" size="sm" onClick={clearThreats}>Temizle</Button>
                        )}
                    </div>
                    <div className="space-y-2 max-h-64 overflow-y-auto">
                        {threats.length === 0 ? (
                            <div className="text-center py-8 text-slate-400">
                                <MapPin className="w-10 h-10 mx-auto mb-2 opacity-50" />
                                <p>Tehdit bekleniyor...</p>
                                <p className="text-xs mt-1">{isConnected ? 'WebSocket bağlı' : 'Bağlanıyor...'}</p>
                            </div>
                        ) : (
                            threats.slice(0, 10).map((threat, idx) => (
                                <div key={threat.id || idx} className="flex items-center gap-3 p-2 rounded-lg bg-slate-800/30 hover:bg-slate-800/50 transition-colors">
                                    <div className={`w-2 h-2 rounded-full ${threat.severity === 'critical' ? 'bg-red-500' :
                                        threat.severity === 'high' ? 'bg-orange-500' :
                                            threat.severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                                        }`} />
                                    <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium text-white truncate">{threat.threat_type}</p>
                                        <p className="text-xs text-slate-400">{threat.source_ip} • {threat.country}</p>
                                    </div>
                                    <Badge
                                        variant={threat.blocked ? 'success' : 'danger'}
                                        size="sm"
                                    >
                                        {threat.blocked ? 'Engellendi' : 'Aktif'}
                                    </Badge>
                                    <span className="text-xs text-slate-500">
                                        {new Date(threat.timestamp).toLocaleTimeString('tr-TR')}
                                    </span>
                                </div>
                            ))
                        )}
                    </div>
                </Card>
            </div>
        </div>
    );
}
