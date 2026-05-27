import { useEffect, useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Shield, AlertTriangle, Activity, Brain, Zap,
    TrendingUp, Server, Cpu, HardDrive, Wifi, RefreshCw,
    ArrowUpRight, Clock, Target, FileText, Download,
    Radio, MapPin, Lock, Unlock, GripVertical
} from 'lucide-react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    PieChart, Pie, Cell
} from 'recharts';
import { Card, Badge, Button, Skeleton } from '../../components/ui';
import { ProgressBar } from '../../components/ui/Progress';
import { useToast } from '../../components/ui/Toast';
import { dashboardApi, attacksApi } from '../../services/api';
import { useNotificationStore } from '../../components/NotificationBell';
import ThreatMap from '../../components/ThreatMap';
import { useWebSocket } from '../../hooks/useWebSocket';
import { generateDashboardReport, generateThreatReport } from '../../utils/generateReport';
import { ThreatMeter, SystemStatus, AlertTicker, StatusBar, GlitchText } from '../../components/hud';
import { Responsive, useContainerWidth } from 'react-grid-layout';
import 'react-grid-layout/css/styles.css';
import { useRealtimeMetrics } from '../../hooks/useRealtimeMetrics';
import { LiveMetrics, LiveAttackFeed, LiveTrafficChart, ConnectionStats } from '../../components/realtime';
import { ThreatRadarWidget, AttackFlowWidget, AssetTreemapWidget, MiniStatWidget } from '../../components/widgets';


const HUD_COLORS = ['var(--hud-cyan)', 'var(--hud-purple)', 'var(--hud-emerald)', 'var(--hud-amber)', 'var(--hud-red)'];

const DEFAULT_LAYOUTS = {
    lg: [
        { i: 'stat-0', x: 0, y: 0, w: 3, h: 2, minW: 2, minH: 2 },
        { i: 'stat-1', x: 3, y: 0, w: 3, h: 2, minW: 2, minH: 2 },
        { i: 'stat-2', x: 6, y: 0, w: 3, h: 2, minW: 2, minH: 2 },
        { i: 'stat-3', x: 9, y: 0, w: 3, h: 2, minW: 2, minH: 2 },
        { i: 'threat-map', x: 0, y: 2, w: 9, h: 6, minW: 4, minH: 4 },
        { i: 'threat-meter', x: 9, y: 2, w: 3, h: 3, minW: 2, minH: 2 },
        { i: 'sys-status', x: 9, y: 5, w: 3, h: 3, minW: 2, minH: 2 },
        { i: 'trend-chart', x: 0, y: 8, w: 8, h: 5, minW: 4, minH: 3 },
        { i: 'alert-feed', x: 8, y: 8, w: 4, h: 5, minW: 3, minH: 3 },
        { i: 'pie-chart', x: 0, y: 13, w: 4, h: 5, minW: 3, minH: 3 },
        { i: 'sys-health', x: 4, y: 13, w: 8, h: 5, minW: 4, minH: 3 },
        { i: 'attacks-table', x: 0, y: 18, w: 12, h: 4, minW: 6, minH: 3 },
        { i: 'live-metrics', x: 0, y: 22, w: 3, h: 8, minW: 3, minH: 6 },
        { i: 'live-traffic', x: 3, y: 22, w: 5, h: 5, minW: 4, minH: 4 },
        { i: 'live-attacks', x: 8, y: 22, w: 4, h: 8, minW: 3, minH: 5 },
        { i: 'conn-stats', x: 3, y: 27, w: 5, h: 3, minW: 3, minH: 2 },
        { i: 'threat-radar', x: 0, y: 30, w: 4, h: 7, minW: 3, minH: 5 },
        { i: 'attack-flow', x: 4, y: 30, w: 4, h: 7, minW: 3, minH: 5 },
        { i: 'asset-treemap', x: 8, y: 30, w: 4, h: 7, minW: 3, minH: 5 },
        { i: 'mini-blocked', x: 0, y: 37, w: 3, h: 3, minW: 2, minH: 2 },
        { i: 'mini-threats', x: 3, y: 37, w: 3, h: 3, minW: 2, minH: 2 },
        { i: 'mini-uptime', x: 6, y: 37, w: 3, h: 3, minW: 2, minH: 2 },
        { i: 'mini-latency', x: 9, y: 37, w: 3, h: 3, minW: 2, minH: 2 },
    ],
    md: [
        { i: 'stat-0', x: 0, y: 0, w: 3, h: 2 },
        { i: 'stat-1', x: 3, y: 0, w: 3, h: 2 },
        { i: 'stat-2', x: 6, y: 0, w: 3, h: 2 },
        { i: 'stat-3', x: 9, y: 0, w: 3, h: 2 },
        { i: 'threat-map', x: 0, y: 2, w: 9, h: 6 },
        { i: 'threat-meter', x: 9, y: 2, w: 3, h: 3 },
        { i: 'sys-status', x: 9, y: 5, w: 3, h: 3 },
        { i: 'trend-chart', x: 0, y: 8, w: 7, h: 5 },
        { i: 'alert-feed', x: 7, y: 8, w: 5, h: 5 },
        { i: 'pie-chart', x: 0, y: 13, w: 4, h: 5 },
        { i: 'sys-health', x: 4, y: 13, w: 8, h: 5 },
        { i: 'attacks-table', x: 0, y: 18, w: 12, h: 4 },
        { i: 'live-metrics', x: 0, y: 22, w: 3, h: 8 },
        { i: 'live-traffic', x: 3, y: 22, w: 5, h: 5 },
        { i: 'live-attacks', x: 8, y: 22, w: 4, h: 8 },
        { i: 'conn-stats', x: 3, y: 27, w: 5, h: 3 },
        { i: 'threat-radar', x: 0, y: 30, w: 4, h: 7 },
        { i: 'attack-flow', x: 4, y: 30, w: 4, h: 7 },
        { i: 'asset-treemap', x: 8, y: 30, w: 4, h: 7 },
        { i: 'mini-blocked', x: 0, y: 37, w: 3, h: 3 },
        { i: 'mini-threats', x: 3, y: 37, w: 3, h: 3 },
        { i: 'mini-uptime', x: 6, y: 37, w: 3, h: 3 },
        { i: 'mini-latency', x: 9, y: 37, w: 3, h: 3 },
    ],
    sm: [
        { i: 'stat-0', x: 0, y: 0, w: 3, h: 2 },
        { i: 'stat-1', x: 3, y: 0, w: 3, h: 2 },
        { i: 'stat-2', x: 0, y: 2, w: 3, h: 2 },
        { i: 'stat-3', x: 3, y: 2, w: 3, h: 2 },
        { i: 'threat-map', x: 0, y: 4, w: 6, h: 5 },
        { i: 'threat-meter', x: 0, y: 9, w: 3, h: 3 },
        { i: 'sys-status', x: 3, y: 9, w: 3, h: 3 },
        { i: 'trend-chart', x: 0, y: 12, w: 6, h: 5 },
        { i: 'alert-feed', x: 0, y: 17, w: 6, h: 5 },
        { i: 'pie-chart', x: 0, y: 22, w: 6, h: 5 },
        { i: 'sys-health', x: 0, y: 27, w: 6, h: 5 },
        { i: 'attacks-table', x: 0, y: 32, w: 6, h: 4 },
        { i: 'live-metrics', x: 0, y: 36, w: 6, h: 8 },
        { i: 'live-traffic', x: 0, y: 44, w: 6, h: 5 },
        { i: 'live-attacks', x: 0, y: 49, w: 6, h: 8 },
        { i: 'conn-stats', x: 0, y: 57, w: 6, h: 3 },
        { i: 'threat-radar', x: 0, y: 60, w: 6, h: 7 },
        { i: 'attack-flow', x: 0, y: 67, w: 6, h: 7 },
        { i: 'asset-treemap', x: 0, y: 74, w: 6, h: 7 },
        { i: 'mini-blocked', x: 0, y: 81, w: 3, h: 3 },
        { i: 'mini-threats', x: 3, y: 81, w: 3, h: 3 },
        { i: 'mini-uptime', x: 0, y: 84, w: 3, h: 3 },
        { i: 'mini-latency', x: 3, y: 84, w: 3, h: 3 },
    ],
};

function getStoredLayouts() {
    try {
        const stored = localStorage.getItem('cyberguard-dashboard-layouts');
        return stored ? JSON.parse(stored) : null;
    } catch (e) { console.warn('Layout okuma hatasi:', e); return null; }
}



// Gotham stat card
function HudStatCard({ title, value, subtext, icon: Icon, color = 'var(--hud-cyan)', pulse }) {
    return (
        <div className="hud-panel group cursor-pointer relative overflow-hidden">
            <div className="flex items-start justify-between">
                <div className="space-y-1">
                    <p className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide">{title}</p>
                    <p className="text-2xl font-bold font-mono tabular-nums" style={{ color }}>{value}</p>
                    {subtext && (
                        <p className="text-[10px] font-mono text-[var(--hud-text-dim)]">{subtext}</p>
                    )}
                </div>
                <div className="p-2 rounded border border-[rgba(255,255,255,0.06)] bg-[rgba(255,255,255,0.02)]">
                    <Icon className={`w-5 h-5 ${pulse ? 'animate-pulse' : ''}`} style={{ color }} />
                </div>
            </div>
        </div>
    );
}

export default function Dashboard() {
    const { containerRef, width } = useContainerWidth();
    const [stats, setStats] = useState(null);
    const [recentAttacks, setRecentAttacks] = useState([]);
    const [attacksByType, setAttacksByType] = useState([]);
    const [hourlyTrend, setHourlyTrend] = useState([]);
    const [loading, setLoading] = useState(true);
    const [generatingPdf, setGeneratingPdf] = useState(false);
    const [gridLocked, setGridLocked] = useState(true);
    const [layouts, setLayouts] = useState(getStoredLayouts() || DEFAULT_LAYOUTS);

    const addNotification = useNotificationStore((s) => s.addNotification);
    const toast = useToast();
    const navigate = useNavigate();

    const { isConnected, threats, systemStats } = useWebSocket();
    useRealtimeMetrics(1000); // Start real-time metric simulation

    const onLayoutChange = useCallback((_, allLayouts) => {
        setLayouts(allLayouts);
        try { localStorage.setItem('cyberguard-dashboard-layouts', JSON.stringify(allLayouts)); } catch (e) { console.warn('Layout kaydetme hatasi:', e); }
    }, []);

    const resetLayout = () => {
        setLayouts(DEFAULT_LAYOUTS);
        localStorage.removeItem('cyberguard-dashboard-layouts');
    };

    useEffect(() => {
        loadDashboardData();
    }, []);

    const loadDashboardData = async () => {
        try {
            setLoading(true);
            const results = await Promise.allSettled([
                dashboardApi.getStats(24),
                dashboardApi.getRecentAttacks(5),
                attacksApi.getByType(24),
                attacksApi.getTimeline(24),
            ]);
            const [statsRes, recentRes, typeRes, trendRes] = results;
            if (statsRes.status === 'fulfilled' && statsRes.value?.data?.success)
                setStats(statsRes.value.data.data);
            if (recentRes.status === 'fulfilled' && recentRes.value?.data?.success)
                setRecentAttacks(recentRes.value.data.data);
            if (typeRes.status === 'fulfilled' && typeRes.value?.data?.success)
                setAttacksByType(typeRes.value.data.data);
            if (trendRes.status === 'fulfilled' && trendRes.value?.data?.success)
                setHourlyTrend(trendRes.value.data.data);
        } catch (error) {
            console.error('Dashboard yukleme hatasi:', error);
        } finally {
            setLoading(false);
        }
    };

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
            addNotification({ type: 'success', title: 'Rapor Olusturuldu', message: filename });
        } catch (error) {
            toast.error('Rapor olusturulurken hata olustu');
            console.error(error);
        } finally {
            setGeneratingPdf(false);
        }
    };

    const handleThreatReport = () => {
        if (threats.length === 0) {
            toast.warning('Henuz tehdit verisi yok');
            return;
        }
        const filename = generateThreatReport(threats);
        toast.success(`Tehdit raporu indirildi: ${filename}`);
    };

    const quickActions = [
        { icon: Shield, label: 'Tarama', color: 'var(--hud-cyan)', action: () => addNotification({ type: 'info', title: 'Tarama Basladi', message: 'Tam sistem taramasi baslatildi' }) },
        { icon: Brain, label: 'Model Eğit', color: 'var(--hud-purple)', action: () => navigate('/models') },
        { icon: Target, label: 'Tehditler', color: 'var(--hud-red)', action: () => navigate('/network') },
        { icon: Zap, label: 'AI Analiz', color: 'var(--hud-amber)', action: () => navigate('/assistant') },
    ];

    const health = systemStats || { cpu: 34, memory: 67, disk: 45, network: 89 };

    if (loading) {
        return (
            <div className="space-y-4 animate-pulse">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                    {[1, 2, 3, 4].map(i => (
                        <div key={i} className="hud-panel h-24">
                            <div className="h-2 w-16 bg-[rgba(56,189,248,0.05)] rounded mb-3" />
                            <div className="h-6 w-12 bg-[rgba(56,189,248,0.08)] rounded" />
                        </div>
                    ))}
                </div>
                <div className="hud-panel h-[300px] flex items-center justify-center">
                    <GlitchText text="Sistem Yükleniyor..." className="text-sm text-[var(--hud-cyan)]" />
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-4 fade-in">
            {/* Top bar */}
            <div className="flex items-center justify-between flex-wrap gap-2">
                <div className="flex items-center gap-3">
                    <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded border text-[10px] font-mono tracking-wider ${
                        isConnected
                            ? 'bg-[rgba(16,185,129,0.06)] text-[var(--hud-emerald)] border-[rgba(16,185,129,0.2)]'
                            : 'bg-[rgba(239,68,68,0.06)] text-[var(--hud-red)] border-[rgba(239,68,68,0.2)]'
                    }`}>
                        <Radio className={`w-3 h-3 ${isConnected ? 'animate-pulse' : ''}`} />
                        {isConnected ? 'Canlı Bağlantı' : 'Bağlantı Yok'}
                    </div>
                    {threats.length > 0 && (
                        <span className="text-[10px] font-mono text-[var(--hud-red)] px-2 py-0.5 bg-[rgba(239,68,68,0.06)] border border-[rgba(239,68,68,0.2)] rounded">
                            {threats.length} Aktif Tehdit
                        </span>
                    )}
                </div>

                <div className="flex items-center gap-2">
                    {quickActions.map(qa => (
                        <button
                            key={qa.label}
                            onClick={qa.action}
                            className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-[var(--hud-border)] hover:border-[rgba(56,189,248,0.3)] bg-[rgba(56,189,248,0.02)] hover:bg-[rgba(56,189,248,0.05)] transition-all text-[10px] font-mono text-[var(--hud-text-muted)] hover:text-[var(--hud-text)]"
                        >
                            <qa.icon className="w-3 h-3" style={{ color: qa.color }} />
                            {qa.label}
                        </button>
                    ))}
                    <div className="w-px h-4 bg-[var(--hud-border)]" />
                    <button onClick={loadDashboardData} className="p-1.5 rounded hover:bg-[rgba(56,189,248,0.06)] text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)] transition-colors">
                        <RefreshCw className="w-3.5 h-3.5" />
                    </button>
                    <button onClick={handleGenerateReport} disabled={generatingPdf}
                        className="flex items-center gap-1 px-2 py-1 rounded border border-[var(--hud-border)] text-[10px] font-mono text-[var(--hud-text-muted)] hover:text-[var(--hud-cyan)] hover:border-[rgba(56,189,248,0.3)] transition-all disabled:opacity-40">
                        <Download className="w-3 h-3" />
                        PDF
                    </button>
                    <button onClick={handleThreatReport}
                        className="flex items-center gap-1 px-2 py-1 rounded border border-[rgba(239,68,68,0.3)] text-[10px] font-mono text-[var(--hud-text-muted)] hover:text-[var(--hud-red)] hover:border-[rgba(239,68,68,0.5)] transition-all">
                        <FileText className="w-3 h-3" />
                        TEHDIT
                    </button>
                    <button onClick={() => setGridLocked(!gridLocked)}
                        className={`flex items-center gap-1 px-2 py-1 rounded border text-[10px] font-mono transition-all ${
                            gridLocked
                                ? 'border-[var(--hud-border)] text-[var(--hud-text-muted)] hover:text-[var(--hud-cyan)]'
                                : 'border-[rgba(56,189,248,0.3)] text-[var(--hud-cyan)] bg-[rgba(56,189,248,0.06)]'
                        }`}>
                        {gridLocked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                        {gridLocked ? 'Kilitle' : 'Düzenle'}
                    </button>
                    {!gridLocked && (
                        <button onClick={resetLayout}
                            className="flex items-center gap-1 px-2 py-1 rounded border border-[rgba(255,171,0,0.3)] text-[10px] font-mono text-[var(--hud-amber)] hover:bg-[rgba(255,171,0,0.06)] transition-all">
                            Sıfırla
                        </button>
                    )}
                </div>
            </div>

            {/* Status bar */}
            <StatusBar />

            {/* Draggable Widget Grid */}
            <div ref={containerRef}>
            <Responsive
                width={width || 1200}
                className="layout"
                layouts={layouts}
                breakpoints={{ lg: 1200, md: 996, sm: 768 }}
                cols={{ lg: 12, md: 12, sm: 6 }}
                rowHeight={40}
                isDraggable={!gridLocked}
                isResizable={!gridLocked}
                onLayoutChange={onLayoutChange}
                draggableHandle=".widget-drag-handle"
                margin={[12, 12]}
                containerPadding={[0, 0]}
            >
                {/* Stat cards */}
                {[
                    { key: 'stat-0', title: 'Toplam Model', value: stats?.total_models || 0, subtext: '+2 bu hafta', icon: Brain, color: 'var(--hud-cyan)' },
                    { key: 'stat-1', title: 'Dağıtılan', value: stats?.deployed_models || stats?.blocked || 0, subtext: 'Aktif Koruma', icon: Shield, color: 'var(--hud-emerald)' },
                    { key: 'stat-2', title: 'Canlı Tehdit', value: threats.length, subtext: 'Gercek Zamanli', icon: AlertTriangle, color: 'var(--hud-red)', pulse: true },
                    { key: 'stat-3', title: 'En İyi Doğruluk', value: `${((stats?.best_accuracy || 0) * 100).toFixed(1)}%`, subtext: stats?.best_model || 'Model yok', icon: TrendingUp, color: 'var(--hud-purple)' },
                ].map(s => (
                    <div key={s.key}>
                        <HudStatCard title={s.title} value={s.value} subtext={s.subtext} icon={s.icon} color={s.color} pulse={s.pulse} />
                    </div>
                ))}

                {/* Threat Map */}
                <div key="threat-map" className="hud-panel p-0 overflow-hidden">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <ThreatMap threats={threats} height="100%" />
                </div>

                {/* Threat Meter */}
                <div key="threat-meter" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <ThreatMeter value={Math.min(100, threats.length * 12 + 15)} />
                </div>

                {/* System Status */}
                <div key="sys-status" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <SystemStatus />
                </div>

                {/* Trend Chart */}
                <div key="trend-chart" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <div className="flex items-center justify-between mb-3">
                        <div>
                            <h3 className="text-xs font-mono text-[var(--hud-text)] tracking-wider">Performans Trendi</h3>
                            <p className="text-[9px] font-mono text-[var(--hud-text-dim)]">Model dogruluk degisimi (24 saat)</p>
                        </div>
                    </div>
                    <div className="h-[calc(100%-40px)]" style={{ minHeight: '120px' }}>
                        {hourlyTrend.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%" minWidth={100} minHeight={100}>
                                <AreaChart data={hourlyTrend}>
                                    <defs>
                                        <linearGradient id="hudCyanGrad" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#38bdf8" stopOpacity={0.2} />
                                            <stop offset="95%" stopColor="#38bdf8" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(56,189,248,0.06)" />
                                    <XAxis dataKey="hour" stroke="rgba(255,255,255,0.15)" fontSize={9} fontFamily="JetBrains Mono, monospace" />
                                    <YAxis stroke="rgba(255,255,255,0.15)" fontSize={9} fontFamily="JetBrains Mono, monospace" />
                                    <Tooltip contentStyle={{ backgroundColor: 'var(--hud-bg)', border: '1px solid var(--hud-border)', borderRadius: '4px', fontFamily: 'JetBrains Mono, monospace', fontSize: '11px' }} />
                                    <Area type="monotone" dataKey="count" stroke="var(--hud-cyan)" strokeWidth={1.5} fillOpacity={1} fill="url(#hudCyanGrad)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-[var(--hud-text-dim)] font-mono text-xs">Veri Bekleniyor...</div>
                        )}
                    </div>
                </div>

                {/* Alert Feed */}
                <div key="alert-feed" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <AlertTicker maxVisible={6} />
                </div>

                {/* Pie Chart */}
                <div key="pie-chart" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <h3 className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide mb-2">Saldırı Dağılımı</h3>
                    <div className="h-[calc(100%-30px)]" style={{ minHeight: '100px' }}>
                        {attacksByType.length > 0 ? (
                            <ResponsiveContainer width="100%" height="100%">
                                <PieChart>
                                    <Pie data={attacksByType} dataKey="count" nameKey="type" cx="50%" cy="50%" outerRadius={55} innerRadius={30} strokeWidth={0}>
                                        {attacksByType.map((_, idx) => (<Cell key={idx} fill={HUD_COLORS[idx % HUD_COLORS.length]} />))}
                                    </Pie>
                                    <Tooltip contentStyle={{ backgroundColor: 'var(--hud-bg)', border: '1px solid var(--hud-border)', borderRadius: '4px', fontFamily: 'JetBrains Mono, monospace', fontSize: '10px' }} />
                                </PieChart>
                            </ResponsiveContainer>
                        ) : (
                            <div className="h-full flex items-center justify-center text-[var(--hud-text-dim)] font-mono text-xs">Veri Yok</div>
                        )}
                    </div>
                </div>

                {/* System Health */}
                <div key="sys-health" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <h3 className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide mb-3">Sistem Sağlığı</h3>
                    <div className="grid grid-cols-2 gap-3">
                        {[
                            { label: 'CPU', value: health.cpu, icon: Cpu, color: health.cpu > 80 ? 'var(--hud-red)' : 'var(--hud-cyan)' },
                            { label: 'Bellek', value: health.memory, icon: Server, color: health.memory > 85 ? 'var(--hud-amber)' : 'var(--hud-emerald)' },
                            { label: 'Depolama', value: health.disk || 45, icon: HardDrive, color: 'var(--hud-cyan)' },
                            { label: 'Ağ', value: health.network, icon: Wifi, color: 'var(--hud-emerald)' },
                        ].map(item => (
                            <div key={item.label} className="space-y-1">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-1">
                                        <item.icon className="w-3 h-3" style={{ color: item.color }} />
                                        <span className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wider">{item.label}</span>
                                    </div>
                                    <span className="text-[10px] font-mono font-bold tabular-nums" style={{ color: item.color }}>{item.value}%</span>
                                </div>
                                <div className="h-1.5 bg-[rgba(255,255,255,0.04)] rounded-sm overflow-hidden border border-[var(--hud-border)]">
                                    <div className="h-full rounded-sm transition-all duration-700" style={{ width: `${item.value}%`, backgroundColor: item.color, boxShadow: `0 0 6px ${item.color}40` }} />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Attacks Table */}
                <div key="attacks-table" className="hud-panel overflow-auto">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <h3 className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide mb-2">Son Saldırılar</h3>
                    {recentAttacks.length > 0 ? (
                        <table className="w-full text-[10px] font-mono">
                            <thead>
                                <tr className="border-b border-[var(--hud-border)]">
                                    <th className="text-left py-1.5 text-[var(--hud-text-dim)] tracking-wider font-medium">Tip</th>
                                    <th className="text-left py-1.5 text-[var(--hud-text-dim)] tracking-wider font-medium">Kaynak</th>
                                    <th className="text-left py-1.5 text-[var(--hud-text-dim)] tracking-wider font-medium">Hedef</th>
                                    <th className="text-right py-1.5 text-[var(--hud-text-dim)] tracking-wider font-medium">Zaman</th>
                                </tr>
                            </thead>
                            <tbody>
                                {recentAttacks.map((attack, idx) => (
                                    <tr key={idx} className="border-b border-[rgba(255,255,255,0.03)] hover:bg-[rgba(56,189,248,0.03)] transition-colors">
                                        <td className="py-1.5 text-[var(--hud-red)]">{attack.type || 'N/A'}</td>
                                        <td className="py-1.5 text-[var(--hud-text-muted)]">{attack.source_ip || attack.src || 'N/A'}</td>
                                        <td className="py-1.5 text-[var(--hud-text-muted)]">{attack.dest_ip || attack.dst || 'N/A'}</td>
                                        <td className="py-1.5 text-right text-[var(--hud-text-dim)]">{attack.timestamp || attack.time || 'N/A'}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    ) : (
                        <div className="flex items-center justify-center h-full text-[var(--hud-text-dim)] font-mono text-xs">Veri Yok</div>
                    )}
                </div>

                {/* Live Metrics (Real-time sparklines) */}
                <div key="live-metrics" className="hud-panel overflow-auto">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <LiveMetrics />
                </div>

                {/* Live Traffic Chart */}
                <div key="live-traffic" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <LiveTrafficChart />
                </div>

                {/* Live Attack Feed */}
                <div key="live-attacks" className="hud-panel overflow-hidden">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <LiveAttackFeed maxItems={12} />
                </div>

                {/* Connection Stats */}
                <div key="conn-stats" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <ConnectionStats />
                </div>

                {/* Threat Radar */}
                <div key="threat-radar" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <ThreatRadarWidget />
                </div>

                {/* Attack Flow Sankey */}
                <div key="attack-flow" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <AttackFlowWidget />
                </div>

                {/* Asset Treemap */}
                <div key="asset-treemap" className="hud-panel">
                    {!gridLocked && <div className="widget-drag-handle absolute top-1 left-1 z-10 cursor-grab text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)]"><GripVertical className="w-4 h-4" /></div>}
                    <AssetTreemapWidget />
                </div>

                {/* Mini Stats */}
                <div key="mini-blocked" className="hud-panel">
                    <MiniStatWidget icon={Shield} label="Engellenen" value="12,847" trend={8.3} color="var(--hud-emerald)" sparkData={[12,15,11,18,22,19,25,21,28,24]} />
                </div>
                <div key="mini-threats" className="hud-panel">
                    <MiniStatWidget icon={AlertTriangle} label="Tehdit" value="342" trend={-12.5} trendInverted color="var(--hud-red)" sparkData={[45,38,42,35,30,28,25,22,20,18]} />
                </div>
                <div key="mini-uptime" className="hud-panel">
                    <MiniStatWidget icon={Activity} label="Uptime" value="99.97" unit="%" trend={0.02} color="var(--hud-cyan)" sparkData={[99.9,99.95,99.97,99.98,99.96,99.97,99.99,99.97,99.98,99.97]} />
                </div>
                <div key="mini-latency" className="hud-panel">
                    <MiniStatWidget icon={Zap} label="Gecikme" value="2.3" unit="ms" trend={-5.1} color="var(--hud-amber)" sparkData={[3.2,2.8,2.5,2.9,2.4,2.6,2.3,2.5,2.2,2.3]} />
                </div>

            </Responsive>
            </div>
        </div>
    );
}
