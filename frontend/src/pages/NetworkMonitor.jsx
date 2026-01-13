import { useState, useEffect, useCallback } from 'react';
import {
    Network, Wifi, Activity, Server, Globe, RefreshCw,
    ArrowUpRight, ArrowDownRight, Monitor, Shield, AlertTriangle
} from 'lucide-react';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';
import { Card, Button, Badge, Table } from '../components/ui';
import { ProgressBar } from '../components/ui/Progress';

const API_BASE = 'http://localhost:8000/api';

export default function NetworkMonitor() {
    const [status, setStatus] = useState(null);
    const [connections, setConnections] = useState([]);
    const [interfaces, setInterfaces] = useState([]);
    const [bandwidth, setBandwidth] = useState([]);
    const [loading, setLoading] = useState(true);
    const [autoRefresh, setAutoRefresh] = useState(true);

    const loadStatus = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/network/status`);
            const data = await res.json();
            if (data.success) {
                // Handle both response structures
                setStatus(data.data || data);
            }
        } catch (error) {
            console.error('Status load error:', error);
        }
    }, []);

    const loadConnections = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/network/connections?limit=30`);
            const data = await res.json();
            if (data.success) {
                const connections = data.data?.connections || data.data || data.connections || [];
                setConnections(Array.isArray(connections) ? connections : []);
            }
        } catch (error) {
            console.error('Connections load error:', error);
        }
    }, []);

    const loadInterfaces = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/network/interfaces`);
            const data = await res.json();
            if (data.success) {
                const interfaces = data.data?.interfaces || data.data || data.interfaces || [];
                setInterfaces(Array.isArray(interfaces) ? interfaces : []);
            }
        } catch (error) {
            console.error('Interfaces load error:', error);
        }
    }, []);

    const loadBandwidth = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/network/bandwidth?minutes=30`);
            const data = await res.json();
            if (data.success) {
                const history = data.data?.history || data.data || data.history || [];
                setBandwidth(Array.isArray(history) ? history : []);
            }
        } catch (error) {
            console.error('Bandwidth load error:', error);
        }
    }, []);

    const loadAll = useCallback(async () => {
        await Promise.all([
            loadStatus(),
            loadConnections(),
            loadInterfaces(),
            loadBandwidth()
        ]);
        setLoading(false);
    }, [loadStatus, loadConnections, loadInterfaces, loadBandwidth]);

    useEffect(() => {
        loadAll();

        // Auto refresh her 5 saniyede
        let interval;
        if (autoRefresh) {
            interval = setInterval(loadAll, 5000);
        }

        return () => clearInterval(interval);
    }, [autoRefresh, loadAll]);

    const formatBytes = (bytes) => {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    const connectionColumns = [
        {
            key: 'local_address',
            label: 'Yerel Adres',
            render: (val) => <span className="font-mono text-xs text-blue-400">{val}</span>
        },
        {
            key: 'remote_address',
            label: 'Uzak Adres',
            render: (val) => <span className="font-mono text-xs text-slate-300">{val}</span>
        },
        {
            key: 'protocol',
            label: 'Protokol',
            width: '80px',
            render: (val) => <Badge variant="primary" size="sm">{val}</Badge>
        },
        {
            key: 'status',
            label: 'Durum',
            width: '120px',
            render: (val) => (
                <Badge
                    variant={val === 'ESTABLISHED' ? 'success' : val === 'LISTENING' ? 'info' : 'default'}
                    size="sm"
                >
                    {val}
                </Badge>
            )
        },
        {
            key: 'application',
            label: 'Uygulama',
            width: '100px',
            render: (val) => <span className="text-slate-400 text-xs">{val || '-'}</span>
        }
    ];

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="animate-spin w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full" />
            </div>
        );
    }

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <Network className="w-7 h-7 text-blue-400" />
                        Ağ İzleme
                    </h1>
                    <p className="text-slate-400 mt-1">Gerçek zamanlı ağ trafiği ve bağlantılar</p>
                </div>
                <div className="flex items-center gap-3">
                    <label className="flex items-center gap-2 text-sm text-slate-400">
                        <input
                            type="checkbox"
                            checked={autoRefresh}
                            onChange={(e) => setAutoRefresh(e.target.checked)}
                            className="rounded"
                        />
                        Otomatik yenile (5s)
                    </label>
                    <Button variant="secondary" size="sm" icon={RefreshCw} onClick={loadAll}>
                        Yenile
                    </Button>
                </div>
            </div>

            {/* Status Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card className="p-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Durum</p>
                            <p className="text-xl font-bold text-emerald-400">{status?.status?.toUpperCase() || 'ONLINE'}</p>
                        </div>
                        <div className="p-2 rounded-lg bg-emerald-500/20">
                            <Activity className="w-5 h-5 text-emerald-400" />
                        </div>
                    </div>
                </Card>

                <Card className="p-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Aktif Bağlantı</p>
                            <p className="text-xl font-bold text-white">{status?.active_connections || connections.length}</p>
                        </div>
                        <div className="p-2 rounded-lg bg-blue-500/20">
                            <Globe className="w-5 h-5 text-blue-400" />
                        </div>
                    </div>
                </Card>

                <Card className="p-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">İndirme</p>
                            <p className="text-xl font-bold text-green-400">{formatBytes(status?.bytes_recv)}</p>
                        </div>
                        <ArrowDownRight className="w-6 h-6 text-green-400" />
                    </div>
                </Card>

                <Card className="p-4">
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Yükleme</p>
                            <p className="text-xl font-bold text-blue-400">{formatBytes(status?.bytes_sent)}</p>
                        </div>
                        <ArrowUpRight className="w-6 h-6 text-blue-400" />
                    </div>
                </Card>
            </div>

            {/* System Stats */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Bandwidth Chart */}
                <Card className="lg:col-span-2">
                    <h3 className="text-lg font-semibold text-white mb-4">📈 Bant Genişliği</h3>
                    <div className="h-64">
                        <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={bandwidth}>
                                <defs>
                                    <linearGradient id="colorDownload" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorUpload" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                                <XAxis dataKey="minute" stroke="#64748b" fontSize={10} />
                                <YAxis stroke="#64748b" fontSize={10} />
                                <Tooltip
                                    contentStyle={{ backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px' }}
                                    formatter={(value) => [`${value} KB/s`, '']}
                                />
                                <Area type="monotone" dataKey="download" name="İndirme" stroke="#22c55e" fill="url(#colorDownload)" />
                                <Area type="monotone" dataKey="upload" name="Yükleme" stroke="#3b82f6" fill="url(#colorUpload)" />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* System Resources */}
                <Card>
                    <h3 className="text-lg font-semibold text-white mb-4">💻 Sistem Kaynakları</h3>
                    <div className="space-y-4">
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-400">CPU</span>
                                <span className="text-white">{status?.cpu_percent || 0}%</span>
                            </div>
                            <ProgressBar
                                value={status?.cpu_percent || 0}
                                variant={status?.cpu_percent > 80 ? 'danger' : 'primary'}
                            />
                        </div>
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-400">Bellek</span>
                                <span className="text-white">{status?.memory_percent || 0}%</span>
                            </div>
                            <ProgressBar
                                value={status?.memory_percent || 0}
                                variant={status?.memory_percent > 80 ? 'danger' : 'success'}
                            />
                        </div>
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-400">Paket Gönderilen</span>
                                <span className="text-white">{(status?.packets_sent || 0).toLocaleString()}</span>
                            </div>
                        </div>
                        <div>
                            <div className="flex justify-between text-sm mb-1">
                                <span className="text-slate-400">Paket Alınan</span>
                                <span className="text-white">{(status?.packets_recv || 0).toLocaleString()}</span>
                            </div>
                        </div>
                    </div>
                </Card>
            </div>

            {/* Network Interfaces */}
            <Card>
                <h3 className="text-lg font-semibold text-white mb-4">🌐 Ağ Arayüzleri</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {Array.isArray(interfaces) && interfaces.map((iface, idx) => (
                        <div key={idx} className="p-4 bg-slate-800/50 rounded-xl border border-slate-700/50">
                            <div className="flex items-center justify-between mb-3">
                                <div className="flex items-center gap-2">
                                    <Wifi className="w-5 h-5 text-blue-400" />
                                    <span className="text-white font-medium">{iface.name}</span>
                                </div>
                                <Badge variant={iface.status === 'up' ? 'success' : 'default'} size="sm">
                                    {iface.status}
                                </Badge>
                            </div>
                            <div className="space-y-1 text-sm">
                                <div className="flex justify-between">
                                    <span className="text-slate-400">IP:</span>
                                    <span className="text-slate-300 font-mono">{iface.ip || '-'}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-400">Hız:</span>
                                    <span className="text-slate-300">{iface.speed}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-400">TX:</span>
                                    <span className="text-green-400">{formatBytes(iface.bytes_sent)}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span className="text-slate-400">RX:</span>
                                    <span className="text-blue-400">{formatBytes(iface.bytes_recv)}</span>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>

            {/* Connections Table */}
            <Card className="p-0">
                <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
                    <h3 className="text-lg font-semibold text-white">🔗 Aktif Bağlantılar</h3>
                    <Badge variant="primary">{connections.length} bağlantı</Badge>
                </div>
                <div className="max-h-80 overflow-auto">
                    <Table
                        columns={connectionColumns}
                        data={connections}
                        compact
                        striped
                        emptyMessage="Aktif bağlantı yok"
                    />
                </div>
            </Card>
        </div>
    );
}
