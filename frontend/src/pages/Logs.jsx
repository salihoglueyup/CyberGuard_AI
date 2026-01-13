import { useState, useEffect, useCallback } from 'react';
import {
    FileText, Search, Filter, RefreshCw, Download, Trash2,
    AlertCircle, AlertTriangle, Info, CheckCircle, Clock
} from 'lucide-react';
import { Card, Button, Badge, Table, Dropdown } from '../components/ui';
import { SearchInput } from '../components/ui/Input';
import { useToast } from '../components/ui/Toast';

const API_BASE = 'http://localhost:8000/api';

const levelIcons = {
    info: <Info className="w-4 h-4 text-blue-400" />,
    warning: <AlertTriangle className="w-4 h-4 text-yellow-400" />,
    error: <AlertCircle className="w-4 h-4 text-red-400" />,
    debug: <CheckCircle className="w-4 h-4 text-slate-400" />,
    success: <CheckCircle className="w-4 h-4 text-green-400" />,
};

const levelBadges = {
    info: 'info',
    warning: 'warning',
    error: 'danger',
    debug: 'default',
    success: 'success',
};

export default function Logs() {
    const [logs, setLogs] = useState([]);
    const [filteredLogs, setFilteredLogs] = useState([]);
    const [stats, setStats] = useState(null);
    const [searchQuery, setSearchQuery] = useState('');
    const [levelFilter, setLevelFilter] = useState('all');
    const [sourceFilter, setSourceFilter] = useState('all');
    const [loading, setLoading] = useState(true);
    const toast = useToast();

    const loadLogs = useCallback(async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/logs?limit=100`);
            const data = await res.json();

            if (data.success) {
                setLogs(data.data || []);
            } else {
                setLogs([]);
            }
        } catch (error) {
            console.error('Logs load error:', error);
            setLogs([]);
        } finally {
            setLoading(false);
        }
    }, []);

    const loadStats = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/logs/stats`);
            const data = await res.json();

            if (data.success) {
                setStats(data.data);
            }
        } catch (error) {
            console.error('Stats load error:', error);
        }
    }, []);

    useEffect(() => {
        loadLogs();
        loadStats();
    }, [loadLogs, loadStats]);

    useEffect(() => {
        filterLogs();
    }, [logs, searchQuery, levelFilter, sourceFilter]);

    const filterLogs = () => {
        let filtered = [...logs];

        if (searchQuery) {
            filtered = filtered.filter(log =>
                (log.message || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
                (log.source || '').toLowerCase().includes(searchQuery.toLowerCase())
            );
        }

        if (levelFilter !== 'all') {
            filtered = filtered.filter(log => log.level === levelFilter);
        }

        if (sourceFilter !== 'all') {
            filtered = filtered.filter(log => log.source === sourceFilter);
        }

        setFilteredLogs(filtered);
    };

    const clearLogs = async () => {
        if (!confirm('Tüm logları silmek istediğinizden emin misiniz?')) return;

        try {
            const res = await fetch(`${API_BASE}/logs`, { method: 'DELETE' });
            const data = await res.json();

            if (data.success) {
                setLogs([]);
                toast.success('Loglar temizlendi');
            } else {
                toast.error('Hata: ' + data.error);
            }
        } catch (error) {
            toast.error('Silme hatası: ' + error.message);
        }
    };

    const exportLogs = () => {
        const content = filteredLogs.map(log =>
            `[${log.timestamp}] [${(log.level || 'info').toUpperCase()}] [${log.source || 'System'}] ${log.message || ''}`
        ).join('\n');

        const blob = new Blob([content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `cyberguard_logs_${new Date().toISOString().split('T')[0]}.txt`;
        a.click();
        toast.success('Loglar indirildi');
    };

    const sources = [...new Set(logs.map(log => log.source).filter(Boolean))];

    // İstatistik hesaplama
    const infoCount = stats?.by_level?.info || logs.filter(l => l.level === 'info').length;
    const warningCount = stats?.by_level?.warning || logs.filter(l => l.level === 'warning').length;
    const errorCount = stats?.by_level?.error || logs.filter(l => l.level === 'error').length;
    const debugCount = stats?.by_level?.debug || logs.filter(l => l.level === 'debug').length;

    const columns = [
        {
            key: 'timestamp',
            label: 'Zaman',
            width: '180px',
            render: (val) => (
                <span className="text-slate-400 text-xs font-mono">
                    {val ? new Date(val).toLocaleString('tr-TR') : '-'}
                </span>
            )
        },
        {
            key: 'level',
            label: 'Seviye',
            width: '100px',
            render: (val) => (
                <Badge variant={levelBadges[val] || 'default'} size="sm">
                    {levelIcons[val] || levelIcons.info}
                    <span className="capitalize ml-1">{val || 'info'}</span>
                </Badge>
            )
        },
        {
            key: 'source',
            label: 'Kaynak',
            width: '120px',
            render: (val) => <span className="text-blue-400">{val || 'System'}</span>
        },
        {
            key: 'message',
            label: 'Mesaj',
            render: (val) => <span className="text-white">{val || '-'}</span>
        },
    ];

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <FileText className="w-7 h-7 text-blue-400" />
                        Sistem Logları
                    </h1>
                    <p className="text-slate-400 mt-1">Gerçek zamanlı uygulama logları</p>
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" icon={RefreshCw} onClick={() => { loadLogs(); loadStats(); }}>
                        Yenile
                    </Button>
                    <Button variant="secondary" size="sm" icon={Download} onClick={exportLogs}>
                        İndir
                    </Button>
                    <Button variant="danger" size="sm" icon={Trash2} onClick={clearLogs}>
                        Temizle
                    </Button>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <Card className="p-4 cursor-pointer hover:border-blue-500/50 transition-colors" onClick={() => setLevelFilter(levelFilter === 'info' ? 'all' : 'info')}>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Info</p>
                            <p className="text-2xl font-bold text-white">{infoCount}</p>
                        </div>
                        {levelIcons.info}
                    </div>
                </Card>
                <Card className="p-4 cursor-pointer hover:border-yellow-500/50 transition-colors" onClick={() => setLevelFilter(levelFilter === 'warning' ? 'all' : 'warning')}>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Warning</p>
                            <p className="text-2xl font-bold text-yellow-400">{warningCount}</p>
                        </div>
                        {levelIcons.warning}
                    </div>
                </Card>
                <Card className="p-4 cursor-pointer hover:border-red-500/50 transition-colors" onClick={() => setLevelFilter(levelFilter === 'error' ? 'all' : 'error')}>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Error</p>
                            <p className="text-2xl font-bold text-red-400">{errorCount}</p>
                        </div>
                        {levelIcons.error}
                    </div>
                </Card>
                <Card className="p-4 cursor-pointer hover:border-slate-500/50 transition-colors" onClick={() => setLevelFilter(levelFilter === 'debug' ? 'all' : 'debug')}>
                    <div className="flex items-center justify-between">
                        <div>
                            <p className="text-slate-400 text-sm">Debug</p>
                            <p className="text-2xl font-bold text-white">{debugCount}</p>
                        </div>
                        {levelIcons.debug}
                    </div>
                </Card>
            </div>

            {/* Filters */}
            <Card>
                <div className="flex items-center gap-4 flex-wrap">
                    <div className="flex-1 min-w-[200px]">
                        <SearchInput
                            value={searchQuery}
                            onChange={setSearchQuery}
                            placeholder="Log ara..."
                        />
                    </div>
                    <Dropdown
                        value={levelFilter}
                        onChange={setLevelFilter}
                        icon={Filter}
                        options={[
                            { value: 'all', label: 'Tüm Seviyeler' },
                            { value: 'info', label: '🔵 Info' },
                            { value: 'warning', label: '🟡 Warning' },
                            { value: 'error', label: '🔴 Error' },
                            { value: 'debug', label: '⚪ Debug' },
                        ]}
                    />
                    <Dropdown
                        value={sourceFilter}
                        onChange={setSourceFilter}
                        options={[
                            { value: 'all', label: 'Tüm Kaynaklar' },
                            ...sources.map(s => ({ value: s, label: s }))
                        ]}
                    />
                    <Badge variant="primary">{filteredLogs.length} kayıt</Badge>
                    {loading && <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-blue-500"></div>}
                </div>
            </Card>

            {/* Logs Table */}
            <Card className="p-0">
                {logs.length === 0 && !loading ? (
                    <div className="flex flex-col items-center justify-center py-12 text-slate-400">
                        <FileText className="w-12 h-12 mb-3 opacity-50" />
                        <p>Henüz log kaydı yok</p>
                        <p className="text-sm mt-1">Uygulama logları burada görünecek</p>
                    </div>
                ) : (
                    <Table
                        columns={columns}
                        data={filteredLogs}
                        emptyMessage="Log bulunamadı"
                    />
                )}
            </Card>
        </div>
    );
}
