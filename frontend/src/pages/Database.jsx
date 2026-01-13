import { useState, useEffect } from 'react';
import {
    Database, Table as TableIcon, RefreshCw, Plus, Trash2,
    Eye, Edit, Search, Filter, Download, Upload, HardDrive
} from 'lucide-react';
import { Card, Button, Badge, Table, Modal } from '../components/ui';
import { SearchInput } from '../components/ui/Input';
import { useToast } from '../components/ui/Toast';

const API_BASE = 'http://localhost:8000/api';

export default function DatabasePage() {
    const [tables, setTables] = useState([]);
    const [selectedTable, setSelectedTable] = useState('attacks');
    const [tableData, setTableData] = useState([]);
    const [searchQuery, setSearchQuery] = useState('');
    const [loading, setLoading] = useState(true);
    const [showSqlModal, setShowSqlModal] = useState(false);
    const [sqlQuery, setSqlQuery] = useState('SELECT * FROM attacks LIMIT 20;');
    const [dbStats, setDbStats] = useState(null);
    const toast = useToast();

    useEffect(() => {
        loadDbStats();
    }, []);

    useEffect(() => {
        loadTableData();
    }, [selectedTable]);

    const loadDbStats = async () => {
        try {
            const res = await fetch(`${API_BASE}/database/stats`);
            const data = await res.json();

            if (data.success) {
                setDbStats(data.data);

                // Tablo listesini oluştur
                const tableList = Object.entries(data.data.tables || {}).map(([name, count]) => ({
                    name,
                    label: name.charAt(0).toUpperCase() + name.slice(1).replace(/_/g, ' '),
                    rows: count
                }));

                setTables(tableList.length > 0 ? tableList : [
                    { name: 'attacks', label: 'Saldırılar', rows: data.data.attacks || 0 },
                    { name: 'network_logs', label: 'Ağ Logları', rows: data.data.network_logs || 0 },
                    { name: 'scan_results', label: 'Tarama Sonuçları', rows: data.data.scan_results || 0 },
                    { name: 'system_logs', label: 'Sistem Logları', rows: data.data.system_logs || 0 },
                    { name: 'chat_history', label: 'Sohbet Geçmişi', rows: data.data.chat_history || 0 },
                ]);
            }
        } catch (error) {
            console.error('DB stats error:', error);
            // Fallback
            setTables([
                { name: 'attacks', label: 'Saldırılar', rows: 0 },
                { name: 'network_logs', label: 'Ağ Logları', rows: 0 },
                { name: 'scan_results', label: 'Tarama Sonuçları', rows: 0 },
            ]);
        } finally {
            setLoading(false);
        }
    };

    const loadTableData = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/database/table/${selectedTable}?limit=50`);
            const data = await res.json();

            if (data.success) {
                setTableData(data.data || []);
            } else {
                setTableData([]);
            }
        } catch (error) {
            console.error('Table data error:', error);
            setTableData([]);
        } finally {
            setLoading(false);
        }
    };

    const exportTable = () => {
        const json = JSON.stringify(tableData, null, 2);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${selectedTable}_export.json`;
        a.click();
        toast.success(`${selectedTable} tablosu indirildi`);
    };

    const executeSql = async () => {
        toast.info('SQL sorgusu çalıştırılıyor...');
        try {
            const res = await fetch(`${API_BASE}/database/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: sqlQuery })
            });
            const data = await res.json();

            if (data.success) {
                setTableData(data.data || []);
                toast.success(`Sorgu başarılı: ${data.data?.length || 0} kayıt`);
            } else {
                toast.error(`Hata: ${data.error}`);
            }
        } catch (error) {
            toast.error(`Sorgu hatası: ${error.message}`);
        }
        setShowSqlModal(false);
    };

    // Dinamik sütunlar
    const getColumns = () => {
        if (tableData.length === 0) return [];
        return Object.keys(tableData[0]).slice(0, 8).map(key => ({
            key,
            label: key.charAt(0).toUpperCase() + key.slice(1).replace(/_/g, ' '),
            render: (val) => {
                if (val === null || val === undefined) return <span className="text-slate-500">-</span>;
                if (typeof val === 'boolean' || val === 0 || val === 1) {
                    const boolVal = val === true || val === 1;
                    return <Badge variant={boolVal ? 'success' : 'danger'} size="sm">{boolVal ? 'Evet' : 'Hayır'}</Badge>;
                }
                if (key.includes('timestamp') || key.includes('created') || key.includes('_at')) {
                    return <span className="text-slate-400 text-xs">{new Date(val).toLocaleString('tr-TR')}</span>;
                }
                if (key === 'severity' || key === 'level') {
                    const colors = { low: 'success', medium: 'warning', high: 'danger', critical: 'danger', info: 'primary', error: 'danger' };
                    return <Badge variant={colors[val] || 'default'} size="sm">{val}</Badge>;
                }
                if (key.includes('ip')) {
                    return <span className="text-blue-400 font-mono text-xs">{String(val)}</span>;
                }
                const strVal = String(val);
                return <span className="text-white">{strVal.length > 50 ? strVal.slice(0, 50) + '...' : strVal}</span>;
            }
        }));
    };

    const filteredData = tableData.filter(row =>
        Object.values(row).some(val =>
            String(val).toLowerCase().includes(searchQuery.toLowerCase())
        )
    );

    // DB Stats
    const totalRows = tables.reduce((sum, t) => sum + (t.rows || 0), 0);
    const dbSize = dbStats?.size ? `${(dbStats.size / 1024 / 1024).toFixed(1)} MB` : '-';

    if (loading && tables.length === 0) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <Database className="w-7 h-7 text-blue-400" />
                        Veritabanı
                    </h1>
                    <p className="text-slate-400 mt-1">SQLite veritabanı yönetimi - Gerçek veriler</p>
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="secondary" size="sm" icon={Upload} onClick={() => setShowSqlModal(true)}>
                        SQL Çalıştır
                    </Button>
                    <Button variant="primary" size="sm" icon={Download} onClick={exportTable}>
                        Dışa Aktar
                    </Button>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                <Card className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-blue-500/20">
                            <Database className="w-5 h-5 text-blue-400" />
                        </div>
                        <div>
                            <p className="text-slate-400 text-sm">Toplam Tablo</p>
                            <p className="text-xl font-bold text-white">{tables.length}</p>
                        </div>
                    </div>
                </Card>
                <Card className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-purple-500/20">
                            <TableIcon className="w-5 h-5 text-purple-400" />
                        </div>
                        <div>
                            <p className="text-slate-400 text-sm">Toplam Satır</p>
                            <p className="text-xl font-bold text-white">{totalRows.toLocaleString()}</p>
                        </div>
                    </div>
                </Card>
                <Card className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-green-500/20">
                            <HardDrive className="w-5 h-5 text-green-400" />
                        </div>
                        <div>
                            <p className="text-slate-400 text-sm">Veritabanı Boyutu</p>
                            <p className="text-xl font-bold text-white">{dbSize}</p>
                        </div>
                    </div>
                </Card>
                <Card className="p-4">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-emerald-500/20">
                            <Badge variant="success" dot>Bağlı</Badge>
                        </div>
                        <div>
                            <p className="text-slate-400 text-sm">Durum</p>
                            <p className="text-xl font-bold text-emerald-400">Aktif</p>
                        </div>
                    </div>
                </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Tables List */}
                <Card className="lg:col-span-1">
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold text-white">📂 Tablolar</h3>
                        <Button variant="ghost" size="sm" icon={RefreshCw} onClick={loadDbStats} />
                    </div>
                    <div className="space-y-2">
                        {tables.map((table) => (
                            <button
                                key={table.name}
                                onClick={() => setSelectedTable(table.name)}
                                className={`
                                    w-full flex items-center justify-between p-3 rounded-lg transition-all
                                    ${selectedTable === table.name
                                        ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                                        : 'text-slate-300 hover:bg-slate-800'
                                    }
                                `}
                            >
                                <div className="flex items-center gap-2">
                                    <TableIcon className="w-4 h-4" />
                                    <span className="text-sm">{table.label}</span>
                                </div>
                                <span className="text-xs text-slate-500">{(table.rows || 0).toLocaleString()}</span>
                            </button>
                        ))}
                    </div>
                </Card>

                {/* Table View */}
                <Card className="lg:col-span-3 p-0">
                    <div className="p-4 border-b border-slate-700/50 flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <h3 className="text-lg font-semibold text-white">
                                {tables.find(t => t.name === selectedTable)?.label || selectedTable}
                            </h3>
                            <Badge variant="primary" size="sm">{filteredData.length} kayıt</Badge>
                            {loading && <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-blue-500"></div>}
                        </div>
                        <div className="flex items-center gap-2">
                            <SearchInput
                                value={searchQuery}
                                onChange={setSearchQuery}
                                placeholder="Tabloda ara..."
                                className="w-48"
                            />
                            <Button variant="ghost" size="sm" icon={RefreshCw} onClick={loadTableData} />
                        </div>
                    </div>

                    <div className="max-h-[500px] overflow-auto">
                        {tableData.length === 0 ? (
                            <div className="flex items-center justify-center h-40 text-slate-400">
                                {loading ? 'Yükleniyor...' : 'Bu tabloda veri bulunamadı'}
                            </div>
                        ) : (
                            <Table
                                columns={getColumns()}
                                data={filteredData}
                                compact
                                striped
                                emptyMessage="Veri bulunamadı"
                            />
                        )}
                    </div>
                </Card>
            </div>

            {/* SQL Modal */}
            <Modal
                isOpen={showSqlModal}
                onClose={() => setShowSqlModal(false)}
                title="SQL Sorgusu Çalıştır"
                size="lg"
            >
                <div className="space-y-4">
                    <div className="text-xs text-slate-400 mb-2">
                        ⚠️ Sadece SELECT sorguları desteklenir
                    </div>
                    <textarea
                        value={sqlQuery}
                        onChange={(e) => setSqlQuery(e.target.value)}
                        className="w-full h-40 p-4 bg-slate-800 border border-slate-700 rounded-lg text-white font-mono text-sm resize-none focus:outline-none focus:border-blue-500"
                        placeholder="SELECT * FROM attacks LIMIT 20;"
                    />
                    <div className="flex justify-end gap-2">
                        <Button variant="secondary" onClick={() => setShowSqlModal(false)}>
                            İptal
                        </Button>
                        <Button variant="primary" onClick={executeSql}>
                            Çalıştır
                        </Button>
                    </div>
                </div>
            </Modal>
        </div>
    );
}
