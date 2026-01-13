import { useState, useEffect } from 'react';
import {
    FileBarChart, Download, Trash2, Eye, Plus,
    FileText, Shield, Brain, Clock, RefreshCw, CheckCircle
} from 'lucide-react';
import { Card, Button, Badge, Table, Modal } from '../components/ui';
import { useToast } from '../components/ui/Toast';

const API_BASE = 'http://localhost:8000/api';

const typeIcons = {
    dashboard: <FileBarChart className="w-5 h-5 text-blue-400" />,
    threat: <Shield className="w-5 h-5 text-red-400" />,
    model: <Brain className="w-5 h-5 text-purple-400" />,
    security: <FileText className="w-5 h-5 text-green-400" />,
};

const typeLabels = {
    dashboard: 'Dashboard Raporu',
    threat: 'Tehdit Raporu',
    model: 'Model Raporu',
    security: 'Güvenlik Raporu'
};

export default function Reports() {
    const [reports, setReports] = useState([]);
    const [loading, setLoading] = useState(true);
    const [generating, setGenerating] = useState(false);
    const [selectedReport, setSelectedReport] = useState(null);
    const [showPreview, setShowPreview] = useState(false);
    const [showNewModal, setShowNewModal] = useState(false);
    const [newReportType, setNewReportType] = useState('dashboard');
    const toast = useToast();

    useEffect(() => {
        loadReports();
    }, []);

    const loadReports = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/reports`);
            const data = await res.json();
            if (data.success) {
                setReports(data.data || []);
            }
        } catch (error) {
            console.error('Reports load error:', error);
            toast.error('Raporlar yüklenemedi');
        } finally {
            setLoading(false);
        }
    };

    const generateReport = async () => {
        setGenerating(true);
        try {
            const res = await fetch(`${API_BASE}/reports/generate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: newReportType })
            });
            const data = await res.json();

            if (data.success) {
                toast.success('Rapor oluşturuldu!');
                setShowNewModal(false);
                loadReports();
            } else {
                toast.error(data.error || 'Rapor oluşturulamadı');
            }
        } catch (error) {
            toast.error('Rapor oluşturma hatası');
        } finally {
            setGenerating(false);
        }
    };

    const downloadReport = async (report) => {
        try {
            const res = await fetch(`${API_BASE}/reports/${report.id}/download`);
            const blob = await res.blob();

            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = report.name || 'report.json';
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);

            toast.success(`${report.name} indirildi`);
        } catch (error) {
            toast.error('İndirme hatası');
        }
    };

    const deleteReport = async (reportId) => {
        try {
            const res = await fetch(`${API_BASE}/reports/${reportId}`, { method: 'DELETE' });
            const data = await res.json();

            if (data.success) {
                setReports(reports.filter(r => r.id !== reportId));
                toast.success('Rapor silindi');
            }
        } catch (error) {
            toast.error('Silme hatası');
        }
    };

    const previewReport = (report) => {
        setSelectedReport(report);
        setShowPreview(true);
    };

    const columns = [
        {
            key: 'type',
            label: 'Tür',
            width: '60px',
            sortable: false,
            render: (val) => typeIcons[val] || <FileText className="w-5 h-5 text-slate-400" />
        },
        {
            key: 'name',
            label: 'Rapor Adı',
            render: (val, row) => (
                <div>
                    <p className="text-white font-medium">{val}</p>
                    <p className="text-xs text-slate-400">{row.typeLabel || typeLabels[row.type]}</p>
                </div>
            )
        },
        {
            key: 'size',
            label: 'Boyut',
            width: '100px',
            render: (val) => <span className="text-slate-400">{val || '-'}</span>
        },
        {
            key: 'createdAt',
            label: 'Oluşturulma',
            width: '150px',
            render: (val) => (
                <div className="flex items-center gap-1 text-slate-400 text-sm">
                    <Clock className="w-3 h-3" />
                    {val ? new Date(val).toLocaleDateString('tr-TR') : '-'}
                </div>
            )
        },
        {
            key: 'actions',
            label: 'İşlemler',
            width: '150px',
            sortable: false,
            render: (_, row) => (
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => previewReport(row)}
                        className="p-1.5 rounded-lg hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
                        title="Önizle"
                    >
                        <Eye className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => downloadReport(row)}
                        className="p-1.5 rounded-lg hover:bg-slate-700 text-slate-400 hover:text-blue-400 transition-colors"
                        title="İndir"
                    >
                        <Download className="w-4 h-4" />
                    </button>
                    <button
                        onClick={() => deleteReport(row.id)}
                        className="p-1.5 rounded-lg hover:bg-slate-700 text-slate-400 hover:text-red-400 transition-colors"
                        title="Sil"
                    >
                        <Trash2 className="w-4 h-4" />
                    </button>
                </div>
            )
        },
    ];

    const stats = [
        { label: 'Toplam Rapor', value: reports.length, icon: FileBarChart, color: 'blue' },
        { label: 'Dashboard', value: reports.filter(r => r.type === 'dashboard').length, icon: FileBarChart, color: 'blue' },
        { label: 'Tehdit', value: reports.filter(r => r.type === 'threat').length, icon: Shield, color: 'red' },
        { label: 'Model', value: reports.filter(r => r.type === 'model').length, icon: Brain, color: 'purple' },
    ];

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <FileBarChart className="w-7 h-7 text-blue-400" />
                        Raporlar
                    </h1>
                    <p className="text-slate-400 mt-1">Oluşturulan tüm raporları görüntüle ve yönet</p>
                </div>
                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" icon={RefreshCw} onClick={loadReports}>
                        Yenile
                    </Button>
                    <Button variant="primary" size="sm" icon={Plus} onClick={() => setShowNewModal(true)}>
                        Yeni Rapor
                    </Button>
                </div>
            </div>

            {/* Stats */}
            <div className="grid grid-cols-4 gap-4">
                {stats.map((stat, idx) => (
                    <Card key={idx} className="p-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-slate-400 text-sm">{stat.label}</p>
                                <p className="text-2xl font-bold text-white">{stat.value}</p>
                            </div>
                            <div className={`p-2 rounded-lg bg-${stat.color}-500/20`}>
                                <stat.icon className={`w-5 h-5 text-${stat.color}-400`} />
                            </div>
                        </div>
                    </Card>
                ))}
            </div>

            {/* Reports Table */}
            <Card className="p-0">
                {loading ? (
                    <div className="flex items-center justify-center py-12">
                        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
                    </div>
                ) : (
                    <Table
                        columns={columns}
                        data={reports}
                        emptyMessage="Henüz rapor oluşturulmamış. 'Yeni Rapor' butonuna tıklayın."
                    />
                )}
            </Card>

            {/* New Report Modal */}
            <Modal
                isOpen={showNewModal}
                onClose={() => setShowNewModal(false)}
                title="Yeni Rapor Oluştur"
            >
                <div className="space-y-4">
                    <p className="text-slate-400">Oluşturmak istediğiniz rapor türünü seçin:</p>

                    <div className="grid grid-cols-2 gap-3">
                        {Object.entries(typeLabels).map(([type, label]) => (
                            <button
                                key={type}
                                onClick={() => setNewReportType(type)}
                                className={`p-4 rounded-xl border-2 transition-all text-left ${newReportType === type
                                        ? 'border-blue-500 bg-blue-500/10'
                                        : 'border-slate-700 hover:border-slate-600'
                                    }`}
                            >
                                <div className="flex items-center gap-3">
                                    {typeIcons[type]}
                                    <span className="text-white font-medium">{label}</span>
                                </div>
                                {newReportType === type && (
                                    <CheckCircle className="w-4 h-4 text-blue-400 mt-2" />
                                )}
                            </button>
                        ))}
                    </div>

                    <div className="flex justify-end gap-2 pt-4">
                        <Button variant="secondary" onClick={() => setShowNewModal(false)}>
                            İptal
                        </Button>
                        <Button
                            variant="primary"
                            icon={FileBarChart}
                            onClick={generateReport}
                            loading={generating}
                        >
                            Rapor Oluştur
                        </Button>
                    </div>
                </div>
            </Modal>

            {/* Preview Modal */}
            <Modal
                isOpen={showPreview}
                onClose={() => setShowPreview(false)}
                title="Rapor Önizleme"
                size="lg"
            >
                {selectedReport && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-4 p-4 bg-slate-800/50 rounded-xl">
                            {typeIcons[selectedReport.type]}
                            <div>
                                <p className="font-semibold text-white">{selectedReport.name}</p>
                                <p className="text-sm text-slate-400">{selectedReport.typeLabel || typeLabels[selectedReport.type]}</p>
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 bg-slate-800/30 rounded-lg">
                                <p className="text-xs text-slate-400">Boyut</p>
                                <p className="text-white font-medium">{selectedReport.size || '-'}</p>
                            </div>
                            <div className="p-3 bg-slate-800/30 rounded-lg">
                                <p className="text-xs text-slate-400">Oluşturulma Tarihi</p>
                                <p className="text-white font-medium">
                                    {selectedReport.createdAt ? new Date(selectedReport.createdAt).toLocaleDateString('tr-TR') : '-'}
                                </p>
                            </div>
                        </div>

                        {/* Rapor verisi */}
                        {selectedReport.data && (
                            <div className="p-4 bg-slate-800/30 rounded-lg">
                                <p className="text-xs text-slate-400 mb-2">Rapor İçeriği</p>
                                <pre className="text-sm text-white overflow-auto max-h-40">
                                    {JSON.stringify(selectedReport.data, null, 2)}
                                </pre>
                            </div>
                        )}

                        <div className="flex justify-end gap-2">
                            <Button variant="secondary" onClick={() => setShowPreview(false)}>
                                Kapat
                            </Button>
                            <Button variant="primary" icon={Download} onClick={() => downloadReport(selectedReport)}>
                                İndir
                            </Button>
                        </div>
                    </div>
                )}
            </Modal>
        </div>
    );
}
