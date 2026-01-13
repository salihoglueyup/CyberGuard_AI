import React, { useState, useEffect } from 'react';
import api from '../services/api';

const ContainerSecurity = () => {
    const [status, setStatus] = useState(null);
    const [containers, setContainers] = useState([]);
    const [images, setImages] = useState([]);
    const [scanHistory, setScanHistory] = useState([]);
    const [vulnerabilities, setVulnerabilities] = useState([]);
    const [activeTab, setActiveTab] = useState('overview');
    const [loading, setLoading] = useState(true);
    const [scanning, setScanning] = useState(false);
    const [scanImage, setScanImage] = useState('');

    const loadData = async () => {
        setLoading(true);
        try {
            const [statusRes, containersRes, imagesRes, historyRes, vulnRes] = await Promise.all([
                api.get('/container/status'),
                api.get('/container/containers'),
                api.get('/container/images'),
                api.get('/container/scans'),
                api.get('/container/vulnerabilities')
            ]);

            if (statusRes.data.success) setStatus(statusRes.data.data);
            if (containersRes.data.success) setContainers(containersRes.data.data.containers || []);
            if (imagesRes.data.success) setImages(imagesRes.data.data.images || []);
            if (historyRes.data.success) setScanHistory(historyRes.data.data.scans || []);
            if (vulnRes.data.success) setVulnerabilities(vulnRes.data.data.vulnerabilities || []);
        } catch (error) {
            console.error('Container verisi yüklenirken hata:', error);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadData();
    }, []);

    const handleScan = async () => {
        if (!scanImage) return;
        setScanning(true);
        try {
            await api.post('/container/scan', { image: scanImage, scan_type: 'full' });
            await loadData();
            setScanImage('');
        } catch (error) {
            console.error('Tarama hatası:', error);
        } finally {
            setScanning(false);
        }
    };

    const getSeverityColor = (severity) => {
        const colors = {
            critical: 'text-red-400 bg-red-900/30',
            high: 'text-orange-400 bg-orange-900/30',
            medium: 'text-yellow-400 bg-yellow-900/30',
            low: 'text-green-400 bg-green-900/30'
        };
        return colors[severity] || 'text-gray-400 bg-gray-700';
    };

    const getStatusColor = (status) => {
        const colors = {
            running: 'bg-green-500',
            stopped: 'bg-gray-500',
            paused: 'bg-yellow-500',
            restarting: 'bg-blue-500'
        };
        return colors[status] || 'bg-gray-500';
    };

    const tabs = [
        { id: 'overview', label: '📊 Genel Bakış' },
        { id: 'containers', label: '📦 Container`lar' },
        { id: 'images', label: '🖼️ İmajlar' },
        { id: 'scans', label: '🔍 Taramalar' },
        { id: 'vulnerabilities', label: '⚠️ Açıklıklar' }
    ];

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-cyan-500 mx-auto"></div>
                    <p className="text-gray-400 mt-4">Container verileri yükleniyor...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-slate-900 to-gray-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                        📦 Container Güvenliği
                    </h1>
                    <p className="text-gray-400 mt-2">Docker container ve imaj güvenlik yönetimi</p>
                </div>

                {/* Tabs */}
                <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`px-4 py-2 rounded-lg font-medium transition-all whitespace-nowrap ${activeTab === tab.id
                                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg'
                                    : 'bg-gray-800/80 text-gray-400 hover:bg-gray-700'
                                }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Overview Tab */}
                {activeTab === 'overview' && (
                    <div className="space-y-6">
                        {/* Stats */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                            <div className="bg-gradient-to-br from-cyan-900/30 to-gray-900 rounded-xl p-5 border border-cyan-500/30">
                                <div className="text-3xl font-bold text-cyan-400">{status?.total_containers || 0}</div>
                                <div className="text-gray-400 text-sm">Container Sayısı</div>
                            </div>
                            <div className="bg-gradient-to-br from-green-900/30 to-gray-900 rounded-xl p-5 border border-green-500/30">
                                <div className="text-3xl font-bold text-green-400">{status?.running_containers || 0}</div>
                                <div className="text-gray-400 text-sm">Çalışan Container</div>
                            </div>
                            <div className="bg-gradient-to-br from-blue-900/30 to-gray-900 rounded-xl p-5 border border-blue-500/30">
                                <div className="text-3xl font-bold text-blue-400">{status?.total_images || 0}</div>
                                <div className="text-gray-400 text-sm">Docker İmajı</div>
                            </div>
                            <div className="bg-gradient-to-br from-red-900/30 to-gray-900 rounded-xl p-5 border border-red-500/30">
                                <div className="text-3xl font-bold text-red-400">{vulnerabilities.length}</div>
                                <div className="text-gray-400 text-sm">Güvenlik Açığı</div>
                            </div>
                        </div>

                        {/* Quick Scan */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                            <h2 className="text-lg font-medium mb-4">🔍 Hızlı İmaj Tarama</h2>
                            <div className="flex gap-4">
                                <input
                                    type="text"
                                    value={scanImage}
                                    onChange={(e) => setScanImage(e.target.value)}
                                    placeholder="nginx:latest, python:3.9, ubuntu:22.04..."
                                    className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-cyan-500"
                                />
                                <button
                                    onClick={handleScan}
                                    disabled={scanning || !scanImage}
                                    className="px-6 py-2 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-lg font-medium hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    {scanning ? '⏳ Taranıyor...' : '🔍 Tara'}
                                </button>
                            </div>
                        </div>

                        {/* Docker Status */}
                        <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                            <h2 className="text-lg font-medium mb-4">🐳 Docker Durumu</h2>
                            <div className="flex items-center gap-4">
                                <div className={`w-4 h-4 rounded-full ${status?.docker_available ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
                                <span className={status?.docker_available ? 'text-green-400' : 'text-red-400'}>
                                    {status?.docker_available ? 'Docker Aktif' : 'Docker Bağlantısı Yok'}
                                </span>
                            </div>
                            {!status?.docker_available && (
                                <p className="text-gray-500 text-sm mt-2">
                                    Docker Desktop'ın çalıştığından emin olun veya Docker Engine'i başlatın.
                                </p>
                            )}
                        </div>
                    </div>
                )}

                {/* Containers Tab */}
                {activeTab === 'containers' && (
                    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                        <h2 className="text-lg font-medium mb-4">📦 Docker Container'ları</h2>
                        {containers.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                                            <th className="pb-3">Durum</th>
                                            <th className="pb-3">İsim</th>
                                            <th className="pb-3">İmaj</th>
                                            <th className="pb-3">Portlar</th>
                                            <th className="pb-3">Oluşturulma</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {containers.map((container, i) => (
                                            <tr key={i} className="border-t border-gray-700/50 hover:bg-gray-700/30">
                                                <td className="py-3">
                                                    <div className="flex items-center gap-2">
                                                        <div className={`w-3 h-3 rounded-full ${getStatusColor(container.status)}`}></div>
                                                        <span className="text-sm capitalize">{container.status}</span>
                                                    </div>
                                                </td>
                                                <td className="py-3 font-medium">{container.name}</td>
                                                <td className="py-3 text-cyan-400">{container.image}</td>
                                                <td className="py-3 text-gray-400 font-mono text-sm">{container.ports || '-'}</td>
                                                <td className="py-3 text-gray-400 text-sm">{container.created || '-'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <div className="text-center py-8 text-gray-500">
                                <p>Çalışan container bulunamadı</p>
                                <p className="text-sm mt-2">Docker Desktop'ı başlatın veya container oluşturun</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Images Tab */}
                {activeTab === 'images' && (
                    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                        <h2 className="text-lg font-medium mb-4">🖼️ Docker İmajları</h2>
                        {images.length > 0 ? (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {images.map((image, i) => (
                                    <div key={i} className="bg-gray-700/50 rounded-xl p-4 hover:bg-gray-700/70 transition">
                                        <div className="font-medium text-white mb-2">{image.name || image.repository}</div>
                                        <div className="text-sm text-gray-400">Tag: <span className="text-cyan-400">{image.tag || 'latest'}</span></div>
                                        <div className="text-sm text-gray-400">Boyut: <span className="text-white">{image.size || '-'}</span></div>
                                        <button
                                            onClick={() => { setScanImage(image.name || image.repository); setActiveTab('overview'); }}
                                            className="mt-3 px-3 py-1 bg-cyan-600/30 text-cyan-400 rounded text-sm hover:bg-cyan-600/50"
                                        >
                                            🔍 Tara
                                        </button>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-8 text-gray-500">
                                <p>Docker imajı bulunamadı</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Scan History Tab */}
                {activeTab === 'scans' && (
                    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                        <h2 className="text-lg font-medium mb-4">🔍 Tarama Geçmişi</h2>
                        {scanHistory.length > 0 ? (
                            <div className="space-y-4">
                                {scanHistory.map((scan, i) => (
                                    <div key={i} className="bg-gray-700/50 rounded-xl p-4">
                                        <div className="flex justify-between items-start">
                                            <div>
                                                <div className="font-medium text-white">{scan.image}</div>
                                                <div className="text-sm text-gray-400">{new Date(scan.timestamp).toLocaleString('tr-TR')}</div>
                                            </div>
                                            <div className="flex gap-2">
                                                <span className="px-2 py-1 bg-red-900/30 text-red-400 rounded text-xs">
                                                    {scan.critical || 0} Kritik
                                                </span>
                                                <span className="px-2 py-1 bg-orange-900/30 text-orange-400 rounded text-xs">
                                                    {scan.high || 0} Yüksek
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <div className="text-center py-8 text-gray-500">
                                <p>Henüz tarama yapılmadı</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Vulnerabilities Tab */}
                {activeTab === 'vulnerabilities' && (
                    <div className="bg-gradient-to-br from-gray-800 to-gray-900 rounded-2xl p-6 border border-gray-700">
                        <h2 className="text-lg font-medium mb-4">⚠️ Güvenlik Açıklıkları</h2>
                        {vulnerabilities.length > 0 ? (
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead>
                                        <tr className="text-left text-gray-400 text-sm border-b border-gray-700">
                                            <th className="pb-3">Seviye</th>
                                            <th className="pb-3">CVE</th>
                                            <th className="pb-3">Paket</th>
                                            <th className="pb-3">İmaj</th>
                                            <th className="pb-3">Çözüm</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {vulnerabilities.map((vuln, i) => (
                                            <tr key={i} className="border-t border-gray-700/50 hover:bg-gray-700/30">
                                                <td className="py-3">
                                                    <span className={`px-2 py-1 rounded text-xs font-medium ${getSeverityColor(vuln.severity)}`}>
                                                        {vuln.severity?.toUpperCase()}
                                                    </span>
                                                </td>
                                                <td className="py-3 font-mono text-cyan-400">{vuln.cve || '-'}</td>
                                                <td className="py-3">{vuln.package || '-'}</td>
                                                <td className="py-3 text-gray-400">{vuln.image || '-'}</td>
                                                <td className="py-3 text-green-400 text-sm">{vuln.fix || 'Güncelleme bekleniyor'}</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        ) : (
                            <div className="text-center py-8 text-gray-500">
                                <p>🎉 Güvenlik açığı bulunamadı!</p>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default ContainerSecurity;
