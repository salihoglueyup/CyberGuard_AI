import React, { useState, useEffect } from 'react';
import api from '../services/api';

const VulnScanner = () => {
    const [target, setTarget] = useState('192.168.1.1');
    const [scanType, setScanType] = useState('full');
    const [scanResult, setScanResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [cveSearch, setCveSearch] = useState('');
    const [cveResult, setCveResult] = useState(null);

    const runScan = async () => {
        setLoading(true);
        try {
            const response = await api.post('/vuln/scan', {
                target: target,
                scan_type: scanType,
                include_cve: true,
                include_ports: true
            });
            if (response.data.success) {
                setScanResult(response.data.data);
            }
        } catch (error) {
            console.error('Error running scan:', error);
        } finally {
            setLoading(false);
        }
    };

    const searchCVE = async () => {
        if (!cveSearch) return;
        try {
            const response = await api.get(`/vuln/cve/${cveSearch}`);
            if (response.data.success) {
                setCveResult(response.data.data);
            }
        } catch (error) {
            console.error('Error searching CVE:', error);
        }
    };

    const getSeverityColor = (severity) => {
        const colors = {
            critical: 'bg-red-500 text-white',
            high: 'bg-orange-500 text-white',
            medium: 'bg-yellow-500 text-black',
            low: 'bg-green-500 text-white'
        };
        return colors[severity] || 'bg-gray-500';
    };

    const getRiskBadge = (risk) => {
        const colors = { high: 'bg-red-900/50 text-red-400', medium: 'bg-yellow-900/50 text-yellow-400', low: 'bg-green-900/50 text-green-400' };
        return colors[risk] || 'bg-gray-700';
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-green-400 to-cyan-500 bg-clip-text text-transparent">
                        🔍 Vulnerability Scanner
                    </h1>
                    <p className="text-gray-400 mt-2">
                        Port tarama, CVE kontrolü ve güvenlik açığı tespiti
                    </p>
                </div>

                {/* Scanner Controls */}
                <div className="bg-gray-800 rounded-xl p-6 mb-6">
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Hedef IP/Hostname</label>
                            <input
                                type="text"
                                value={target}
                                onChange={(e) => setTarget(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
                                placeholder="192.168.1.1 veya example.com"
                            />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Tarama Tipi</label>
                            <select
                                value={scanType}
                                onChange={(e) => setScanType(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
                            >
                                <option value="quick">Quick Scan</option>
                                <option value="full">Full Scan</option>
                                <option value="deep">Deep Scan</option>
                            </select>
                        </div>
                        <div className="flex items-end">
                            <button
                                onClick={runScan}
                                disabled={loading}
                                className="w-full bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
                            >
                                {loading ? '⏳ Taranıyor...' : '🔍 Tarama Başlat'}
                            </button>
                        </div>
                    </div>
                </div>

                {/* CVE Search */}
                <div className="bg-gray-800 rounded-xl p-6 mb-6">
                    <h2 className="text-lg font-medium mb-4">📋 CVE Arama</h2>
                    <div className="flex gap-4">
                        <input
                            type="text"
                            value={cveSearch}
                            onChange={(e) => setCveSearch(e.target.value)}
                            className="flex-1 bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
                            placeholder="CVE-2024-0001"
                        />
                        <button
                            onClick={searchCVE}
                            className="bg-cyan-600 hover:bg-cyan-700 px-6 py-2 rounded-lg font-medium transition-colors"
                        >
                            Ara
                        </button>
                    </div>

                    {cveResult && (
                        <div className="mt-4 bg-gray-700/50 rounded-lg p-4">
                            <div className="flex items-center gap-3 mb-2">
                                <span className={`px-2 py-1 rounded text-xs font-bold ${getSeverityColor(cveResult.severity)}`}>
                                    {cveResult.severity?.toUpperCase()}
                                </span>
                                <span className="font-bold">{cveResult.id}</span>
                                <span className="text-gray-400">CVSS: {cveResult.cvss}</span>
                            </div>
                            <p className="text-gray-300">{cveResult.description}</p>
                            <div className="mt-2 text-sm text-green-400">
                                <strong>Çözüm:</strong> {cveResult.solution}
                            </div>
                        </div>
                    )}
                </div>

                {/* Scan Results */}
                {scanResult && (
                    <div className="space-y-6">
                        {/* Summary */}
                        <div className="bg-gray-800 rounded-xl p-6">
                            <h2 className="text-lg font-medium mb-4">📊 Tarama Özeti</h2>
                            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-red-400">{scanResult.summary?.critical || 0}</div>
                                    <div className="text-gray-400 text-sm">Critical</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-orange-400">{scanResult.summary?.high || 0}</div>
                                    <div className="text-gray-400 text-sm">High</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-yellow-400">{scanResult.summary?.medium || 0}</div>
                                    <div className="text-gray-400 text-sm">Medium</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-green-400">{scanResult.summary?.low || 0}</div>
                                    <div className="text-gray-400 text-sm">Low</div>
                                </div>
                                <div className="text-center">
                                    <div className="text-3xl font-bold text-cyan-400">{scanResult.open_ports?.length || 0}</div>
                                    <div className="text-gray-400 text-sm">Open Ports</div>
                                </div>
                            </div>
                        </div>

                        {/* Open Ports */}
                        {scanResult.open_ports && scanResult.open_ports.length > 0 && (
                            <div className="bg-gray-800 rounded-xl p-6">
                                <h2 className="text-lg font-medium mb-4">🔌 Açık Portlar</h2>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                    {scanResult.open_ports.map((port, i) => (
                                        <div key={i} className="bg-gray-700/50 rounded-lg p-4">
                                            <div className="flex items-center justify-between mb-2">
                                                <span className="font-mono text-xl font-bold text-cyan-400">:{port.port}</span>
                                                <span className={`px-2 py-1 rounded text-xs ${getRiskBadge(port.risk)}`}>
                                                    {port.risk}
                                                </span>
                                            </div>
                                            <div className="text-gray-400 text-sm">{port.service}</div>
                                            <div className="text-gray-500 text-xs">{port.version}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Vulnerabilities */}
                        {scanResult.vulnerabilities && scanResult.vulnerabilities.length > 0 && (
                            <div className="bg-gray-800 rounded-xl p-6">
                                <h2 className="text-lg font-medium mb-4">⚠️ Güvenlik Açıkları</h2>
                                <div className="space-y-4">
                                    {scanResult.vulnerabilities.map((vuln, i) => (
                                        <div key={i} className="bg-gray-700/50 rounded-lg p-4">
                                            <div className="flex items-start justify-between">
                                                <div>
                                                    <div className="flex items-center gap-2 mb-1">
                                                        <span className={`px-2 py-1 rounded text-xs font-bold ${getSeverityColor(vuln.severity)}`}>
                                                            {vuln.severity?.toUpperCase()}
                                                        </span>
                                                        <span className="font-bold">{vuln.title}</span>
                                                    </div>
                                                    <p className="text-gray-400 text-sm">{vuln.description}</p>
                                                    {vuln.cve_id && (
                                                        <span className="inline-block mt-2 bg-gray-600 px-2 py-1 rounded text-xs">
                                                            {vuln.cve_id}
                                                        </span>
                                                    )}
                                                </div>
                                                <div className="text-right">
                                                    <div className="text-2xl font-bold text-red-400">{vuln.cvss}</div>
                                                    <div className="text-gray-500 text-xs">CVSS</div>
                                                </div>
                                            </div>
                                            <div className="mt-3 pt-3 border-t border-gray-600">
                                                <span className="text-green-400 text-sm">
                                                    <strong>Öneri:</strong> {vuln.recommendation}
                                                </span>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
};

export default VulnScanner;
