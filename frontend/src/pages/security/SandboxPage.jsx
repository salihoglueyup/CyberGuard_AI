import React, { useState, useRef, useEffect } from 'react';
import api from '../../services/api';

const SandboxPage = () => {
    const [file, setFile] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [result, setResult] = useState(null);
    const [recentAnalyses, setRecentAnalyses] = useState([]);
    const [stats, setStats] = useState(null);
    const pollTimerRef = useRef(null);

    React.useEffect(() => {
        loadRecent();
        loadStats();
        return () => {
            if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        };
    }, []);

    const loadRecent = async () => {
        try {
            const response = await api.get('/sandbox/recent');
            setRecentAnalyses(response.data.data?.recent || []);
        } catch (error) {
            console.error('Error loading recent:', error);
            setRecentAnalyses([]);
        }
    };

    const loadStats = async () => {
        try {
            const response = await api.get('/sandbox/stats');
            setStats(response.data.data);
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    };

    const handleFileUpload = async () => {
        if (!file) return;
        setAnalyzing(true);
        setResult(null);

        try {
            const formData = new FormData();
            formData.append('file', file);
            const submitRes = await api.post('/sandbox/submit', formData, {
                headers: { 'Content-Type': 'multipart/form-data' }
            });

            // Poll for results
            const submissionId = submitRes.data.data.submission_id;
            let attempts = 0;
            const pollResult = async () => {
                if (attempts >= 10) {
                    setAnalyzing(false);
                    loadRecent();
                    return;
                }
                attempts++;
                try {
                    const analysisRes = await api.get(`/sandbox/analysis/${submissionId}`);
                    if (analysisRes.data.data.status === 'completed') {
                        setResult(analysisRes.data.data.result);
                        setAnalyzing(false);
                        loadRecent();
                        return;
                    }
                } catch (err) {
                    console.error('Poll error:', err);
                }
                pollTimerRef.current = setTimeout(pollResult, 3000);
            };
            pollTimerRef.current = setTimeout(pollResult, 3000);
        } catch (error) {
            console.error('Error analyzing file:', error);
            setAnalyzing(false);
            loadRecent();
        }
    };

    const getSeverityColor = (score) => {
        if (score > 70) return 'text-red-500';
        if (score > 40) return 'text-orange-500';
        return 'text-green-500';
    };

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] p-6">
            <div className="mb-6">
                <h1 className="text-xl font-semibold text-[var(--hud-text)]">Malware Sandbox</h1>
                <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">Şüpheli dosyaları güvenli ortamda analiz edin</p>
            </div>

            {/* Stats */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Analizler (24s)</p>
                        <p className="text-2xl font-bold text-orange-400">{stats.submissions_24h}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Zararlı Tespit</p>
                        <p className="text-2xl font-bold text-red-400">{stats.malicious_found}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Ort. Süre</p>
                        <p className="text-2xl font-bold text-[var(--hud-cyan)]">{stats.avg_analysis_time_seconds}s</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Kuyruk</p>
                        <p className="text-2xl font-bold text-yellow-400">{stats.queue_length}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Top Malware</p>
                        <p className="text-lg font-bold text-purple-400">{stats.top_malware_families?.[0]}</p>
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Upload Section */}
                <div className="bg-[var(--hud-surface)] rounded-lg p-6 border border-[var(--hud-border)]">
                    <h3 className="text-lg font-semibold text-orange-400 mb-4">📤 Dosya Yükle</h3>

                    <div className="border-2 border-dashed border-[var(--hud-border)] rounded-lg p-8 text-center mb-4">
                        <input
                            type="file"
                            onChange={(e) => setFile(e.target.files[0])}
                            className="hidden"
                            id="file-upload"
                        />
                        <label htmlFor="file-upload" className="cursor-pointer">
                            <div className="text-4xl mb-2">📁</div>
                            <p className="text-[var(--hud-text-muted)]">
                                {file ? file.name : 'Dosya seçin veya sürükleyin'}
                            </p>
                            {file && (
                                <p className="text-sm text-[var(--hud-text-dim)] mt-1">
                                    {(file.size / 1024).toFixed(1)} KB
                                </p>
                            )}
                        </label>
                    </div>

                    <button
                        onClick={handleFileUpload}
                        disabled={!file || analyzing}
                        className="w-full px-4 py-3 bg-orange-600 hover:bg-orange-700 disabled:opacity-50 rounded-lg font-medium"
                    >
                        {analyzing ? '⏳ Analiz ediliyor...' : '🔬 Analiz Et'}
                    </button>

                    {/* Result */}
                    {result && (
                        <div className={`mt-4 p-4 rounded-lg ${result.is_malicious ? 'bg-red-900/30 border border-red-500/50' : 'bg-green-900/30 border border-green-500/50'}`}>
                            <div className="flex justify-between items-center mb-2">
                                <span className={`text-xl font-bold ${result.is_malicious ? 'text-red-400' : 'text-green-400'}`}>
                                    {result.is_malicious ? '🚨 ZARARLI' : '✅ TEMİZ'}
                                </span>
                                <span className={`text-2xl font-bold ${getSeverityColor(result.threat_score)}`}>
                                    {result.threat_score}/100
                                </span>
                            </div>

                            {result.is_malicious && (
                                <>
                                    <p className="text-[var(--hud-text)]">Aile: <span className="text-red-400">{result.malware_family}</span></p>
                                    <div className="mt-2">
                                        <p className="text-sm text-[var(--hud-text-muted)]">Davranışlar:</p>
                                        <div className="flex flex-wrap gap-2 mt-1">
                                            {result.behaviors?.map((b, i) => (
                                                <span key={i} className="px-2 py-1 bg-red-600/30 rounded text-xs">{b}</span>
                                            ))}
                                        </div>
                                    </div>
                                </>
                            )}
                        </div>
                    )}
                </div>

                {/* Recent Analyses */}
                <div className="bg-[var(--hud-surface)] rounded-lg p-6 border border-[var(--hud-border)]">
                    <h3 className="text-lg font-semibold text-orange-400 mb-4">📋 Son Analizler</h3>

                    <div className="space-y-3 max-h-[400px] overflow-y-auto">
                        {recentAnalyses?.length > 0 ? recentAnalyses.map((analysis, index) => (
                            <div
                                key={analysis.id || index}
                                className={`p-3 rounded-lg border-l-4 ${analysis.verdict === 'malicious'
                                    ? 'bg-red-900/20 border-red-500'
                                    : analysis.verdict === 'suspicious'
                                        ? 'bg-yellow-900/20 border-yellow-500'
                                        : 'bg-green-900/20 border-green-500'
                                    }`}
                            >
                                <div className="flex justify-between">
                                    <span className="font-medium">{analysis.filename}</span>
                                    <span className={`font-bold ${getSeverityColor(analysis.risk_score || 0)}`}>
                                        {analysis.risk_score || 0}/100
                                    </span>
                                </div>
                                <div className="flex justify-between text-xs text-[var(--hud-text-dim)] mt-1">
                                    <span className={`capitalize ${analysis.verdict === 'malicious' ? 'text-red-400' : analysis.verdict === 'suspicious' ? 'text-yellow-400' : 'text-green-400'}`}>
                                        {analysis.verdict || 'Bilinmiyor'}
                                    </span>
                                    <span>{analysis.timestamp ? new Date(analysis.timestamp).toLocaleString() : '-'}</span>
                                </div>
                            </div>
                        )) : (
                            <div className="text-center py-4 text-[var(--hud-text-dim)]">
                                <p>Henüz analiz yapılmadı</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SandboxPage;
