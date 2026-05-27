import { useState, useEffect } from 'react';
import { predictionApi } from '../../services/api';

export default function Predictions() {
    const [activeTab, setActiveTab] = useState('single');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [result, setResult] = useState(null);

    // Single prediction
    const [selectedModel, setSelectedModel] = useState('');
    const [features, setFeatures] = useState('');

    // Batch prediction
    const [batchFile, setBatchFile] = useState(null);
    const [batchResult, setBatchResult] = useState(null);

    // File analysis
    const [analysisFile, setAnalysisFile] = useState(null);
    const [analysisResult, setAnalysisResult] = useState(null);

    // History
    const [history, setHistory] = useState([]);

    // Auto-load history when tab becomes active
    useEffect(() => {
        if (activeTab === 'history' && history.length === 0) {
            loadHistory();
        }
    }, [activeTab]);

    const tabs = [
        { id: 'single', name: 'Tekil Tahmin', icon: '🎯' },
        { id: 'batch', name: 'Toplu Tahmin', icon: '📊' },
        { id: 'file', name: 'Dosya Analizi', icon: '📁' },
        { id: 'history', name: 'Geçmiş', icon: '📜' },
    ];

    const handleSinglePredict = async () => {
        if (!features.trim()) {
            setError('Lütfen özellikleri girin');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const featureArray = features.split(',').map(f => parseFloat(f.trim()));
            const res = await predictionApi.predict(selectedModel || 'auto', featureArray);
            setResult(res.data);
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    const handleBatchPredict = async () => {
        if (!batchFile) {
            setError('Lütfen CSV dosyası seçin');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const res = await predictionApi.analyzeFile(batchFile);
            setBatchResult(res.data);
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    const handleFileAnalysis = async () => {
        if (!analysisFile) {
            setError('Lütfen PCAP dosyası seçin');
            return;
        }

        setLoading(true);
        setError(null);
        try {
            const res = await predictionApi.analyzeFile(analysisFile);
            setAnalysisResult(res.data);
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    const loadHistory = async () => {
        setLoading(true);
        try {
            const res = await predictionApi.getHistory(50);
            setHistory(res.data?.predictions || []);
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    const renderSinglePrediction = () => (
        <div className="space-y-6">
            <div className="card p-6">
                <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">Tekil Tahmin</h3>

                <div className="space-y-4">
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Model Seçimi</label>
                        <select
                            className="input w-full"
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                        >
                            <option value="">Otomatik (En İyi Model)</option>
                            <option value="ssa_lstmids">SSA-LSTMIDS</option>
                            <option value="bilstm">BiLSTM</option>
                            <option value="transformer">Transformer</option>
                        </select>
                    </div>

                    <div>
                        <label className="block text-sm text-slate-400 mb-2">
                            Özellikler (virgülle ayrılmış)
                        </label>
                        <textarea
                            className="input w-full h-32"
                            placeholder="0.5, 0.3, 0.8, 0.2, ..."
                            value={features}
                            onChange={(e) => setFeatures(e.target.value)}
                        />
                    </div>

                    <button
                        className="btn-primary w-full"
                        onClick={handleSinglePredict}
                        disabled={loading}
                    >
                        {loading ? 'Tahmin Yapılıyor...' : '🎯 Tahmin Yap'}
                    </button>
                </div>
            </div>

            {result && (
                <div className="card p-6">
                    <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">Sonuç</h3>
                    <div className="grid grid-cols-2 gap-4">
                        <div className="p-4 bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-400">Tahmin</div>
                            <div className={`text-2xl font-bold ${result.prediction === 'ATTACK' ? 'text-red-400' : 'text-green-400'
                                }`}>
                                {result.prediction || result.label || 'NORMAL'}
                            </div>
                        </div>
                        <div className="p-4 bg-slate-800 rounded-lg">
                            <div className="text-sm text-slate-400">Güven Skoru</div>
                            <div className="text-2xl font-bold text-[var(--hud-cyan)]">
                                {((result.confidence || result.probability || 0) * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>

                    {result.attack_type && (
                        <div className="mt-4 p-4 bg-red-500/20 border border-red-500 rounded-lg">
                            <div className="text-sm text-red-400">Tespit Edilen Saldırı</div>
                            <div className="text-xl font-bold text-red-400">{result.attack_type}</div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );

    const renderBatchPrediction = () => (
        <div className="space-y-6">
            <div className="card p-6">
                <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">Toplu Tahmin</h3>

                <div className="space-y-4">
                    <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
                        <input
                            type="file"
                            accept=".csv"
                            onChange={(e) => setBatchFile(e.target.files[0])}
                            className="hidden"
                            id="batch-file"
                        />
                        <label htmlFor="batch-file" className="cursor-pointer">
                            <div className="text-4xl mb-2">📄</div>
                            <div className="text-slate-300">
                                {batchFile ? batchFile.name : 'CSV dosyası yükle'}
                            </div>
                            <div className="text-sm text-slate-500 mt-1">
                                veya sürükleyip bırakın
                            </div>
                        </label>
                    </div>

                    <button
                        className="btn-primary w-full"
                        onClick={handleBatchPredict}
                        disabled={loading || !batchFile}
                    >
                        {loading ? 'Analiz Ediliyor...' : '📊 Toplu Tahmin Yap'}
                    </button>
                </div>
            </div>

            {batchResult && (
                <div className="card p-6">
                    <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">Toplu Sonuçlar</h3>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                        <div className="p-4 bg-slate-800 rounded-lg text-center">
                            <div className="text-3xl font-bold text-[var(--hud-text)]">{batchResult.total || 0}</div>
                            <div className="text-sm text-slate-400">Toplam</div>
                        </div>
                        <div className="p-4 bg-slate-800 rounded-lg text-center">
                            <div className="text-3xl font-bold text-green-400">{batchResult.normal || 0}</div>
                            <div className="text-sm text-slate-400">Normal</div>
                        </div>
                        <div className="p-4 bg-slate-800 rounded-lg text-center">
                            <div className="text-3xl font-bold text-red-400">{batchResult.attacks || 0}</div>
                            <div className="text-sm text-slate-400">Saldırı</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );

    const renderFileAnalysis = () => (
        <div className="space-y-6">
            <div className="card p-6">
                <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">PCAP Dosya Analizi</h3>

                <div className="space-y-4">
                    <div className="border-2 border-dashed border-slate-600 rounded-lg p-8 text-center">
                        <input
                            type="file"
                            accept=".pcap,.pcapng"
                            onChange={(e) => setAnalysisFile(e.target.files[0])}
                            className="hidden"
                            id="analysis-file"
                        />
                        <label htmlFor="analysis-file" className="cursor-pointer">
                            <div className="text-4xl mb-2">📦</div>
                            <div className="text-slate-300">
                                {analysisFile ? analysisFile.name : 'PCAP dosyası yükle'}
                            </div>
                            <div className="text-sm text-slate-500 mt-1">
                                .pcap veya .pcapng formatı
                            </div>
                        </label>
                    </div>

                    <button
                        className="btn-primary w-full"
                        onClick={handleFileAnalysis}
                        disabled={loading || !analysisFile}
                    >
                        {loading ? 'Analiz Ediliyor...' : '🔍 Dosyayı Analiz Et'}
                    </button>
                </div>
            </div>

            {analysisResult && (
                <div className="card p-6">
                    <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">Analiz Sonuçları</h3>
                    <pre className="bg-slate-900 p-4 rounded-lg text-sm text-slate-300 overflow-auto max-h-96">
                        {JSON.stringify(analysisResult, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    );

    const renderHistory = () => (
        <div className="space-y-6">
            <div className="card p-6">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-lg font-semibold text-[var(--hud-text)]">Tahmin Geçmişi</h3>
                    <button className="btn-secondary" onClick={loadHistory}>
                        🔄 Yenile
                    </button>
                </div>

                {history.length === 0 ? (
                    <p className="text-slate-400 text-center py-8">Henüz tahmin geçmişi yok.</p>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="border-b border-slate-700">
                                    <th className="text-left py-3 px-4 text-slate-400">Tarih</th>
                                    <th className="text-left py-3 px-4 text-slate-400">Model</th>
                                    <th className="text-left py-3 px-4 text-slate-400">Sonuç</th>
                                    <th className="text-left py-3 px-4 text-slate-400">Güven</th>
                                </tr>
                            </thead>
                            <tbody>
                                {history.map((item, idx) => (
                                    <tr key={idx} className="border-b border-slate-800">
                                        <td className="py-3 px-4 text-slate-300">{item.timestamp}</td>
                                        <td className="py-3 px-4 text-slate-300">{item.model}</td>
                                        <td className="py-3 px-4">
                                            <span className={`px-2 py-1 rounded text-xs ${item.prediction === 'ATTACK'
                                                ? 'bg-red-500/20 text-red-400'
                                                : 'bg-green-500/20 text-green-400'
                                                }`}>
                                                {item.prediction}
                                            </span>
                                        </td>
                                        <td className="py-3 px-4 text-[var(--hud-cyan)]">
                                            {(item.confidence * 100).toFixed(1)}%
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );

    return (
        <div className="p-6">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-xl font-semibold text-[var(--hud-text)]">Predictions</h1>
                <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">ML modelleri ile saldırı tahmini</p>
            </div>

            {/* Error */}
            {error && (
                <div className="bg-red-500/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg mb-6">
                    {error}
                </div>
            )}

            {/* Tabs */}
            <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 rounded-lg font-medium whitespace-nowrap transition-all ${activeTab === tab.id
                            ? 'bg-cyan-500 text-[var(--hud-text)]'
                            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                    >
                        {tab.icon} {tab.name}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {activeTab === 'single' && renderSinglePrediction()}
            {activeTab === 'batch' && renderBatchPrediction()}
            {activeTab === 'file' && renderFileAnalysis()}
            {activeTab === 'history' && renderHistory()}
        </div>
    );
}
