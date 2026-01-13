import { useState, useEffect } from 'react';
import {
    Brain, PlayCircle, CheckCircle, XCircle, BarChart3, Zap,
    TrendingUp, Eye, Target, Dumbbell, GitCompare, Settings,
    Upload, Download, Trash2, Rocket, Archive, RefreshCw, Square, History
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
    LineChart, Line, Cell
} from 'recharts';
import { modelsApi, trainingApi } from '../services/api';

const TABS = [
    { id: 'overview', label: '📊 Overview', icon: BarChart3 },
    { id: 'inspect', label: '🔍 Model İnceleme', icon: Eye },
    { id: 'predict', label: '🎯 Tahmin', icon: Target },
    { id: 'training', label: '🏗️ Model Eğitimi', icon: Dumbbell },
    { id: 'compare', label: '📈 Karşılaştırma', icon: GitCompare },
    { id: 'manage', label: '⚙️ Yönetim', icon: Settings },
];

export default function MLModels() {
    const [activeTab, setActiveTab] = useState('overview');
    const [models, setModels] = useState([]);
    const [stats, setStats] = useState(null);
    const [selectedModel, setSelectedModel] = useState(null);
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(true);
    const [predicting, setPredicting] = useState(false);
    const [selectedForCompare, setSelectedForCompare] = useState([]);

    // Training states
    const [trainingSession, setTrainingSession] = useState(null);
    const [trainingSessions, setTrainingSessions] = useState([]);
    const [isTraining, setIsTraining] = useState(false);
    const [trainingConfig, setTrainingConfig] = useState({
        model_name: `CyberDefender_${new Date().toISOString().slice(0, 10).replace(/-/g, '')}`,
        description: 'Deep Learning model for threat detection',
        epochs: 150,  // Daha uzun eğitim
        batch_size: 64,  // Daha büyük batch
        hidden_layers: [512, 256, 128, 64],  // Daha derin ağ
        dropout_rate: 0.3,
        learning_rate: 0.0005,  // Daha düşük learning rate
        data_limit: 100000  // Daha fazla veri
    });

    const [features, setFeatures] = useState({
        source_ip: '192.168.1.100',
        destination_ip: '10.0.0.1',
        port: 80,
        severity: 'medium',
        attack_type: 'DDoS',
        connection_count: 1
    });

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            setLoading(true);
            const [modelsRes, statsRes] = await Promise.all([
                modelsApi.getAll(),
                modelsApi.getStats()
            ]);

            if (modelsRes.data.success) {
                setModels(modelsRes.data.data);
                if (modelsRes.data.data.length > 0) {
                    setSelectedModel(modelsRes.data.data[0]);
                }
            }
            if (statsRes.data.success) setStats(statsRes.data.data);
        } catch (error) {
            console.error('Models load error:', error);
        } finally {
            setLoading(false);
        }
    };

    const handlePredict = async () => {
        if (!selectedModel) return;
        setPredicting(true);
        setPrediction(null);

        try {
            const res = await modelsApi.predict(selectedModel.id, features);
            setPrediction(res.data);
        } catch (error) {
            setPrediction({ success: false, error: error.message });
        } finally {
            setPredicting(false);
        }
    };

    const handleDeploy = async (modelId) => {
        try {
            const res = await modelsApi.deploy(modelId);
            if (res.data.success) {
                alert('Model başarıyla deploy edildi!');
                loadData();
            } else {
                alert(`Hata: ${res.data.error}`);
            }
        } catch (error) {
            alert(`Deploy hatası: ${error.message}`);
        }
    };

    const handleArchive = async (modelId) => {
        try {
            const res = await modelsApi.archive(modelId);
            if (res.data.success) {
                alert('Model arşivlendi!');
                loadData();
            } else {
                alert(`Hata: ${res.data.error}`);
            }
        } catch (error) {
            alert(`Arşivleme hatası: ${error.message}`);
        }
    };

    const handleDelete = async (modelId) => {
        if (confirm('Bu modeli silmek istediğinizden emin misiniz? Bu işlem geri alınamaz!')) {
            try {
                const res = await modelsApi.delete(modelId);
                if (res.data.success) {
                    alert('Model başarıyla silindi!');
                    if (selectedModel?.id === modelId) {
                        setSelectedModel(null);
                    }
                    loadData();
                } else {
                    alert(`Hata: ${res.data.error}`);
                }
            } catch (error) {
                alert(`Silme hatası: ${error.message}`);
            }
        }
    };

    // Training functions
    const startTraining = async () => {
        try {
            setIsTraining(true);
            const res = await trainingApi.start(trainingConfig);
            if (res.data.success) {
                setTrainingSession({ session_id: res.data.session_id, status: 'pending', progress: 0 });
                // Start polling for status
                pollTrainingStatus(res.data.session_id);
            } else {
                alert(`Hata: ${res.data.error}`);
                setIsTraining(false);
            }
        } catch (error) {
            alert(`Training başlatılamadı: ${error.message}`);
            setIsTraining(false);
        }
    };

    const pollTrainingStatus = async (sessionId) => {
        const poll = async () => {
            try {
                const res = await trainingApi.getStatus(sessionId);
                if (res.data.success) {
                    const status = res.data.data;
                    setTrainingSession(status);

                    if (status.status === 'running' || status.status === 'pending') {
                        setTimeout(poll, 3000); // 3 saniyede bir kontrol
                    } else {
                        setIsTraining(false);
                        if (status.status === 'completed') {
                            loadData(); // Model listesini yenile
                        }
                    }
                }
            } catch (error) {
                console.error('Status check error:', error);
            }
        };
        poll();
    };

    const stopTraining = async () => {
        if (trainingSession?.session_id) {
            try {
                await trainingApi.stop(trainingSession.session_id);
                setIsTraining(false);
            } catch (error) {
                console.error('Stop error:', error);
            }
        }
    };

    const loadTrainingSessions = async () => {
        try {
            const res = await trainingApi.getSessions();
            if (res.data.success) {
                setTrainingSessions(res.data.data || []);
            }
        } catch (error) {
            console.error('Sessions load error:', error);
        }
    };

    const toggleCompare = (model) => {
        setSelectedForCompare(prev => {
            if (prev.find(m => m.id === model.id)) {
                return prev.filter(m => m.id !== model.id);
            }
            return [...prev, model];
        });
    };

    if (loading) {
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
                    <h1 className="text-3xl font-bold text-white">🧠 ML Model Dashboard</h1>
                    <p className="text-slate-400 mt-1">Training, İnceleme, Karşılaştırma, Analiz</p>
                </div>
                <button onClick={loadData} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center gap-2">
                    <RefreshCw className="w-4 h-4" /> Yenile
                </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 overflow-x-auto pb-2">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 rounded-lg flex items-center gap-2 whitespace-nowrap transition-colors ${activeTab === tab.id ? 'bg-blue-600' : 'bg-slate-800 hover:bg-slate-700'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* ========================================
          TAB 1: OVERVIEW
          ======================================== */}
            {activeTab === 'overview' && (
                <div className="space-y-6">
                    {/* Stats Cards */}
                    {stats && (
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                            <div className="card">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-400 text-sm">📦 Toplam Model</p>
                                        <p className="text-2xl font-bold text-white">{stats.total_models}</p>
                                    </div>
                                    <Brain className="w-8 h-8 text-blue-400" />
                                </div>
                            </div>
                            <div className="card">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-400 text-sm">🚀 Deployed</p>
                                        <p className="text-2xl font-bold text-emerald-400">{stats.deployed}</p>
                                    </div>
                                    <Rocket className="w-8 h-8 text-emerald-400" />
                                </div>
                            </div>
                            <div className="card">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-400 text-sm">🏆 En İyi</p>
                                        <p className="text-lg font-bold text-white truncate">{stats.best_model || '-'}</p>
                                    </div>
                                    <TrendingUp className="w-8 h-8 text-purple-400" />
                                </div>
                            </div>
                            <div className="card">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-slate-400 text-sm">📊 En Yüksek Acc</p>
                                        <p className="text-2xl font-bold text-amber-400">{(stats.best_accuracy * 100).toFixed(1)}%</p>
                                    </div>
                                    <BarChart3 className="w-8 h-8 text-amber-400" />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Performance Chart */}
                    <div className="card">
                        <h3 className="text-lg font-semibold text-white mb-4">📈 Model Performans Karşılaştırması</h3>
                        <div className="h-80">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={models.map(m => ({
                                    name: m.name?.slice(0, 15) || 'Model',
                                    Accuracy: (m.accuracy || 0) * 100,
                                    Precision: (m.precision || 0) * 100,
                                    Recall: (m.recall || 0) * 100,
                                    F1: (m.f1_score || 0) * 100
                                }))}>
                                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                    <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 10 }} />
                                    <YAxis stroke="#9ca3af" domain={[0, 100]} />
                                    <Tooltip contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151' }} />
                                    <Legend />
                                    <Bar dataKey="Accuracy" fill="#3b82f6" />
                                    <Bar dataKey="Precision" fill="#22c55e" />
                                    <Bar dataKey="Recall" fill="#f59e0b" />
                                    <Bar dataKey="F1" fill="#8b5cf6" />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    {/* Model List */}
                    <div className="card">
                        <h3 className="text-lg font-semibold text-white mb-4">📋 Model Detayları</h3>
                        <div className="space-y-4">
                            {models.map((model, idx) => (
                                <div key={model.id || idx} className="p-4 bg-slate-800 rounded-lg">
                                    <div className="flex items-center justify-between">
                                        <div className="flex items-center gap-3">
                                            <Brain className={`w-6 h-6 ${model.status === 'deployed' ? 'text-emerald-400' : 'text-slate-400'}`} />
                                            <div>
                                                <p className="text-white font-medium">🤖 {model.name}</p>
                                                <p className="text-slate-400 text-sm">{model.status || 'Unknown'}</p>
                                            </div>
                                        </div>
                                        <span className={`px-2 py-1 text-xs rounded ${model.status === 'deployed' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700'
                                            }`}>{model.status}</span>
                                    </div>
                                    <div className="grid grid-cols-3 gap-4 mt-4 text-sm">
                                        <div>
                                            <p className="text-slate-500">Genel Bilgiler</p>
                                            <p className="text-slate-300">ID: <code className="text-xs">{model.id?.slice(0, 20)}...</code></p>
                                            <p className="text-slate-300">Tip: {model.model_type || '-'}</p>
                                            <p className="text-slate-300">Framework: {model.framework || '-'}</p>
                                            <p className="text-slate-300">Tarih: {model.created_at?.slice(0, 10) || '-'}</p>
                                        </div>
                                        <div>
                                            <p className="text-slate-500">Performans</p>
                                            <p className="text-slate-300">✅ Accuracy: {((model.accuracy || 0) * 100).toFixed(2)}%</p>
                                            <p className="text-slate-300">🎯 Precision: {((model.precision || 0) * 100).toFixed(2)}%</p>
                                            <p className="text-slate-300">📊 Recall: {((model.recall || 0) * 100).toFixed(2)}%</p>
                                            <p className="text-slate-300">⚖️ F1-Score: {((model.f1_score || 0) * 100).toFixed(2)}%</p>
                                        </div>
                                        <div>
                                            <p className="text-slate-500">Training</p>
                                            <p className="text-slate-300">Train: {model.train_samples?.toLocaleString() || '-'}</p>
                                            <p className="text-slate-300">Test: {model.test_samples?.toLocaleString() || '-'}</p>
                                            <p className="text-slate-300">Epochs: {model.epochs || '-'}</p>
                                            <p className="text-slate-400 text-xs mt-1">{model.description || ''}</p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            )}

            {/* ========================================
          TAB 2: MODEL İNCELEME
          ======================================== */}
            {activeTab === 'inspect' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="card lg:col-span-1">
                        <h3 className="text-lg font-semibold text-white mb-4">🔍 Model Seç</h3>
                        <div className="space-y-2 max-h-[600px] overflow-y-auto">
                            {models.map((model) => (
                                <div
                                    key={model.id}
                                    onClick={() => setSelectedModel(model)}
                                    className={`p-3 rounded-lg cursor-pointer transition-colors ${selectedModel?.id === model.id ? 'bg-blue-600' : 'bg-slate-800 hover:bg-slate-700'
                                        }`}
                                >
                                    <p className="font-medium">{model.name}</p>
                                    <p className="text-sm text-slate-400">{((model.accuracy || 0) * 100).toFixed(1)}% acc</p>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card lg:col-span-2">
                        {selectedModel ? (
                            <div className="space-y-6">
                                <div className="flex items-center justify-between">
                                    <h3 className="text-xl font-bold text-white">{selectedModel.name}</h3>
                                    <span className={`px-3 py-1 rounded-full text-sm ${selectedModel.status === 'deployed' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700'
                                        }`}>{selectedModel.status}</span>
                                </div>

                                {/* Metrik Kartları */}
                                <div className="grid grid-cols-4 gap-4">
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <p className="text-2xl font-bold text-blue-400">{((selectedModel.accuracy || 0) * 100).toFixed(1)}%</p>
                                        <p className="text-slate-400 text-sm">Accuracy</p>
                                    </div>
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <p className="text-2xl font-bold text-emerald-400">{((selectedModel.precision || 0) * 100).toFixed(1)}%</p>
                                        <p className="text-slate-400 text-sm">Precision</p>
                                    </div>
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <p className="text-2xl font-bold text-amber-400">{((selectedModel.recall || 0) * 100).toFixed(1)}%</p>
                                        <p className="text-slate-400 text-sm">Recall</p>
                                    </div>
                                    <div className="bg-slate-800 p-3 rounded-lg text-center">
                                        <p className="text-2xl font-bold text-purple-400">{((selectedModel.f1_score || 0) * 100).toFixed(1)}%</p>
                                        <p className="text-slate-400 text-sm">F1-Score</p>
                                    </div>
                                </div>

                                {/* Grafiksel Analiz - Radar Chart */}
                                <div className="bg-slate-800 p-4 rounded-lg">
                                    <h4 className="font-semibold text-white mb-3">📊 Performans Radar Grafiği</h4>
                                    <ResponsiveContainer width="100%" height={250}>
                                        <RadarChart data={[
                                            { metric: 'Accuracy', value: (selectedModel.accuracy || 0) * 100, fullMark: 100 },
                                            { metric: 'Precision', value: (selectedModel.precision || 0) * 100, fullMark: 100 },
                                            { metric: 'Recall', value: (selectedModel.recall || 0) * 100, fullMark: 100 },
                                            { metric: 'F1-Score', value: (selectedModel.f1_score || 0) * 100, fullMark: 100 },
                                            { metric: 'TPR', value: (selectedModel.recall || 0) * 100, fullMark: 100 },
                                        ]}>
                                            <PolarGrid stroke="#334155" />
                                            <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#64748b', fontSize: 10 }} />
                                            <Radar
                                                name={selectedModel.name}
                                                dataKey="value"
                                                stroke="#3b82f6"
                                                fill="#3b82f6"
                                                fillOpacity={0.4}
                                            />
                                            <Legend />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Bar Chart - Metrik Karşılaştırma */}
                                <div className="bg-slate-800 p-4 rounded-lg">
                                    <h4 className="font-semibold text-white mb-3">📈 Metrik Karşılaştırması</h4>
                                    <ResponsiveContainer width="100%" height={200}>
                                        <BarChart data={[
                                            { name: 'Accuracy', value: (selectedModel.accuracy || 0) * 100, fill: '#3b82f6' },
                                            { name: 'Precision', value: (selectedModel.precision || 0) * 100, fill: '#10b981' },
                                            { name: 'Recall', value: (selectedModel.recall || 0) * 100, fill: '#f59e0b' },
                                            { name: 'F1-Score', value: (selectedModel.f1_score || 0) * 100, fill: '#a855f7' },
                                        ]} layout="vertical">
                                            <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                                            <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8' }} />
                                            <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8' }} width={80} />
                                            <Tooltip
                                                contentStyle={{ backgroundColor: '#1e293b', border: 'none', borderRadius: '8px' }}
                                                labelStyle={{ color: '#fff' }}
                                                formatter={(value) => [`${value.toFixed(1)}%`, 'Değer']}
                                            />
                                            <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                                                {[
                                                    { name: 'Accuracy', fill: '#3b82f6' },
                                                    { name: 'Precision', fill: '#10b981' },
                                                    { name: 'Recall', fill: '#f59e0b' },
                                                    { name: 'F1-Score', fill: '#a855f7' },
                                                ].map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.fill} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>

                                {/* Model Detayları */}
                                <div className="bg-slate-800 p-4 rounded-lg">
                                    <h4 className="font-semibold text-white mb-3">🔧 Model Detayları</h4>
                                    <div className="grid grid-cols-2 gap-4 text-sm">
                                        <div>
                                            <p className="text-slate-400">Model ID: <code className="text-white">{selectedModel.id}</code></p>
                                            <p className="text-slate-400">Framework: <span className="text-white">{selectedModel.framework}</span></p>
                                            <p className="text-slate-400">Model Type: <span className="text-white">{selectedModel.model_type}</span></p>
                                        </div>
                                        <div>
                                            <p className="text-slate-400">Train Samples: <span className="text-white">{selectedModel.train_samples?.toLocaleString()}</span></p>
                                            <p className="text-slate-400">Test Samples: <span className="text-white">{selectedModel.test_samples?.toLocaleString()}</span></p>
                                            <p className="text-slate-400">Epochs: <span className="text-white">{selectedModel.epochs}</span></p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-slate-800 p-4 rounded-lg">
                                    <h4 className="font-semibold text-white mb-2">📝 Açıklama</h4>
                                    <p className="text-slate-300">{selectedModel.description || 'Açıklama yok'}</p>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-12">
                                <Brain className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                                <p className="text-slate-400">Bir model seçin</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ========================================
          TAB 3: TAHMİN
          ======================================== */}
            {activeTab === 'predict' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="card">
                        <h3 className="text-lg font-semibold text-white mb-4">🎯 Hızlı Tahmin</h3>

                        {/* Model Seçimi */}
                        <div className="flex gap-4 mb-6">
                            <div className="flex-1">
                                <label className="block text-slate-400 text-sm mb-1">Model</label>
                                <select
                                    value={selectedModel?.id || ''}
                                    onChange={(e) => setSelectedModel(models.find(m => m.id === e.target.value))}
                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                >
                                    {models.filter(m => m.status === 'deployed').map((m) => (
                                        <option key={m.id} value={m.id}>{m.name}</option>
                                    ))}
                                </select>
                            </div>
                            {selectedModel && (
                                <div className="text-right">
                                    <p className="text-slate-400 text-sm">Accuracy</p>
                                    <p className="text-2xl font-bold text-blue-400">{((selectedModel.accuracy || 0) * 100).toFixed(1)}%</p>
                                </div>
                            )}
                        </div>

                        {/* Features */}
                        <div className="grid grid-cols-2 gap-4 mb-6">
                            <div>
                                <label className="block text-slate-400 text-sm mb-1">🌐 Kaynak IP</label>
                                <input
                                    type="text"
                                    value={features.source_ip}
                                    onChange={(e) => setFeatures({ ...features, source_ip: e.target.value })}
                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                />
                            </div>
                            <div>
                                <label className="block text-slate-400 text-sm mb-1">🎯 Hedef IP</label>
                                <input
                                    type="text"
                                    value={features.destination_ip}
                                    onChange={(e) => setFeatures({ ...features, destination_ip: e.target.value })}
                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                />
                            </div>
                            <div>
                                <label className="block text-slate-400 text-sm mb-1">🔌 Port</label>
                                <input
                                    type="number"
                                    value={features.port}
                                    onChange={(e) => setFeatures({ ...features, port: parseInt(e.target.value) })}
                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                />
                            </div>
                            <div>
                                <label className="block text-slate-400 text-sm mb-1">⚠️ Severity</label>
                                <select
                                    value={features.severity}
                                    onChange={(e) => setFeatures({ ...features, severity: e.target.value })}
                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                >
                                    <option value="low">Low</option>
                                    <option value="medium">Medium</option>
                                    <option value="high">High</option>
                                    <option value="critical">Critical</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-slate-400 text-sm mb-1">🎭 Saldırı Türü</label>
                                <select
                                    value={features.attack_type}
                                    onChange={(e) => setFeatures({ ...features, attack_type: e.target.value })}
                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                >
                                    <option value="DDoS">DDoS</option>
                                    <option value="SQL Injection">SQL Injection</option>
                                    <option value="XSS">XSS</option>
                                    <option value="Brute Force">Brute Force</option>
                                    <option value="Port Scan">Port Scan</option>
                                    <option value="Malware">Malware</option>
                                </select>
                            </div>
                            <div>
                                <label className="block text-slate-400 text-sm mb-1">🔗 Bağlantı Sayısı</label>
                                <input
                                    type="number"
                                    value={features.connection_count}
                                    onChange={(e) => setFeatures({ ...features, connection_count: parseInt(e.target.value) })}
                                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                />
                            </div>
                        </div>

                        <button
                            onClick={handlePredict}
                            disabled={predicting || !selectedModel}
                            className="w-full py-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 
                rounded-lg transition-colors flex items-center justify-center gap-2"
                        >
                            <PlayCircle className={`w-5 h-5 ${predicting ? 'animate-spin' : ''}`} />
                            🚀 TAHMİN YAP
                        </button>
                    </div>

                    {/* Sonuç */}
                    <div className="card">
                        <h3 className="text-lg font-semibold text-white mb-4">Sonuç</h3>

                        {prediction ? (
                            <div className={`p-6 rounded-xl ${prediction.success
                                ? prediction.prediction === 'malicious'
                                    ? 'bg-red-500/10 border border-red-500/30'
                                    : 'bg-emerald-500/10 border border-emerald-500/30'
                                : 'bg-yellow-500/10 border border-yellow-500/30'
                                }`}>
                                {prediction.success ? (
                                    <div className="space-y-4">
                                        <div className="flex items-center gap-3">
                                            {prediction.prediction === 'malicious'
                                                ? <XCircle className="w-12 h-12 text-red-400" />
                                                : <CheckCircle className="w-12 h-12 text-emerald-400" />
                                            }
                                            <div>
                                                <span className="text-3xl font-bold text-white">
                                                    {prediction.prediction === 'malicious' ? '⚠️ ZARARLI' : '✅ GÜVENLİ'}
                                                </span>
                                                <p className="text-slate-400">{prediction.model_name}</p>
                                            </div>
                                        </div>

                                        <div className="grid grid-cols-2 gap-4 mt-4">
                                            <div className="bg-slate-800/50 p-4 rounded-lg text-center">
                                                <p className="text-slate-400 text-sm">Güven Skoru</p>
                                                <p className="text-3xl font-bold text-white">{prediction.confidence_percentage}%</p>
                                                <p className="text-sm text-slate-500">+{(prediction.confidence_percentage - 50).toFixed(1)}%</p>
                                            </div>
                                            <div className="bg-slate-800/50 p-4 rounded-lg text-center">
                                                <p className="text-slate-400 text-sm">Risk Seviyesi</p>
                                                <p className={`text-3xl font-bold ${prediction.risk_level === 'CRITICAL' ? 'text-red-400' :
                                                    prediction.risk_level === 'HIGH' ? 'text-orange-400' :
                                                        prediction.risk_level === 'MEDIUM' ? 'text-yellow-400' :
                                                            'text-emerald-400'
                                                    }`}>
                                                    {prediction.risk_level === 'CRITICAL' && '🔴'}
                                                    {prediction.risk_level === 'HIGH' && '🟠'}
                                                    {prediction.risk_level === 'MEDIUM' && '🟡'}
                                                    {prediction.risk_level === 'LOW' && '🟢'}
                                                    {prediction.risk_level === 'SAFE' && '✅'}
                                                    {' '}{prediction.risk_level}
                                                </p>
                                            </div>
                                        </div>

                                        <div className="bg-blue-500/10 border border-blue-500/30 p-4 rounded-lg mt-4">
                                            <p className="text-blue-400">📝 <strong>Açıklama:</strong></p>
                                            <p className="text-slate-300 mt-1">{prediction.explanation}</p>
                                        </div>

                                        <details className="mt-4">
                                            <summary className="text-slate-400 cursor-pointer hover:text-white">🔍 Detaylı Bilgi</summary>
                                            <pre className="mt-2 p-3 bg-slate-800 rounded-lg text-xs overflow-auto max-h-48">
                                                {JSON.stringify(prediction, null, 2)}
                                            </pre>
                                        </details>
                                    </div>
                                ) : (
                                    <p className="text-yellow-400">❌ {prediction.error}</p>
                                )}
                            </div>
                        ) : (
                            <div className="text-center py-12">
                                <Target className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                                <p className="text-slate-400">Tahmin sonucu burada görünecek</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* ========================================
          TAB 4: MODEL EĞİTİMİ
          ======================================== */}
            {activeTab === 'training' && (
                <div className="space-y-6">
                    {/* Aktif Training */}
                    {isTraining && trainingSession && (
                        <div className="card border-2 border-blue-500/50">
                            <div className="flex items-center justify-between mb-4">
                                <h3 className="text-lg font-semibold text-white">
                                    ⏳ Training Devam Ediyor: {trainingSession.config?.model_name}
                                </h3>
                                <span className="px-3 py-1 bg-blue-500/20 text-blue-400 rounded-full text-sm">
                                    {trainingSession.status}
                                </span>
                            </div>

                            <div className="mb-4">
                                <div className="flex justify-between text-sm text-slate-400 mb-1">
                                    <span>Epoch {trainingSession.current_epoch || 0}/{trainingSession.total_epochs || trainingConfig.epochs}</span>
                                    <span>{trainingSession.progress?.toFixed(1) || 0}%</span>
                                </div>
                                <div className="w-full bg-slate-700 rounded-full h-4">
                                    <div
                                        className="bg-blue-600 h-4 rounded-full transition-all duration-500"
                                        style={{ width: `${trainingSession.progress || 0}%` }}
                                    ></div>
                                </div>
                            </div>

                            {trainingSession.logs && (
                                <div className="bg-slate-800 rounded-lg p-3 max-h-40 overflow-y-auto mb-4">
                                    <p className="text-slate-400 text-sm mb-2">📝 Training Logs</p>
                                    {trainingSession.logs.slice(-5).map((log, i) => (
                                        <p key={i} className="text-xs text-slate-300 font-mono">
                                            {log.level === 'success' && '✅'}
                                            {log.level === 'error' && '❌'}
                                            {log.level === 'warning' && '⚠️'}
                                            {log.level === 'info' && 'ℹ️'}
                                            {' '}{log.timestamp?.slice(11, 19)} - {log.message}
                                        </p>
                                    ))}
                                </div>
                            )}

                            <div className="flex gap-4">
                                <button
                                    onClick={() => trainingSession.session_id && pollTrainingStatus(trainingSession.session_id)}
                                    className="flex-1 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg flex items-center justify-center gap-2"
                                >
                                    <RefreshCw className="w-4 h-4" /> Yenile
                                </button>
                                <button
                                    onClick={stopTraining}
                                    className="flex-1 py-2 bg-red-600 hover:bg-red-700 rounded-lg flex items-center justify-center gap-2"
                                >
                                    <Square className="w-4 h-4" /> Durdur
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Training Tamamlandı */}
                    {trainingSession?.status === 'completed' && !isTraining && (
                        <div className="card border-2 border-emerald-500/50">
                            <div className="flex items-center gap-3 mb-4">
                                <CheckCircle className="w-10 h-10 text-emerald-400" />
                                <div>
                                    <h3 className="text-xl font-bold text-white">✅ Training Tamamlandı!</h3>
                                    <p className="text-slate-400">{trainingSession.config?.model_name}</p>
                                </div>
                            </div>

                            <div className="grid grid-cols-3 gap-4 mb-4">
                                <div className="bg-slate-800 p-3 rounded-lg text-center">
                                    <p className="text-slate-400 text-sm">Model ID</p>
                                    <p className="text-white font-mono text-sm truncate">{trainingSession.model_id}</p>
                                </div>
                                <div className="bg-slate-800 p-3 rounded-lg text-center">
                                    <p className="text-slate-400 text-sm">Süre</p>
                                    <p className="text-white font-bold">{trainingSession.duration}</p>
                                </div>
                                <div className="bg-slate-800 p-3 rounded-lg text-center">
                                    <p className="text-slate-400 text-sm">Accuracy</p>
                                    <p className="text-emerald-400 font-bold">
                                        {(trainingSession.result?.summary?.accuracy * 100 || 0).toFixed(2)}%
                                    </p>
                                </div>
                            </div>

                            <button
                                onClick={() => setTrainingSession(null)}
                                className="w-full py-2 bg-blue-600 hover:bg-blue-700 rounded-lg"
                            >
                                🎉 Yeni Model Eğit
                            </button>
                        </div>
                    )}

                    {/* Training Form */}
                    {!isTraining && trainingSession?.status !== 'completed' && (
                        <div className="card">
                            <h3 className="text-lg font-semibold text-white mb-4">🏗️ Yeni Model Eğitimi</h3>

                            <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-6">
                                <p className="text-blue-400">💡 Parametreleri ayarlayın ve 'Eğitime Başla' butonuna tıklayın.</p>
                            </div>

                            <div className="grid grid-cols-2 gap-6 mb-6">
                                <div>
                                    <h4 className="font-semibold text-white mb-3">📝 Model Bilgileri</h4>
                                    <div className="space-y-4">
                                        <div>
                                            <label className="block text-slate-400 text-sm mb-1">Model Adı</label>
                                            <input
                                                type="text"
                                                value={trainingConfig.model_name}
                                                onChange={(e) => setTrainingConfig({ ...trainingConfig, model_name: e.target.value })}
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-slate-400 text-sm mb-1">Açıklama</label>
                                            <textarea
                                                value={trainingConfig.description}
                                                onChange={(e) => setTrainingConfig({ ...trainingConfig, description: e.target.value })}
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white h-20"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-slate-400 text-sm mb-1">📊 Veri Sayısı</label>
                                            <input
                                                type="number"
                                                value={trainingConfig.data_limit}
                                                onChange={(e) => setTrainingConfig({ ...trainingConfig, data_limit: parseInt(e.target.value) })}
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                            />
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <h4 className="font-semibold text-white mb-3">⚙️ Hyperparameters</h4>
                                    <div className="grid grid-cols-2 gap-4">
                                        <div>
                                            <label className="block text-slate-400 text-sm mb-1">🔄 Epochs</label>
                                            <input
                                                type="number"
                                                value={trainingConfig.epochs}
                                                onChange={(e) => setTrainingConfig({ ...trainingConfig, epochs: parseInt(e.target.value) })}
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                            />
                                        </div>
                                        <div>
                                            <label className="block text-slate-400 text-sm mb-1">📦 Batch Size</label>
                                            <select
                                                value={trainingConfig.batch_size}
                                                onChange={(e) => setTrainingConfig({ ...trainingConfig, batch_size: parseInt(e.target.value) })}
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                            >
                                                <option value={16}>16</option>
                                                <option value={32}>32</option>
                                                <option value={64}>64</option>
                                                <option value={128}>128</option>
                                                <option value={256}>256</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-slate-400 text-sm mb-1">📈 Learning Rate</label>
                                            <select
                                                value={trainingConfig.learning_rate}
                                                onChange={(e) => setTrainingConfig({ ...trainingConfig, learning_rate: parseFloat(e.target.value) })}
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                            >
                                                <option value={0.0001}>0.0001</option>
                                                <option value={0.0005}>0.0005</option>
                                                <option value={0.001}>0.001</option>
                                                <option value={0.005}>0.005</option>
                                                <option value={0.01}>0.01</option>
                                            </select>
                                        </div>
                                        <div>
                                            <label className="block text-slate-400 text-sm mb-1">🎲 Dropout</label>
                                            <input
                                                type="number"
                                                value={trainingConfig.dropout_rate}
                                                step={0.1}
                                                min={0}
                                                max={0.7}
                                                onChange={(e) => setTrainingConfig({ ...trainingConfig, dropout_rate: parseFloat(e.target.value) })}
                                                className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded-lg text-white"
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <div className="flex gap-4">
                                <button
                                    onClick={startTraining}
                                    className="flex-1 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold flex items-center justify-center gap-2"
                                >
                                    <Rocket className="w-5 h-5" /> 🚀 EĞİTİME BAŞLA
                                </button>
                                <button
                                    onClick={loadTrainingSessions}
                                    className="px-4 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg flex items-center gap-2"
                                >
                                    <History className="w-5 h-5" /> Geçmiş
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Training Geçmişi */}
                    {trainingSessions.length > 0 && (
                        <div className="card">
                            <h3 className="text-lg font-semibold text-white mb-4">📜 Training Geçmişi</h3>
                            <div className="space-y-2">
                                {trainingSessions.slice(0, 5).map((session, i) => (
                                    <div key={i} className="flex items-center justify-between p-3 bg-slate-800 rounded-lg">
                                        <div className="flex items-center gap-3">
                                            {session.status === 'completed' && <CheckCircle className="w-5 h-5 text-emerald-400" />}
                                            {session.status === 'failed' && <XCircle className="w-5 h-5 text-red-400" />}
                                            {session.status === 'running' && <PlayCircle className="w-5 h-5 text-blue-400 animate-spin" />}
                                            {session.status === 'pending' && <RefreshCw className="w-5 h-5 text-slate-400" />}
                                            <div>
                                                <p className="text-white font-medium">{session.config?.model_name}</p>
                                                <p className="text-slate-400 text-sm">{session.start_time?.slice(0, 19)}</p>
                                            </div>
                                        </div>
                                        <span className={`px-2 py-1 rounded text-xs ${session.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' :
                                            session.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                                                'bg-slate-700'
                                            }`}>{session.status}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}

            {/* ========================================
          TAB 5: KARŞILAŞTIRMA
          ======================================== */}
            {activeTab === 'compare' && (
                <div className="space-y-6">
                    <div className="card">
                        <h3 className="text-lg font-semibold text-white mb-4">📊 Karşılaştırılacak Modelleri Seçin</h3>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {models.map((model) => (
                                <label
                                    key={model.id}
                                    className={`p-3 rounded-lg cursor-pointer border transition-colors ${selectedForCompare.find(m => m.id === model.id)
                                        ? 'border-blue-500 bg-blue-500/10'
                                        : 'border-slate-700 hover:border-slate-600'
                                        }`}
                                >
                                    <input
                                        type="checkbox"
                                        checked={!!selectedForCompare.find(m => m.id === model.id)}
                                        onChange={() => toggleCompare(model)}
                                        className="hidden"
                                    />
                                    <p className="font-medium text-white">{model.name}</p>
                                    <p className="text-sm text-slate-400">{((model.accuracy || 0) * 100).toFixed(1)}% acc</p>
                                </label>
                            ))}
                        </div>
                    </div>

                    {selectedForCompare.length >= 2 && (
                        <>
                            {/* Metrik Tablosu */}
                            <div className="card">
                                <h3 className="text-lg font-semibold text-white mb-4">📊 Metrik Karşılaştırması</h3>
                                <div className="overflow-x-auto">
                                    <table className="w-full text-sm">
                                        <thead>
                                            <tr className="text-left text-slate-400 border-b border-slate-700">
                                                <th className="pb-2">Model</th>
                                                <th className="pb-2">Accuracy</th>
                                                <th className="pb-2">Precision</th>
                                                <th className="pb-2">Recall</th>
                                                <th className="pb-2">F1-Score</th>
                                                <th className="pb-2">Status</th>
                                            </tr>
                                        </thead>
                                        <tbody className="divide-y divide-slate-800">
                                            {selectedForCompare.map((m) => (
                                                <tr key={m.id}>
                                                    <td className="py-2 text-white font-medium">{m.name}</td>
                                                    <td className="py-2 text-blue-400">{((m.accuracy || 0) * 100).toFixed(2)}%</td>
                                                    <td className="py-2 text-emerald-400">{((m.precision || 0) * 100).toFixed(2)}%</td>
                                                    <td className="py-2 text-amber-400">{((m.recall || 0) * 100).toFixed(2)}%</td>
                                                    <td className="py-2 text-purple-400">{((m.f1_score || 0) * 100).toFixed(2)}%</td>
                                                    <td className="py-2"><span className={`px-2 py-1 rounded text-xs ${m.status === 'deployed' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-slate-700'
                                                        }`}>{m.status}</span></td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            {/* Radar Chart */}
                            <div className="card">
                                <h3 className="text-lg font-semibold text-white mb-4">🎯 Radar Chart</h3>
                                <div className="h-96">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <RadarChart data={[
                                            { metric: 'Accuracy', ...Object.fromEntries(selectedForCompare.map(m => [m.name, (m.accuracy || 0) * 100])) },
                                            { metric: 'Precision', ...Object.fromEntries(selectedForCompare.map(m => [m.name, (m.precision || 0) * 100])) },
                                            { metric: 'Recall', ...Object.fromEntries(selectedForCompare.map(m => [m.name, (m.recall || 0) * 100])) },
                                            { metric: 'F1-Score', ...Object.fromEntries(selectedForCompare.map(m => [m.name, (m.f1_score || 0) * 100])) },
                                        ]}>
                                            <PolarGrid stroke="#374151" />
                                            <PolarAngleAxis dataKey="metric" stroke="#9ca3af" />
                                            <PolarRadiusAxis domain={[0, 100]} stroke="#374151" />
                                            {selectedForCompare.map((m, i) => (
                                                <Radar
                                                    key={m.id}
                                                    name={m.name}
                                                    dataKey={m.name}
                                                    stroke={['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6'][i % 4]}
                                                    fill={['#3b82f6', '#22c55e', '#f59e0b', '#8b5cf6'][i % 4]}
                                                    fillOpacity={0.2}
                                                />
                                            ))}
                                            <Legend />
                                        </RadarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            )}

            {/* ========================================
          TAB 6: YÖNETİM
          ======================================== */}
            {activeTab === 'manage' && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="card">
                        <h3 className="text-lg font-semibold text-white mb-4">⚙️ Model Seç</h3>
                        <div className="space-y-2">
                            {models.map((model) => (
                                <div
                                    key={model.id}
                                    onClick={() => setSelectedModel(model)}
                                    className={`p-3 rounded-lg cursor-pointer transition-colors ${selectedModel?.id === model.id ? 'bg-blue-600' : 'bg-slate-800 hover:bg-slate-700'
                                        }`}
                                >
                                    <p className="font-medium">{model.name}</p>
                                    <p className="text-sm text-slate-400">{model.status}</p>
                                </div>
                            ))}
                        </div>
                    </div>

                    <div className="card lg:col-span-2">
                        {selectedModel ? (
                            <div className="space-y-6">
                                <h3 className="text-xl font-bold text-white">{selectedModel.name}</h3>

                                <div className="grid grid-cols-4 gap-4">
                                    <button
                                        onClick={() => handleDeploy(selectedModel.id)}
                                        className="p-4 bg-emerald-600 hover:bg-emerald-700 rounded-lg flex flex-col items-center gap-2"
                                    >
                                        <Rocket className="w-6 h-6" />
                                        <span>🚀 Deploy</span>
                                    </button>
                                    <button
                                        onClick={() => handleArchive(selectedModel.id)}
                                        className="p-4 bg-amber-600 hover:bg-amber-700 rounded-lg flex flex-col items-center gap-2"
                                    >
                                        <Archive className="w-6 h-6" />
                                        <span>📦 Archive</span>
                                    </button>
                                    <button
                                        onClick={() => {
                                            const json = JSON.stringify(selectedModel, null, 2);
                                            const blob = new Blob([json], { type: 'application/json' });
                                            const url = URL.createObjectURL(blob);
                                            const a = document.createElement('a');
                                            a.href = url;
                                            a.download = `${selectedModel.name}.json`;
                                            a.click();
                                        }}
                                        className="p-4 bg-blue-600 hover:bg-blue-700 rounded-lg flex flex-col items-center gap-2"
                                    >
                                        <Download className="w-6 h-6" />
                                        <span>📥 Export</span>
                                    </button>
                                    <button
                                        onClick={() => handleDelete(selectedModel.id)}
                                        className="p-4 bg-red-600 hover:bg-red-700 rounded-lg flex flex-col items-center gap-2"
                                    >
                                        <Trash2 className="w-6 h-6" />
                                        <span>🗑️ Delete</span>
                                    </button>
                                </div>

                                <div className="bg-slate-800 p-4 rounded-lg">
                                    <h4 className="font-semibold text-white mb-3">📋 Model Detayları</h4>
                                    <pre className="text-xs text-slate-300 overflow-auto max-h-80">
                                        {JSON.stringify(selectedModel, null, 2)}
                                    </pre>
                                </div>
                            </div>
                        ) : (
                            <div className="text-center py-12">
                                <Settings className="w-16 h-16 text-slate-600 mx-auto mb-4" />
                                <p className="text-slate-400">Yönetmek için bir model seçin</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
