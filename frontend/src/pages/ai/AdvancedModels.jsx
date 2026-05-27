import { useState, useEffect } from 'react';
import {
    Brain, Zap, GitCompare, Target, Settings, Rocket,
    TrendingUp, Activity, Layers, Cpu, BarChart3, RefreshCw,
    Play, Pause, CheckCircle, XCircle, Clock, Award
} from 'lucide-react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend,
    LineChart, Line, Cell
} from 'recharts';
import { advancedModelsApi } from '../../services/api';
import { useToast } from '../../components/ui/Toast';

const TABS = [
    { id: 'models', label: '🧠 Modeller', icon: Brain },
    { id: 'compare', label: '📊 Karşılaştırma', icon: GitCompare },
    { id: 'train', label: '🏋️ Eğitim', icon: Rocket },
    { id: 'optimize', label: '🔧 Optimizasyon', icon: Settings },
    { id: 'ensemble', label: '🎯 Ensemble', icon: Layers },
];

const MODEL_COLORS = {
    lstm: '#3B82F6',
    bilstm: '#8B5CF6',
    transformer: '#10B981',
    gru: '#F59E0B',
};

export default function AdvancedModels() {
    const [activeTab, setActiveTab] = useState('models');
    const [models, setModels] = useState([]);
    const [comparison, setComparison] = useState([]);
    const [trainings, setTrainings] = useState([]);
    const [optimizations, setOptimizations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const toast = useToast();

    // Training config
    const [trainConfig, setTrainConfig] = useState({
        model_type: 'bilstm',
        epochs: 50,
        batch_size: 64,
        learning_rate: 0.001,
        use_smote: false,
        use_attention: true,
        lstm_units: 120,
        dropout_rate: 0.3
    });

    // Optimization config
    const [optConfig, setOptConfig] = useState({
        model_type: 'bilstm',
        algorithm: 'ssa',
        max_iterations: 20,
        population_size: 10
    });

    // Ensemble config
    const [ensembleConfig, setEnsembleConfig] = useState({
        model_ids: ['bilstm', 'transformer'],
        voting: 'soft',
        weights: null
    });

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            setLoading(true);
            const [modelsRes, compareRes, trainingsRes, optRes] = await Promise.all([
                advancedModelsApi.getModels(),
                advancedModelsApi.compare(),
                advancedModelsApi.getTrainings(),
                advancedModelsApi.getOptimizations()
            ]);

            if (modelsRes.data.success) setModels(modelsRes.data.data);
            if (compareRes.data.success) setComparison(compareRes.data.data);
            if (trainingsRes.data.success) setTrainings(trainingsRes.data.data);
            if (optRes.data.success) setOptimizations(optRes.data.data);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleStartTraining = async () => {
        try {
            const res = await advancedModelsApi.train(trainConfig);
            if (res.data.success) {
                loadData();
            }
        } catch (err) {
            setError(err.message);
        }
    };

    const handleStartOptimization = async () => {
        try {
            const res = await advancedModelsApi.optimize(optConfig);
            if (res.data.success) {
                loadData();
            }
        } catch (err) {
            setError(err.message);
        }
    };

    const handleCreateEnsemble = async () => {
        try {
            const res = await advancedModelsApi.createEnsemble(ensembleConfig);
            if (res.data.success) {
                toast.success('Ensemble oluşturuldu!');
            }
        } catch (err) {
            setError(err.message);
        }
    };

    // Model kartı
    const ModelCard = ({ model }) => (
        <div className={`bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-5 hover:border-purple-500/50 transition-all ${model.recommended ? 'ring-2 ring-purple-500/30' : ''}`}>
            <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                    <div className={`w-12 h-12 rounded-lg flex items-center justify-center`} style={{ backgroundColor: MODEL_COLORS[model.id] + '20' }}>
                        <Brain className="w-6 h-6" style={{ color: MODEL_COLORS[model.id] }} />
                    </div>
                    <div>
                        <h3 className="text-lg font-semibold text-[var(--hud-text)] flex items-center gap-2">
                            {model.name}
                            {model.recommended && <Award className="w-4 h-4 text-yellow-400" />}
                        </h3>
                        <p className="text-sm text-[var(--hud-text-muted)]">{model.description}</p>
                    </div>
                </div>
                {model.edge_ready && (
                    <span className="px-2 py-1 bg-green-500/20 text-green-400 text-xs rounded-full">⚡ Edge Ready</span>
                )}
            </div>
            <div className="grid grid-cols-2 gap-3 mt-4">
                <div className="bg-[var(--hud-bg)]/50 rounded-lg p-3">
                    <p className="text-xs text-[var(--hud-text-dim)]">Parametreler</p>
                    <p className="text-lg font-bold text-[var(--hud-text)]">{model.params}</p>
                </div>
                <div className="bg-[var(--hud-bg)]/50 rounded-lg p-3">
                    <p className="text-xs text-[var(--hud-text-dim)]">Hız</p>
                    <p className="text-lg font-bold text-[var(--hud-text)] capitalize">{model.speed}</p>
                </div>
            </div>
        </div>
    );

    // Karşılaştırma tablosu
    const ComparisonTable = () => (
        <div className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl overflow-hidden">
            <table className="w-full">
                <thead className="bg-[var(--hud-bg)]/50">
                    <tr>
                        <th className="px-4 py-3 text-left text-sm text-[var(--hud-text-muted)]">Model</th>
                        <th className="px-4 py-3 text-center text-sm text-[var(--hud-text-muted)]">Accuracy</th>
                        <th className="px-4 py-3 text-center text-sm text-[var(--hud-text-muted)]">F1-Score</th>
                        <th className="px-4 py-3 text-center text-sm text-[var(--hud-text-muted)]">Süre</th>
                        <th className="px-4 py-3 text-center text-sm text-[var(--hud-text-muted)]">Params</th>
                    </tr>
                </thead>
                <tbody>
                    {comparison.map((model, idx) => (
                        <tr key={model.id} className={`border-t border-[var(--hud-border)] ${idx === 0 ? 'bg-purple-500/10' : ''}`}>
                            <td className="px-4 py-3">
                                <div className="flex items-center gap-2">
                                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: MODEL_COLORS[model.id] }} />
                                    <span className="text-[var(--hud-text)] font-medium">{model.name}</span>
                                    {idx === 0 && <span className="text-xs bg-yellow-500/20 text-yellow-400 px-2 py-0.5 rounded">🏆 Best</span>}
                                </div>
                            </td>
                            <td className="px-4 py-3 text-center">
                                <span className="text-lg font-bold text-[var(--hud-text)]">{(model.accuracy * 100).toFixed(1)}%</span>
                            </td>
                            <td className="px-4 py-3 text-center">
                                <span className="text-[var(--hud-text)]">{(model.f1_score * 100).toFixed(1)}%</span>
                            </td>
                            <td className="px-4 py-3 text-center text-[var(--hud-text-muted)]">
                                {model.train_time ? `${model.train_time.toFixed(1)}s` : '-'}
                            </td>
                            <td className="px-4 py-3 text-center text-[var(--hud-text-muted)]">{model.params}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );

    // Radar Chart
    const ComparisonRadar = () => {
        const metrics = ['accuracy', 'f1', 'precision', 'recall'];
        const radarData = metrics.map(metric => {
            const entry = { metric };
            comparison.forEach(m => {
                const val = metric === 'f1' ? m.f1_score : (m[metric] || 0);
                entry[m.name] = val * 100;
            });
            return entry;
        });

        return (
            <div className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-5">
                <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">📊 Metrik Karşılaştırma</h3>
                <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                        <PolarGrid stroke="#374151" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#9CA3AF', fontSize: 12 }} />
                        <PolarRadiusAxis domain={[0, 100]} tick={{ fill: '#6B7280' }} />
                        {comparison.map((m) => (
                            <Radar
                                key={m.id}
                                name={m.name}
                                dataKey={m.name}
                                stroke={MODEL_COLORS[m.id]}
                                fill={MODEL_COLORS[m.id]}
                                fillOpacity={0.2}
                            />
                        ))}
                        <Legend />
                    </RadarChart>
                </ResponsiveContainer>
            </div>
        );
    };

    // Training Form
    const TrainingForm = () => (
        <div className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-5">
            <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">🏋️ Model Eğitimi Başlat</h3>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Model Tipi</label>
                    <select
                        value={trainConfig.model_type}
                        onChange={(e) => setTrainConfig({ ...trainConfig, model_type: e.target.value })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    >
                        <option value="lstm">LSTM</option>
                        <option value="bilstm">BiLSTM + Attention ⭐</option>
                        <option value="transformer">Transformer</option>
                        <option value="gru">GRU (Hafif)</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Epochs</label>
                    <input
                        type="number"
                        value={trainConfig.epochs}
                        onChange={(e) => setTrainConfig({ ...trainConfig, epochs: parseInt(e.target.value) })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    />
                </div>
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Batch Size</label>
                    <input
                        type="number"
                        value={trainConfig.batch_size}
                        onChange={(e) => setTrainConfig({ ...trainConfig, batch_size: parseInt(e.target.value) })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    />
                </div>
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Learning Rate</label>
                    <input
                        type="number"
                        step="0.0001"
                        value={trainConfig.learning_rate}
                        onChange={(e) => setTrainConfig({ ...trainConfig, learning_rate: parseFloat(e.target.value) })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    />
                </div>
            </div>

            <div className="flex items-center gap-4 mb-4">
                <label className="flex items-center gap-2 text-[var(--hud-text-muted)]">
                    <input
                        type="checkbox"
                        checked={trainConfig.use_smote}
                        onChange={(e) => setTrainConfig({ ...trainConfig, use_smote: e.target.checked })}
                        className="rounded"
                    />
                    SMOTE (Veri Dengeleme)
                </label>
                <label className="flex items-center gap-2 text-[var(--hud-text-muted)]">
                    <input
                        type="checkbox"
                        checked={trainConfig.use_attention}
                        onChange={(e) => setTrainConfig({ ...trainConfig, use_attention: e.target.checked })}
                        className="rounded"
                    />
                    Attention
                </label>
            </div>

            <button
                onClick={handleStartTraining}
                className="w-full bg-purple-600 hover:bg-purple-700 text-[var(--hud-text)] py-2 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
                <Play className="w-4 h-4" />
                Eğitimi Başlat
            </button>
        </div>
    );

    // Optimization Form
    const OptimizationForm = () => (
        <div className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-5">
            <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">🔧 Hiperparametre Optimizasyonu</h3>

            <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Model</label>
                    <select
                        value={optConfig.model_type}
                        onChange={(e) => setOptConfig({ ...optConfig, model_type: e.target.value })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    >
                        <option value="bilstm">BiLSTM + Attention</option>
                        <option value="transformer">Transformer</option>
                        <option value="lstm">LSTM</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Algoritma</label>
                    <select
                        value={optConfig.algorithm}
                        onChange={(e) => setOptConfig({ ...optConfig, algorithm: e.target.value })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    >
                        <option value="ssa">SSA (Salp Swarm) 🏆</option>
                        <option value="pso">PSO (Particle Swarm)</option>
                        <option value="jaya">JAYA (Parametresiz)</option>
                    </select>
                </div>
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Max İterasyon</label>
                    <input
                        type="number"
                        value={optConfig.max_iterations}
                        onChange={(e) => setOptConfig({ ...optConfig, max_iterations: parseInt(e.target.value) })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    />
                </div>
                <div>
                    <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Popülasyon</label>
                    <input
                        type="number"
                        value={optConfig.population_size}
                        onChange={(e) => setOptConfig({ ...optConfig, population_size: parseInt(e.target.value) })}
                        className="w-full bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg px-3 py-2 text-[var(--hud-text)]"
                    />
                </div>
            </div>

            <button
                onClick={handleStartOptimization}
                className="w-full bg-blue-600 hover:bg-blue-700 text-[var(--hud-text)] py-2 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
                <Zap className="w-4 h-4" />
                Optimizasyonu Başlat
            </button>
        </div>
    );

    // Active Optimizations
    const ActiveOptimizations = () => (
        <div className="space-y-3">
            {optimizations.filter(o => o.status === 'running').map(opt => (
                <div key={opt.id} className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-4">
                    <div className="flex items-center justify-between mb-2">
                        <span className="text-[var(--hud-text)] font-medium">{opt.algorithm} - {opt.model_type}</span>
                        <span className="text-green-400 text-sm flex items-center gap-1">
                            <Activity className="w-3 h-3 animate-pulse" />
                            Running
                        </span>
                    </div>
                    <div className="mb-2">
                        <div className="flex justify-between text-sm text-[var(--hud-text-muted)] mb-1">
                            <span>İlerleme</span>
                            <span>{opt.progress?.toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-[var(--hud-panel)] rounded-full h-2">
                            <div
                                className="bg-blue-500 h-2 rounded-full transition-all"
                                style={{ width: `${opt.progress}%` }}
                            />
                        </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                            <span className="text-[var(--hud-text-dim)]">Best Score:</span>
                            <span className="text-[var(--hud-text)] ml-2">{(opt.best_score * 100).toFixed(2)}%</span>
                        </div>
                        <div>
                            <span className="text-[var(--hud-text-dim)]">İterasyon:</span>
                            <span className="text-[var(--hud-text)] ml-2">{opt.current_iteration}/{opt.max_iterations}</span>
                        </div>
                    </div>
                </div>
            ))}
            {optimizations.filter(o => o.status === 'running').length === 0 && (
                <p className="text-[var(--hud-text-dim)] text-center py-4">Aktif optimizasyon yok</p>
            )}
        </div>
    );

    // Ensemble Builder
    const EnsembleBuilder = () => (
        <div className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-5">
            <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">🎯 Ensemble Model Oluştur</h3>

            <div className="mb-4">
                <label className="block text-sm text-[var(--hud-text-muted)] mb-2">Modelleri Seç (min 2)</label>
                <div className="grid grid-cols-2 gap-2">
                    {['lstm', 'bilstm', 'transformer', 'gru'].map(id => (
                        <label key={id} className={`flex items-center gap-2 p-3 rounded-lg border cursor-pointer transition-all ${ensembleConfig.model_ids.includes(id)
                            ? 'bg-purple-500/20 border-purple-500'
                            : 'bg-[var(--hud-bg)]/50 border-[var(--hud-border)] hover:border-[var(--hud-border)]'
                            }`}>
                            <input
                                type="checkbox"
                                checked={ensembleConfig.model_ids.includes(id)}
                                onChange={(e) => {
                                    if (e.target.checked) {
                                        setEnsembleConfig({
                                            ...ensembleConfig,
                                            model_ids: [...ensembleConfig.model_ids, id]
                                        });
                                    } else {
                                        setEnsembleConfig({
                                            ...ensembleConfig,
                                            model_ids: ensembleConfig.model_ids.filter(m => m !== id)
                                        });
                                    }
                                }}
                                className="hidden"
                            />
                            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: MODEL_COLORS[id] }} />
                            <span className="text-[var(--hud-text)]">{id.toUpperCase()}</span>
                        </label>
                    ))}
                </div>
            </div>

            <div className="mb-4">
                <label className="block text-sm text-[var(--hud-text-muted)] mb-1">Voting Stratejisi</label>
                <div className="flex gap-2">
                    <button
                        onClick={() => setEnsembleConfig({ ...ensembleConfig, voting: 'soft' })}
                        className={`flex-1 py-2 rounded-lg transition-colors ${ensembleConfig.voting === 'soft'
                            ? 'bg-purple-600 text-[var(--hud-text)]'
                            : 'bg-[var(--hud-bg)] text-[var(--hud-text-muted)] hover:bg-[var(--hud-surface)]'
                            }`}
                    >
                        Soft Voting
                    </button>
                    <button
                        onClick={() => setEnsembleConfig({ ...ensembleConfig, voting: 'hard' })}
                        className={`flex-1 py-2 rounded-lg transition-colors ${ensembleConfig.voting === 'hard'
                            ? 'bg-purple-600 text-[var(--hud-text)]'
                            : 'bg-[var(--hud-bg)] text-[var(--hud-text-muted)] hover:bg-[var(--hud-surface)]'
                            }`}
                    >
                        Hard Voting
                    </button>
                </div>
            </div>

            <button
                onClick={handleCreateEnsemble}
                disabled={ensembleConfig.model_ids.length < 2}
                className="w-full bg-green-600 hover:bg-green-700 disabled:bg-[var(--hud-panel)] disabled:cursor-not-allowed text-[var(--hud-text)] py-2 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
                <Layers className="w-4 h-4" />
                Ensemble Oluştur
            </button>
        </div>
    );

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500" />
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-xl font-semibold text-[var(--hud-text)] flex items-center gap-2">
                        <Brain className="w-6 h-6 text-purple-400" />
                        Gelişmiş Modeller
                    </h1>
                    <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">BiLSTM, Transformer, GRU, Ensemble</p>
                </div>
                <button
                    onClick={loadData}
                    className="flex items-center gap-2 px-4 py-2 bg-[var(--hud-panel)] hover:bg-[var(--hud-border)] rounded-lg text-[var(--hud-text)] transition-colors"
                >
                    <RefreshCw className="w-4 h-4" />
                    Yenile
                </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-2 overflow-x-auto pb-2">
                {TABS.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg whitespace-nowrap transition-colors ${activeTab === tab.id
                            ? 'bg-purple-600 text-[var(--hud-text)]'
                            : 'bg-[var(--hud-surface)] text-[var(--hud-text-muted)] hover:bg-[var(--hud-panel)]'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            {activeTab === 'models' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {models.map(model => (
                        <ModelCard key={model.id} model={model} />
                    ))}
                </div>
            )}

            {activeTab === 'compare' && (
                <div className="space-y-6">
                    <ComparisonTable />
                    <ComparisonRadar />
                </div>
            )}

            {activeTab === 'train' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <TrainingForm />
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold text-[var(--hud-text)]">📋 Aktif Eğitimler</h3>
                        {trainings.filter(t => t.status === 'training').length > 0 ? (
                            trainings.filter(t => t.status === 'training').map(t => (
                                <div key={t.id} className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-4">
                                    <div className="flex justify-between mb-2">
                                        <span className="text-[var(--hud-text)]">{t.model_type}</span>
                                        <span className="text-green-400 text-sm">Training</span>
                                    </div>
                                    <div className="w-full bg-[var(--hud-panel)] rounded-full h-2">
                                        <div
                                            className="bg-purple-500 h-2 rounded-full"
                                            style={{ width: `${t.progress}%` }}
                                        />
                                    </div>
                                    <p className="text-sm text-[var(--hud-text-muted)] mt-1">
                                        Epoch {t.current_epoch}/{t.total_epochs}
                                    </p>
                                </div>
                            ))
                        ) : (
                            <p className="text-[var(--hud-text-dim)]">Aktif eğitim yok</p>
                        )}
                    </div>
                </div>
            )}

            {activeTab === 'optimize' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <OptimizationForm />
                    <div>
                        <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">📊 Aktif Optimizasyonlar</h3>
                        <ActiveOptimizations />
                    </div>
                </div>
            )}

            {activeTab === 'ensemble' && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <EnsembleBuilder />
                    <div className="bg-[var(--hud-surface)]/50 border border-[var(--hud-border)] rounded-xl p-5">
                        <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">ℹ️ Ensemble Hakkında</h3>
                        <div className="space-y-3 text-[var(--hud-text-muted)] text-sm">
                            <p>
                                <strong className="text-[var(--hud-text)]">Soft Voting:</strong> Modellerin olasılık tahminlerini ortalar.
                                Daha güvenilir sonuçlar.
                            </p>
                            <p>
                                <strong className="text-[var(--hud-text)]">Hard Voting:</strong> Her model bir oy verir, çoğunluk kazanır.
                                Daha hızlı.
                            </p>
                            <p>
                                <strong className="text-[var(--hud-text)]">Önerilen:</strong> BiLSTM + Transformer kombinasyonu en iyi sonuçları verir.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
