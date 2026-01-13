import { useState, useEffect } from 'react';
import { advancedMLApi } from '../services/api';

export default function AdvancedML() {
    const [activeTab, setActiveTab] = useState('automl');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // AutoML State
    const [automlStatus, setAutomlStatus] = useState(null);
    const [automlConfig, setAutomlConfig] = useState({
        search_strategy: 'bayesian',
        model_type: 'lstm',
        max_trials: 10,
        epochs_per_trial: 10,
    });

    // XAI State
    const [featureImportance, setFeatureImportance] = useState(null);
    const [selectedModel, setSelectedModel] = useState('latest');

    // Drift State
    const [driftStatus, setDriftStatus] = useState(null);

    // Federated State
    const [flStatus, setFLStatus] = useState(null);

    // A/B Testing State
    const [abTests, setABTests] = useState([]);

    const tabs = [
        { id: 'automl', name: 'AutoML', icon: '🤖' },
        { id: 'xai', name: 'Explainability', icon: '🔍' },
        { id: 'ab', name: 'A/B Testing', icon: '⚖️' },
        { id: 'drift', name: 'Drift Detection', icon: '📊' },
        { id: 'federated', name: 'Federated Learning', icon: '🌐' },
    ];

    const loadTabData = async (tab) => {
        setLoading(true);
        setError(null);
        try {
            if (tab === 'automl') {
                const autoRes = await advancedMLApi.automlStatus();
                setAutomlStatus(autoRes.data);
            } else if (tab === 'xai') {
                const xaiRes = await advancedMLApi.getFeatureImportance(selectedModel);
                setFeatureImportance(xaiRes.data);
            } else if (tab === 'drift') {
                const driftRes = await advancedMLApi.getDriftStatus();
                setDriftStatus(driftRes.data);
            } else if (tab === 'federated') {
                const flRes = await advancedMLApi.getFLStatus();
                setFLStatus(flRes.data);
            } else if (tab === 'ab') {
                const abRes = await advancedMLApi.getABTests();
                setABTests(abRes.data?.tests || []);
            }
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    useEffect(() => {
        loadTabData(activeTab);
    }, [activeTab]);

    const startAutoMLSearch = async () => {
        setLoading(true);
        try {
            await advancedMLApi.automlSearch(automlConfig);
            loadTabData('automl');
        } catch (err) {
            setError(err.message);
        }
        setLoading(false);
    };

    const renderAutoML = () => (
        <div className="space-y-6">
            {/* Config Card */}
            <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4">AutoML Konfigürasyonu</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Search Strategy</label>
                        <select
                            className="input w-full"
                            value={automlConfig.search_strategy}
                            onChange={(e) => setAutomlConfig({ ...automlConfig, search_strategy: e.target.value })}
                        >
                            <option value="grid">Grid Search</option>
                            <option value="random">Random Search</option>
                            <option value="bayesian">Bayesian Optimization</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Model Type</label>
                        <select
                            className="input w-full"
                            value={automlConfig.model_type}
                            onChange={(e) => setAutomlConfig({ ...automlConfig, model_type: e.target.value })}
                        >
                            <option value="lstm">LSTM</option>
                            <option value="bilstm">BiLSTM</option>
                            <option value="gru">GRU</option>
                            <option value="cnn_lstm">CNN-LSTM</option>
                            <option value="transformer">Transformer</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Max Trials</label>
                        <input
                            type="number"
                            className="input w-full"
                            value={automlConfig.max_trials}
                            onChange={(e) => setAutomlConfig({ ...automlConfig, max_trials: parseInt(e.target.value) })}
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-slate-400 mb-2">Epochs per Trial</label>
                        <input
                            type="number"
                            className="input w-full"
                            value={automlConfig.epochs_per_trial}
                            onChange={(e) => setAutomlConfig({ ...automlConfig, epochs_per_trial: parseInt(e.target.value) })}
                        />
                    </div>
                </div>
                <button
                    className="btn-primary mt-4"
                    onClick={startAutoMLSearch}
                    disabled={loading}
                >
                    {loading ? 'Başlatılıyor...' : '🚀 AutoML Başlat'}
                </button>
            </div>

            {/* Status Card */}
            {automlStatus && (
                <div className="card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">AutoML Durumu</h3>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                            <div className="text-3xl font-bold text-cyan-400">{automlStatus.completed_trials || 0}</div>
                            <div className="text-sm text-slate-400">Tamamlanan Trial</div>
                        </div>
                        <div className="text-center">
                            <div className="text-3xl font-bold text-green-400">
                                {automlStatus.best_accuracy ? `${(automlStatus.best_accuracy * 100).toFixed(2)}%` : '-'}
                            </div>
                            <div className="text-sm text-slate-400">En İyi Accuracy</div>
                        </div>
                        <div className="text-center">
                            <div className={`text-3xl font-bold ${automlStatus.is_running ? 'text-yellow-400' : 'text-slate-400'}`}>
                                {automlStatus.is_running ? '⏳' : '✅'}
                            </div>
                            <div className="text-sm text-slate-400">
                                {automlStatus.is_running ? 'Çalışıyor' : 'Bekleniyor'}
                            </div>
                        </div>
                        <div className="text-center">
                            <div className="text-3xl font-bold text-purple-400">{automlStatus.total_trials || 0}</div>
                            <div className="text-sm text-slate-400">Toplam Trial</div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );

    const renderXAI = () => (
        <div className="space-y-6">
            {/* Model Selector */}
            <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Model Seçimi</h3>
                <select
                    className="input w-full max-w-xs"
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                >
                    <option value="latest">En Son Model</option>
                    <option value="best">En İyi Model</option>
                </select>
                <button
                    className="btn-secondary ml-4"
                    onClick={() => loadTabData('xai')}
                >
                    Analiz Et
                </button>
            </div>

            {/* Feature Importance */}
            {featureImportance && (
                <div className="card p-6">
                    <h3 className="text-lg font-semibold text-white mb-4">Feature Importance</h3>
                    <div className="space-y-3">
                        {featureImportance.features?.slice(0, 15).map((feature, idx) => (
                            <div key={idx} className="flex items-center gap-3">
                                <span className="w-32 text-sm text-slate-400 truncate">{feature.name}</span>
                                <div className="flex-1 h-4 bg-slate-700 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-cyan-500 to-blue-500"
                                        style={{ width: `${feature.importance * 100}%` }}
                                    />
                                </div>
                                <span className="w-16 text-sm text-slate-300 text-right">
                                    {(feature.importance * 100).toFixed(1)}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );

    const renderABTesting = () => (
        <div className="space-y-6">
            <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4">A/B Test Oluştur</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <input className="input" placeholder="Test Adı" />
                    <input className="input" placeholder="Model A ID" />
                    <input className="input" placeholder="Model B ID" />
                </div>
                <button className="btn-primary mt-4">➕ Test Oluştur</button>
            </div>

            <div className="card p-6">
                <h3 className="text-lg font-semibold text-white mb-4">Aktif Testler</h3>
                {abTests.length === 0 ? (
                    <p className="text-slate-400">Henüz aktif test yok.</p>
                ) : (
                    <div className="space-y-3">
                        {abTests.map((test, idx) => (
                            <div key={idx} className="p-4 bg-slate-800 rounded-lg flex justify-between items-center">
                                <div>
                                    <div className="font-medium text-white">{test.name}</div>
                                    <div className="text-sm text-slate-400">
                                        {test.model_a} vs {test.model_b}
                                    </div>
                                </div>
                                <span className={`px-3 py-1 rounded-full text-xs ${test.status === 'running' ? 'bg-green-500/20 text-green-400' : 'bg-slate-600 text-slate-300'
                                    }`}>
                                    {test.status}
                                </span>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );

    const renderDrift = () => (
        <div className="space-y-6">
            {driftStatus && (
                <>
                    {/* Status Overview */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div className={`card p-6 border-2 ${driftStatus.has_drift ? 'border-red-500' : 'border-green-500'
                            }`}>
                            <div className="text-center">
                                <div className="text-4xl mb-2">{driftStatus.has_drift ? '⚠️' : '✅'}</div>
                                <div className="text-lg font-semibold text-white">
                                    {driftStatus.has_drift ? 'Drift Tespit Edildi' : 'Normal'}
                                </div>
                            </div>
                        </div>
                        <div className="card p-6">
                            <div className="text-center">
                                <div className="text-3xl font-bold text-cyan-400">
                                    {driftStatus.psi_score?.toFixed(3) || '-'}
                                </div>
                                <div className="text-sm text-slate-400">PSI Score</div>
                            </div>
                        </div>
                        <div className="card p-6">
                            <div className="text-center">
                                <div className="text-3xl font-bold text-purple-400">
                                    {driftStatus.ks_statistic?.toFixed(3) || '-'}
                                </div>
                                <div className="text-sm text-slate-400">KS Statistic</div>
                            </div>
                        </div>
                    </div>

                    {/* Severity */}
                    <div className="card p-6">
                        <h3 className="text-lg font-semibold text-white mb-4">Drift Seviyesi</h3>
                        <div className="flex items-center gap-4">
                            <div className="flex-1 h-4 bg-slate-700 rounded-full overflow-hidden">
                                <div
                                    className={`h-full ${driftStatus.severity === 'high' ? 'bg-red-500' :
                                        driftStatus.severity === 'medium' ? 'bg-yellow-500' : 'bg-green-500'
                                        }`}
                                    style={{ width: `${(driftStatus.psi_score || 0) * 200}%` }}
                                />
                            </div>
                            <span className={`px-3 py-1 rounded-full text-xs font-medium ${driftStatus.severity === 'high' ? 'bg-red-500/20 text-red-400' :
                                driftStatus.severity === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
                                    'bg-green-500/20 text-green-400'
                                }`}>
                                {driftStatus.severity || 'none'}
                            </span>
                        </div>
                    </div>
                </>
            )}
        </div>
    );

    const renderFederated = () => (
        <div className="space-y-6">
            {flStatus && (
                <>
                    {/* Status Cards */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div className="card p-6 text-center">
                            <div className={`text-4xl mb-2 ${flStatus.is_running ? 'animate-pulse' : ''}`}>
                                {flStatus.is_running ? '🔄' : '⏸️'}
                            </div>
                            <div className="text-sm text-slate-400">
                                {flStatus.is_running ? 'Aktif' : 'Bekleniyor'}
                            </div>
                        </div>
                        <div className="card p-6 text-center">
                            <div className="text-3xl font-bold text-cyan-400">{flStatus.current_round || 0}</div>
                            <div className="text-sm text-slate-400">Mevcut Round</div>
                        </div>
                        <div className="card p-6 text-center">
                            <div className="text-3xl font-bold text-green-400">{flStatus.connected_clients || 0}</div>
                            <div className="text-sm text-slate-400">Bağlı Client</div>
                        </div>
                        <div className="card p-6 text-center">
                            <div className="text-3xl font-bold text-purple-400">
                                {flStatus.global_accuracy ? `${(flStatus.global_accuracy * 100).toFixed(1)}%` : '-'}
                            </div>
                            <div className="text-sm text-slate-400">Global Accuracy</div>
                        </div>
                    </div>

                    {/* Start Round Button */}
                    <div className="card p-6">
                        <button
                            className="btn-primary"
                            onClick={async () => {
                                await advancedMLApi.startFLRound();
                                loadTabData('federated');
                            }}
                            disabled={flStatus.is_running}
                        >
                            🚀 Yeni Round Başlat
                        </button>
                    </div>
                </>
            )}
        </div>
    );

    return (
        <div className="p-6">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-3xl font-bold text-white">🧠 Advanced ML</h1>
                <p className="text-slate-400 mt-1">AutoML, Explainability, A/B Testing, Drift Detection, Federated Learning</p>
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
                            ? 'bg-cyan-500 text-white'
                            : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
                            }`}
                    >
                        {tab.icon} {tab.name}
                    </button>
                ))}
            </div>

            {/* Tab Content */}
            {loading ? (
                <div className="flex justify-center py-12">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500"></div>
                </div>
            ) : (
                <>
                    {activeTab === 'automl' && renderAutoML()}
                    {activeTab === 'xai' && renderXAI()}
                    {activeTab === 'ab' && renderABTesting()}
                    {activeTab === 'drift' && renderDrift()}
                    {activeTab === 'federated' && renderFederated()}
                </>
            )}
        </div>
    );
}
