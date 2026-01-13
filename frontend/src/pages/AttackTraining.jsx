import { useState, useEffect } from 'react';
import {
    Target, Zap, Shield, Play, Pause, RefreshCw,
    Activity, AlertTriangle, CheckCircle, Clock, Cpu, Database,
    BarChart3, Settings, Layers, Radio
} from 'lucide-react';
import api from '../services/api';

const ATTACK_TYPES = [
    { id: 'all', name: 'Tüm Saldırılar', icon: '🎯', severity: 'mixed' },
    { id: 'dos', name: 'DoS', icon: '💥', severity: 'high' },
    { id: 'ddos', name: 'DDoS', icon: '🌊', severity: 'critical' },
    { id: 'probe', name: 'Probe/Scan', icon: '🔍', severity: 'medium' },
    { id: 'r2l', name: 'R2L', icon: '🔓', severity: 'high' },
    { id: 'u2r', name: 'U2R', icon: '👤', severity: 'critical' },
    { id: 'botnet', name: 'Botnet', icon: '🤖', severity: 'critical' },
    { id: 'mirai', name: 'Mirai', icon: '🦠', severity: 'critical' },
    { id: 'gafgyt', name: 'Gafgyt', icon: '🐛', severity: 'critical' },
];

const DATASETS = [
    { id: 'nsl_kdd', name: 'NSL-KDD', size: '107 MB', attacks: ['dos', 'probe', 'r2l', 'u2r'] },
    { id: 'bot_iot', name: 'BoT-IoT', size: '7.76 GB', attacks: ['mirai', 'gafgyt', 'ddos'] },
    { id: 'cicids2017', name: 'CICIDS2017', size: '258 MB', attacks: ['dos', 'ddos', 'botnet', 'bruteforce'] },
];

const SEVERITY_COLORS = {
    low: 'bg-green-500/20 text-green-400',
    medium: 'bg-yellow-500/20 text-yellow-400',
    high: 'bg-orange-500/20 text-orange-400',
    critical: 'bg-red-500/20 text-red-400',
    mixed: 'bg-purple-500/20 text-purple-400',
};

export default function AttackTraining() {
    const [activeTab, setActiveTab] = useState('train');
    const [loading, setLoading] = useState(false);
    const [trainingSessions, setTrainingSessions] = useState([]);
    const [trainedModels, setTrainedModels] = useState([]);

    // Training config
    const [config, setConfig] = useState({
        dataset: 'nsl_kdd',
        attack_types: ['all'],
        model_type: 'ssa_lstmids',
        epochs: 100,
        batch_size: 120,
        use_smote: true,
        use_ssa_bayesian: false,
        patience: 20,
        max_samples: 100000,
        model_name: ''
    });

    // Real-time IDS state
    const [idsStatus, setIdsStatus] = useState(null);
    const [idsAlerts, setIdsAlerts] = useState([]);
    const [idsConfig, setIdsConfig] = useState({
        model_path: '',
        threshold: 0.5,
        window_size: 10
    });

    useEffect(() => {
        loadData();
    }, []);

    useEffect(() => {
        // IDS status polling
        let interval;
        if (idsStatus?.status === 'running') {
            interval = setInterval(fetchIdsStatus, 2000);
        }
        return () => clearInterval(interval);
    }, [idsStatus?.status]);

    const loadData = async () => {
        try {
            setLoading(true);
            const [sessionsRes, modelsRes] = await Promise.all([
                api.get('/training/attack-specific'),
                api.get('/training/models')
            ]);

            if (sessionsRes.data.success) setTrainingSessions(sessionsRes.data.data);
            if (modelsRes.data.success) setTrainedModels(modelsRes.data.data);
        } catch (err) {
            console.error('Error loading data:', err);
        } finally {
            setLoading(false);
        }
    };

    const fetchIdsStatus = async () => {
        try {
            const [statusRes, alertsRes] = await Promise.all([
                api.get('/training/realtime-ids/status'),
                api.get('/training/realtime-ids/alerts?limit=20')
            ]);
            if (statusRes.data.success) setIdsStatus(statusRes.data.data);
            if (alertsRes.data.success) setIdsAlerts(alertsRes.data.data);
        } catch (err) {
            console.error('Error fetching IDS status:', err);
        }
    };

    const startTraining = async () => {
        try {
            const res = await api.post('/training/attack-specific', config);
            if (res.data.success) {
                loadData();
            }
        } catch (err) {
            console.error('Error starting training:', err);
        }
    };

    const startIDS = async () => {
        try {
            const res = await api.post('/training/realtime-ids/start', idsConfig);
            if (res.data.success) {
                setIdsStatus(res.data.data);
                fetchIdsStatus();
            }
        } catch (err) {
            console.error('Error starting IDS:', err);
        }
    };

    const stopIDS = async () => {
        try {
            await api.post('/training/realtime-ids/stop');
            setIdsStatus({ status: 'stopped' });
        } catch (err) {
            console.error('Error stopping IDS:', err);
        }
    };

    const toggleAttackType = (attackId) => {
        if (attackId === 'all') {
            setConfig({ ...config, attack_types: ['all'] });
        } else {
            const current = config.attack_types.filter(a => a !== 'all');
            if (current.includes(attackId)) {
                const updated = current.filter(a => a !== attackId);
                setConfig({ ...config, attack_types: updated.length ? updated : ['all'] });
            } else {
                setConfig({ ...config, attack_types: [...current, attackId] });
            }
        }
    };

    // Training Tab
    const TrainingTab = () => (
        <div className="space-y-6">
            {/* Dataset Selection */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Database className="w-5 h-5 text-blue-400" />
                    Dataset Seçimi
                </h3>
                <div className="grid grid-cols-3 gap-3">
                    {DATASETS.map(ds => (
                        <button
                            key={ds.id}
                            onClick={() => setConfig({ ...config, dataset: ds.id })}
                            className={`p-4 rounded-lg border transition-all text-left ${config.dataset === ds.id
                                ? 'bg-blue-500/20 border-blue-500'
                                : 'bg-gray-900/50 border-gray-700 hover:border-gray-600'
                                }`}
                        >
                            <h4 className="font-medium text-white">{ds.name}</h4>
                            <p className="text-sm text-gray-400">{ds.size}</p>
                            <div className="flex flex-wrap gap-1 mt-2">
                                {ds.attacks.slice(0, 3).map(a => (
                                    <span key={a} className="text-xs bg-gray-700 px-1.5 py-0.5 rounded">
                                        {a}
                                    </span>
                                ))}
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* Attack Type Selection */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Target className="w-5 h-5 text-red-400" />
                    Saldırı Tipi Seçimi
                </h3>
                <div className="grid grid-cols-3 md:grid-cols-5 gap-2">
                    {ATTACK_TYPES.map(attack => (
                        <button
                            key={attack.id}
                            onClick={() => toggleAttackType(attack.id)}
                            className={`p-3 rounded-lg border transition-all ${config.attack_types.includes(attack.id)
                                ? 'bg-purple-500/20 border-purple-500'
                                : 'bg-gray-900/50 border-gray-700 hover:border-gray-600'
                                }`}
                        >
                            <span className="text-2xl">{attack.icon}</span>
                            <p className="text-sm text-white mt-1">{attack.name}</p>
                            <span className={`text-xs px-1.5 py-0.5 rounded ${SEVERITY_COLORS[attack.severity]}`}>
                                {attack.severity}
                            </span>
                        </button>
                    ))}
                </div>
            </div>

            {/* Training Parameters */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Settings className="w-5 h-5 text-gray-400" />
                    Eğitim Parametreleri
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Epochs</label>
                        <input
                            type="number"
                            value={config.epochs}
                            onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
                        <input
                            type="number"
                            value={config.batch_size}
                            onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Patience</label>
                        <input
                            type="number"
                            value={config.patience}
                            onChange={(e) => setConfig({ ...config, patience: parseInt(e.target.value) })}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Max Samples</label>
                        <input
                            type="number"
                            value={config.max_samples}
                            onChange={(e) => setConfig({ ...config, max_samples: parseInt(e.target.value) })}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                </div>

                <div className="flex items-center gap-6 mt-4">
                    <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={config.use_smote}
                            onChange={(e) => setConfig({ ...config, use_smote: e.target.checked })}
                            className="rounded"
                        />
                        SMOTE (Veri Dengeleme)
                    </label>
                    <label className="flex items-center gap-2 text-gray-400 cursor-pointer">
                        <input
                            type="checkbox"
                            checked={config.use_ssa_bayesian}
                            onChange={(e) => setConfig({ ...config, use_ssa_bayesian: e.target.checked })}
                            className="rounded"
                        />
                        SSA + Bayesian Optimizasyon
                    </label>
                </div>

                <div className="mt-4">
                    <label className="block text-sm text-gray-400 mb-1">Model İsmi (opsiyonel)</label>
                    <input
                        type="text"
                        value={config.model_name}
                        onChange={(e) => setConfig({ ...config, model_name: e.target.value })}
                        placeholder="örn: ddos_detector_v1"
                        className="w-full md:w-1/2 bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                    />
                </div>
            </div>

            {/* Start Button */}
            <button
                onClick={startTraining}
                className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white py-3 rounded-xl flex items-center justify-center gap-2 transition-all font-semibold"
            >
                <Play className="w-5 h-5" />
                Eğitimi Başlat
            </button>

            {/* Active Sessions */}
            {trainingSessions.filter(s => s.status !== 'completed' && s.status !== 'failed').length > 0 && (
                <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
                    <h3 className="text-lg font-semibold text-white mb-4">📋 Aktif Eğitimler</h3>
                    {trainingSessions
                        .filter(s => s.status !== 'completed' && s.status !== 'failed')
                        .map(session => (
                            <div key={session.id} className="bg-gray-900/50 rounded-lg p-4 mb-2">
                                <div className="flex justify-between mb-2">
                                    <span className="text-white">{session.dataset} - {session.attack_types.join(', ')}</span>
                                    <span className="text-blue-400 text-sm">{session.status}</span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-2">
                                    <div
                                        className="bg-blue-500 h-2 rounded-full transition-all"
                                        style={{ width: `${session.progress}%` }}
                                    />
                                </div>
                                <p className="text-sm text-gray-400 mt-1">{session.progress}% tamamlandı</p>
                            </div>
                        ))}
                </div>
            )}
        </div>
    );

    // Real-time IDS Tab
    const RealTimeIDSTab = () => (
        <div className="space-y-6">
            {/* IDS Config */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <Radio className="w-5 h-5 text-green-400" />
                    Real-time IDS Ayarları
                </h3>
                <div className="grid grid-cols-3 gap-4 mb-4">
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Model</label>
                        <select
                            value={idsConfig.model_path}
                            onChange={(e) => setIdsConfig({ ...idsConfig, model_path: e.target.value })}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        >
                            <option value="">Model Seç...</option>
                            {trainedModels.map(m => (
                                <option key={m.name} value={m.path}>{m.name}</option>
                            ))}
                        </select>
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Threshold</label>
                        <input
                            type="number"
                            step="0.1"
                            min="0"
                            max="1"
                            value={idsConfig.threshold}
                            onChange={(e) => setIdsConfig({ ...idsConfig, threshold: parseFloat(e.target.value) })}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                    <div>
                        <label className="block text-sm text-gray-400 mb-1">Window Size</label>
                        <input
                            type="number"
                            value={idsConfig.window_size}
                            onChange={(e) => setIdsConfig({ ...idsConfig, window_size: parseInt(e.target.value) })}
                            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-white"
                        />
                    </div>
                </div>

                <div className="flex gap-2">
                    {idsStatus?.status === 'running' ? (
                        <button
                            onClick={stopIDS}
                            className="flex-1 bg-red-600 hover:bg-red-700 text-white py-2 rounded-lg flex items-center justify-center gap-2"
                        >
                            <Pause className="w-4 h-4" />
                            IDS Durdur
                        </button>
                    ) : (
                        <button
                            onClick={startIDS}
                            disabled={!idsConfig.model_path}
                            className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 text-white py-2 rounded-lg flex items-center justify-center gap-2"
                        >
                            <Play className="w-4 h-4" />
                            IDS Başlat
                        </button>
                    )}
                </div>
            </div>

            {/* IDS Metrics */}
            {idsStatus?.status === 'running' && idsStatus.metrics && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                        <p className="text-gray-400 text-sm">Toplam Paket</p>
                        <p className="text-2xl font-bold text-white">{idsStatus.metrics.total_packets.toLocaleString()}</p>
                    </div>
                    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                        <p className="text-gray-400 text-sm">Saldırı Tespit</p>
                        <p className="text-2xl font-bold text-red-400">{idsStatus.metrics.attack_packets}</p>
                    </div>
                    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                        <p className="text-gray-400 text-sm">Paket/Saniye</p>
                        <p className="text-2xl font-bold text-blue-400">{idsStatus.metrics.packets_per_second?.toFixed(1)}</p>
                    </div>
                    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
                        <p className="text-gray-400 text-sm">İşlem Süresi</p>
                        <p className="text-2xl font-bold text-green-400">{idsStatus.metrics.processing_time_avg_ms?.toFixed(2)} ms</p>
                    </div>
                </div>
            )}

            {/* Alerts */}
            <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                    <AlertTriangle className="w-5 h-5 text-yellow-400" />
                    Son Alert'ler
                </h3>
                <div className="space-y-2 max-h-96 overflow-y-auto">
                    {idsAlerts.length > 0 ? (
                        idsAlerts.map((alert, idx) => (
                            <div key={idx} className={`p-3 rounded-lg border ${alert.severity === 'critical' ? 'bg-red-500/10 border-red-500/50' :
                                alert.severity === 'high' ? 'bg-orange-500/10 border-orange-500/50' :
                                    'bg-yellow-500/10 border-yellow-500/50'
                                }`}>
                                <div className="flex justify-between">
                                    <span className="font-medium text-white">{alert.attack_type}</span>
                                    <span className={`text-xs px-2 py-0.5 rounded ${SEVERITY_COLORS[alert.severity]}`}>
                                        {alert.severity}
                                    </span>
                                </div>
                                <p className="text-sm text-gray-400 mt-1">{alert.description}</p>
                                <p className="text-xs text-gray-500 mt-1">{alert.timestamp}</p>
                            </div>
                        ))
                    ) : (
                        <p className="text-gray-500 text-center py-4">Henüz alert yok</p>
                    )}
                </div>
            </div>
        </div>
    );

    // Models Tab
    const ModelsTab = () => (
        <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-5">
            <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                <Layers className="w-5 h-5 text-purple-400" />
                Eğitilmiş Modeller
            </h3>
            <div className="space-y-3">
                {trainedModels.length > 0 ? (
                    trainedModels.map((model, idx) => (
                        <div key={idx} className="bg-gray-900/50 rounded-lg p-4 flex items-center justify-between">
                            <div>
                                <h4 className="font-medium text-white">{model.name}</h4>
                                <p className="text-sm text-gray-400">{model.size_mb} MB • {model.created_at}</p>
                            </div>
                            <button
                                onClick={() => setIdsConfig({ ...idsConfig, model_path: model.path })}
                                className="px-3 py-1 bg-purple-600/30 text-purple-400 rounded-lg text-sm hover:bg-purple-600/50"
                            >
                                IDS'de Kullan
                            </button>
                        </div>
                    ))
                ) : (
                    <p className="text-gray-500 text-center py-4">Henüz eğitilmiş model yok</p>
                )}
            </div>
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
                    <h1 className="text-2xl font-bold text-white flex items-center gap-2">
                        <Target className="w-7 h-7 text-red-500" />
                        Saldırı Bazlı Model Eğitimi
                    </h1>
                    <p className="text-gray-400 text-sm">DDoS, DoS, Botnet, Probe ve daha fazlası</p>
                </div>
                <button
                    onClick={loadData}
                    className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-white"
                >
                    <RefreshCw className="w-4 h-4" />
                    Yenile
                </button>
            </div>

            {/* Tabs */}
            <div className="flex gap-2">
                {[
                    { id: 'train', label: '🎯 Eğitim', icon: Target },
                    { id: 'realtime', label: '📡 Real-time IDS', icon: Radio },
                    { id: 'models', label: '🧠 Modeller', icon: Layers },
                ].map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${activeTab === tab.id
                            ? 'bg-purple-600 text-white'
                            : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                            }`}
                    >
                        <tab.icon className="w-4 h-4" />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            {activeTab === 'train' && <TrainingTab />}
            {activeTab === 'realtime' && <RealTimeIDSTab />}
            {activeTab === 'models' && <ModelsTab />}
        </div>
    );
}
