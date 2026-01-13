/**
 * AI/ML Hub - CyberGuard AI
 * ==========================
 * 
 * Tüm AI ve ML özelliklerini tek sayfada birleştiren mega hub.
 * 
 * 12 Sekme:
 * 1. Dashboard - Genel bakış
 * 2. Models - Tüm modeller
 * 3. Training - Eğitim
 * 4. Compare - Karşılaştırma
 * 5. Ensemble - Ensemble builder
 * 6. Intelligence - Zero-day, RL, Meta
 * 7. Explainability - XAI, SHAP
 * 8. Reports - LLM raporlama
 * 9. AutoML - AutoML, Drift, Federated
 * 10. Playground - İnteraktif test
 * 11. Analytics - Performance metrics
 * 12. Settings - Gelişmiş ayarlar
 */

import { useState, useEffect, useCallback } from 'react';
import {
    Brain, Zap, Shield, FileText, Settings, Activity, AlertTriangle,
    Eye, Target, Layers, Cpu, TrendingUp, BarChart3, RefreshCw,
    Play, Pause, CheckCircle, XCircle, Clock, Award, Search,
    Download, Upload, Trash2, GitCompare, Rocket, Archive,
    AlertCircle, Info, ChevronRight, Sparkles, Dumbbell,
    LayoutDashboard, Beaker, LineChart, Gamepad2, Sliders
} from 'lucide-react';
import {
    aiDecisionApi,
    modelsApi,
    trainingApi
} from '../services/api';

// 12 Sekme Tanımları
const TABS = [
    { id: 'dashboard', label: '📊 Dashboard', icon: LayoutDashboard, color: '#3B82F6' },
    { id: 'models', label: '🔬 Models', icon: Brain, color: '#8B5CF6' },
    { id: 'training', label: '🎓 Training', icon: Dumbbell, color: '#10B981' },
    { id: 'compare', label: '📊 Compare', icon: GitCompare, color: '#F59E0B' },
    { id: 'ensemble', label: '🎯 Ensemble', icon: Layers, color: '#EC4899' },
    { id: 'intelligence', label: '🧠 Intelligence', icon: Zap, color: '#6366F1' },
    { id: 'explainability', label: '📊 XAI', icon: Eye, color: '#14B8A6' },
    { id: 'reports', label: '📝 Reports', icon: FileText, color: '#F97316' },
    { id: 'automl', label: '🔧 AutoML', icon: Beaker, color: '#84CC16' },
    { id: 'playground', label: '🎮 Playground', icon: Gamepad2, color: '#A855F7' },
    { id: 'analytics', label: '📈 Analytics', icon: LineChart, color: '#06B6D4' },
    { id: 'settings', label: '⚙️ Settings', icon: Sliders, color: '#6B7280' },
];

// Attack types
const ATTACK_TYPES = [
    'Normal', 'DDoS', 'DoS', 'Probe', 'PortScan',
    'BruteForce', 'WebAttack', 'Bot', 'Infiltration', 'ZERO_DAY'
];

// Model Types
const MODEL_TYPES = ['gru', 'lstm', 'bilstm', 'transformer', 'ensemble', 'ssa-lstm'];

export default function AIMLHub() {
    const [activeTab, setActiveTab] = useState('dashboard');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Global states
    const [engineHealth, setEngineHealth] = useState(null);
    const [models, setModels] = useState([]);
    const [trainings, setTrainings] = useState([]);

    // AI Decision states
    const [zeroDayStats, setZeroDayStats] = useState(null);
    const [thresholdStats, setThresholdStats] = useState(null);
    const [explanation, setExplanation] = useState(null);
    const [generatedReport, setGeneratedReport] = useState(null);

    // Analytics states
    const [systemMetrics, setSystemMetrics] = useState(null);
    const [modelPerformance, setModelPerformance] = useState(null);

    // Settings state
    const [settings, setSettings] = useState({
        vae: { beta: 4, latent_dim: 32, sensitivity: 3 },
        rl: { algorithm: 'Double DQN', epsilon: 0.1, gamma: 0.99 },
        xai: { shap_samples: 100, top_features: 5 },
        alerts: { auto_alert: true, threshold: 0.8 }
    });

    // Test inputs
    const [testInput, setTestInput] = useState({
        attackType: 'DDoS',
        confidence: 0.85,
        anomalyScore: 0.7,
    });

    // Load data on mount
    useEffect(() => {
        loadInitialData();
    }, []);

    // Load tab-specific data
    useEffect(() => {
        loadTabData();
    }, [activeTab]);

    const loadInitialData = async () => {
        try {
            const healthRes = await aiDecisionApi.engine.health().catch(() => null);
            if (healthRes?.data) setEngineHealth(healthRes.data);
        } catch (err) {
            console.error('Initial load error:', err);
        }
    };

    const loadTabData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            switch (activeTab) {
                case 'dashboard':
                    await loadDashboardData();
                    break;
                case 'models':
                case 'compare':
                    await loadModelsData();
                    break;
                case 'training':
                    await loadTrainingData();
                    break;
                case 'intelligence':
                    await loadIntelligenceData();
                    break;
                case 'analytics':
                    await loadAnalyticsData();
                    break;
                default:
                    break;
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [activeTab]);

    const loadDashboardData = async () => {
        const [healthRes, modelsRes] = await Promise.all([
            aiDecisionApi.engine.health().catch(() => ({ data: null })),
            modelsApi.getAll().catch(() => ({ data: { models: [] } })),
        ]);
        if (healthRes.data) setEngineHealth(healthRes.data);
        if (modelsRes.data?.models) setModels(modelsRes.data.models);
    };

    const loadModelsData = async () => {
        const res = await modelsApi.getAll().catch(() => ({ data: { models: [] } }));
        setModels(res.data?.models || []);
    };

    const loadTrainingData = async () => {
        const res = await trainingApi.getSessions().catch(() => ({ data: [] }));
        const data = res.data;
        // API farklı formatlar dönebilir
        if (Array.isArray(data)) {
            setTrainings(data);
        } else if (data?.sessions && Array.isArray(data.sessions)) {
            setTrainings(data.sessions);
        } else {
            setTrainings([]);
        }
    };

    const loadIntelligenceData = async () => {
        const [zdRes, thRes] = await Promise.all([
            aiDecisionApi.zeroDay.getStats().catch(() => ({ data: { success: false } })),
            aiDecisionApi.threshold.getStats().catch(() => ({ data: { success: false } })),
        ]);
        if (zdRes.data?.success) setZeroDayStats(zdRes.data.stats);
        if (thRes.data?.success) setThresholdStats(thRes.data.stats);
    };

    const loadAnalyticsData = async () => {
        try {
            const [sysRes, perfRes] = await Promise.all([
                fetch('/api/dashboard/system/metrics').then(r => r.json()).catch(() => ({ success: false })),
                fetch('/api/dashboard/model-performance').then(r => r.json()).catch(() => ({ success: false })),
            ]);
            if (sysRes.success) setSystemMetrics(sysRes.data);
            if (perfRes.success) setModelPerformance(perfRes.data);
        } catch (err) {
            console.error('Analytics load error:', err);
        }
    };

    // Action handlers
    const handleInitialize = async () => {
        setLoading(true);
        try {
            await aiDecisionApi.engine.initialize();
            alert('AI Engine initialization started!');
            setTimeout(loadInitialData, 2000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleTrainVAE = async () => {
        setLoading(true);
        try {
            await aiDecisionApi.zeroDay.train(30, settings.vae.sensitivity);
            alert('VAE training started!');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleTrainRL = async () => {
        setLoading(true);
        try {
            await aiDecisionApi.threshold.train(100);
            alert('RL training started!');
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleExplain = async () => {
        setLoading(true);
        try {
            const features = Array(78).fill(0).map(() => Math.random());
            const res = await aiDecisionApi.explain.attack(features, testInput.attackType, 5);
            if (res.data?.success) setExplanation(res.data.explanation);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const handleGenerateReport = async () => {
        setLoading(true);
        try {
            const res = await aiDecisionApi.report.generate(
                testInput.attackType, testInput.confidence, explanation, 'attack_summary'
            );
            if (res.data?.success) setGeneratedReport(res.data.report);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // ============= RENDER FUNCTIONS =============

    // Dashboard Tab
    const renderDashboard = () => (
        <div className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-4 gap-4">
                <StatCard
                    title="AI Engine"
                    value={engineHealth?.status === 'healthy' ? '✅ Active' : '❌ Inactive'}
                    color="blue"
                />
                <StatCard
                    title="Models"
                    value={models.length}
                    color="purple"
                />
                <StatCard
                    title="Zero-Day"
                    value={zeroDayStats?.trained ? 'Trained' : 'Not Trained'}
                    color="red"
                />
                <StatCard
                    title="RL Agent"
                    value={thresholdStats?.training_episodes || 0}
                    subtitle="episodes"
                    color="green"
                />
            </div>

            {/* Quick Actions */}
            <div className="card p-4">
                <h3 className="text-lg font-bold mb-4">⚡ Quick Actions</h3>
                <div className="flex gap-4">
                    <button onClick={handleInitialize} className="btn-primary">
                        <Rocket size={16} className="mr-2" /> Initialize Engine
                    </button>
                    <button onClick={handleTrainVAE} className="btn-secondary">
                        <Play size={16} className="mr-2" /> Train VAE
                    </button>
                    <button onClick={handleTrainRL} className="btn-secondary">
                        <Play size={16} className="mr-2" /> Train RL
                    </button>
                    <button onClick={loadTabData} className="btn-secondary">
                        <RefreshCw size={16} className="mr-2" /> Refresh
                    </button>
                </div>
            </div>

            {/* Component Status */}
            <div className="card p-4">
                <h3 className="text-lg font-bold mb-4">🧠 AI Components</h3>
                <div className="grid grid-cols-5 gap-4">
                    {[
                        { name: 'Zero-Day', active: zeroDayStats?.trained || engineHealth?.components?.zero_day },
                        { name: 'Meta-Selector', active: engineHealth?.components?.meta_selector ?? true },
                        { name: 'RL Agent', active: thresholdStats?.training_episodes > 0 || engineHealth?.components?.rl_threshold },
                        { name: 'XAI', active: engineHealth?.components?.explainer ?? true },
                        { name: 'LLM', active: engineHealth?.components?.llm_reporter ?? true },
                    ].map((comp, i) => (
                        <div key={i} className="text-center p-3 bg-gray-800/50 rounded-lg">
                            <div className={`text-2xl mb-1 ${comp.active ? 'text-green-400' : 'text-yellow-400'}`}>
                                {comp.active ? '✅' : '⏳'}
                            </div>
                            <div className="text-xs text-gray-400">{comp.name}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );

    // Models Tab
    const renderModels = () => (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">🔬 Model Manager</h2>
                <button onClick={loadModelsData} className="btn-secondary">
                    <RefreshCw size={16} className="mr-2" /> Refresh
                </button>
            </div>

            <div className="grid grid-cols-3 gap-4">
                {MODEL_TYPES.map((type) => (
                    <div key={type} className="card p-4 hover:border-blue-500/50 transition-all">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 rounded-lg bg-blue-500/10">
                                <Cpu className="text-blue-400" size={20} />
                            </div>
                            <div>
                                <h3 className="font-semibold uppercase">{type}</h3>
                                <span className="text-xs text-gray-400">IDS Model</span>
                            </div>
                        </div>
                        <div className="flex gap-2 mt-4">
                            <button className="btn-sm btn-primary flex-1">Deploy</button>
                            <button className="btn-sm btn-secondary flex-1">Train</button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );

    // Training Tab
    const renderTraining = () => (
        <div className="space-y-6">
            <h2 className="text-xl font-bold">🎓 Training Center</h2>

            <div className="grid grid-cols-2 gap-6">
                {/* Training Form */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">New Training</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Model Type</label>
                            <select className="input-field w-full">
                                {MODEL_TYPES.map(t => <option key={t} value={t}>{t.toUpperCase()}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Epochs</label>
                            <input type="number" className="input-field w-full" defaultValue={50} />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Batch Size</label>
                            <input type="number" className="input-field w-full" defaultValue={32} />
                        </div>
                        <button className="btn-primary w-full">
                            <Play size={16} className="mr-2" /> Start Training
                        </button>
                    </div>
                </div>

                {/* Active Trainings */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Active Trainings</h3>
                    {trainings.length === 0 ? (
                        <div className="text-center py-8 text-gray-400">
                            No active trainings
                        </div>
                    ) : (
                        <div className="space-y-2">
                            {trainings.map((t, i) => (
                                <div key={i} className="p-3 bg-gray-800/50 rounded-lg flex justify-between">
                                    <span>{t.model_type}</span>
                                    <span className="text-blue-400">{t.progress}%</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

    // Intelligence Tab
    const renderIntelligence = () => (
        <div className="space-y-6">
            {/* Engine Status */}
            <div className="card p-4 border-purple-500/30">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <Sparkles className="text-purple-400" size={20} />
                        AI Engine
                    </h3>
                    <button onClick={handleInitialize} className="btn-primary" disabled={loading}>
                        <Rocket size={16} className="mr-2" /> Initialize
                    </button>
                </div>

                <div className="grid grid-cols-4 gap-4">
                    {['Engine', 'Zero-Day', 'Meta', 'RL'].map((name, i) => (
                        <div key={i} className="text-center p-3 bg-gray-800/50 rounded-lg">
                            <div className={`text-2xl mb-1 ${engineHealth?.status === 'healthy' ? 'text-green-400' : 'text-red-400'}`}>
                                ✅
                            </div>
                            <div className="text-xs text-gray-400">{name}</div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Zero-Day */}
            <div className="card p-4 border-red-500/30">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <AlertTriangle className="text-red-400" size={20} />
                        Zero-Day Detection (VAE)
                    </h3>
                    <button onClick={handleTrainVAE} className="btn-secondary" disabled={loading}>
                        <Play size={16} className="mr-2" /> Train
                    </button>
                </div>

                <div className="grid grid-cols-3 gap-4">
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-sm text-gray-400">Status</div>
                        <div className={`text-lg font-bold ${zeroDayStats?.trained ? 'text-green-400' : 'text-yellow-400'}`}>
                            {zeroDayStats?.trained ? 'Trained' : 'Not Trained'}
                        </div>
                    </div>
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-sm text-gray-400">Threshold</div>
                        <div className="text-lg font-bold text-blue-400">
                            {zeroDayStats?.threshold?.toFixed(4) || 'N/A'}
                        </div>
                    </div>
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-sm text-gray-400">β Value</div>
                        <div className="text-lg font-bold text-purple-400">
                            {settings.vae.beta}
                        </div>
                    </div>
                </div>
            </div>

            {/* RL Threshold */}
            <div className="card p-4 border-green-500/30">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <Target className="text-green-400" size={20} />
                        RL Threshold Optimizer
                    </h3>
                    <button onClick={handleTrainRL} className="btn-secondary" disabled={loading}>
                        <Play size={16} className="mr-2" /> Train
                    </button>
                </div>

                <div className="grid grid-cols-3 gap-4">
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-sm text-gray-400">Algorithm</div>
                        <div className="text-lg font-bold text-blue-400">{settings.rl.algorithm}</div>
                    </div>
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-sm text-gray-400">Episodes</div>
                        <div className="text-lg font-bold text-green-400">
                            {thresholdStats?.training_episodes || 0}
                        </div>
                    </div>
                    <div className="p-3 bg-gray-800/50 rounded-lg">
                        <div className="text-sm text-gray-400">Epsilon</div>
                        <div className="text-lg font-bold text-yellow-400">
                            {thresholdStats?.current_epsilon?.toFixed(3) || settings.rl.epsilon}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );

    // Settings Tab
    const renderSettings = () => (
        <div className="space-y-6">
            <h2 className="text-xl font-bold">⚙️ Advanced Settings</h2>

            <div className="grid grid-cols-2 gap-6">
                {/* VAE Settings */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 text-red-400">🔴 VAE (Zero-Day)</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">β Value</label>
                            <input
                                type="range" min="1" max="10" step="0.5"
                                value={settings.vae.beta}
                                onChange={(e) => setSettings(s => ({ ...s, vae: { ...s.vae, beta: parseFloat(e.target.value) } }))}
                                className="w-full"
                            />
                            <span className="text-sm text-blue-400">{settings.vae.beta}</span>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Latent Dimension</label>
                            <select
                                className="input-field w-full"
                                value={settings.vae.latent_dim}
                                onChange={(e) => setSettings(s => ({ ...s, vae: { ...s.vae, latent_dim: parseInt(e.target.value) } }))}
                            >
                                {[16, 32, 64, 128].map(v => <option key={v} value={v}>{v}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Sensitivity (1-5)</label>
                            <input
                                type="range" min="1" max="5"
                                value={settings.vae.sensitivity}
                                onChange={(e) => setSettings(s => ({ ...s, vae: { ...s.vae, sensitivity: parseInt(e.target.value) } }))}
                                className="w-full"
                            />
                            <span className="text-sm text-purple-400">{settings.vae.sensitivity}</span>
                        </div>
                    </div>
                </div>

                {/* RL Settings */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 text-green-400">🟢 RL (Threshold)</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Algorithm</label>
                            <select
                                className="input-field w-full"
                                value={settings.rl.algorithm}
                                onChange={(e) => setSettings(s => ({ ...s, rl: { ...s.rl, algorithm: e.target.value } }))}
                            >
                                {['DQN', 'Double DQN', 'Dueling DQN'].map(v => <option key={v} value={v}>{v}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Epsilon</label>
                            <input
                                type="range" min="0.01" max="1" step="0.01"
                                value={settings.rl.epsilon}
                                onChange={(e) => setSettings(s => ({ ...s, rl: { ...s.rl, epsilon: parseFloat(e.target.value) } }))}
                                className="w-full"
                            />
                            <span className="text-sm text-blue-400">{settings.rl.epsilon}</span>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Gamma</label>
                            <input
                                type="range" min="0.9" max="0.999" step="0.001"
                                value={settings.rl.gamma}
                                onChange={(e) => setSettings(s => ({ ...s, rl: { ...s.rl, gamma: parseFloat(e.target.value) } }))}
                                className="w-full"
                            />
                            <span className="text-sm text-green-400">{settings.rl.gamma}</span>
                        </div>
                    </div>
                </div>

                {/* XAI Settings */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 text-blue-400">🔵 XAI</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">SHAP Samples</label>
                            <input
                                type="range" min="10" max="1000" step="10"
                                value={settings.xai.shap_samples}
                                onChange={(e) => setSettings(s => ({ ...s, xai: { ...s.xai, shap_samples: parseInt(e.target.value) } }))}
                                className="w-full"
                            />
                            <span className="text-sm text-blue-400">{settings.xai.shap_samples}</span>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Top Features</label>
                            <input
                                type="range" min="3" max="20"
                                value={settings.xai.top_features}
                                onChange={(e) => setSettings(s => ({ ...s, xai: { ...s.xai, top_features: parseInt(e.target.value) } }))}
                                className="w-full"
                            />
                            <span className="text-sm text-purple-400">{settings.xai.top_features}</span>
                        </div>
                    </div>
                </div>

                {/* Alert Settings */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 text-yellow-400">🟡 Alerts</h3>
                    <div className="space-y-4">
                        <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-400">Auto Alert</span>
                            <button
                                className={`w-12 h-6 rounded-full transition-colors ${settings.alerts.auto_alert ? 'bg-green-500' : 'bg-gray-600'}`}
                                onClick={() => setSettings(s => ({ ...s, alerts: { ...s.alerts, auto_alert: !s.alerts.auto_alert } }))}
                            >
                                <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${settings.alerts.auto_alert ? 'translate-x-6' : 'translate-x-0.5'}`} />
                            </button>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Threshold</label>
                            <input
                                type="range" min="0.5" max="0.99" step="0.01"
                                value={settings.alerts.threshold}
                                onChange={(e) => setSettings(s => ({ ...s, alerts: { ...s.alerts, threshold: parseFloat(e.target.value) } }))}
                                className="w-full"
                            />
                            <span className="text-sm text-yellow-400">{settings.alerts.threshold}</span>
                        </div>
                    </div>
                </div>
            </div>

            <button className="btn-primary">
                <CheckCircle size={16} className="mr-2" /> Save Settings
            </button>
        </div>
    );

    // Explainability Tab
    const renderExplainability = () => (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">📊 XAI - Explainable AI</h2>
                <button onClick={handleExplain} className="btn-primary" disabled={loading}>
                    <Eye size={16} className="mr-2" /> Generate Explanation
                </button>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* Test Input */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Test Attack</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Attack Type</label>
                            <select
                                className="input-field w-full"
                                value={testInput.attackType}
                                onChange={(e) => setTestInput(s => ({ ...s, attackType: e.target.value }))}
                            >
                                {ATTACK_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Confidence</label>
                            <input
                                type="range" min="0" max="1" step="0.01"
                                value={testInput.confidence}
                                onChange={(e) => setTestInput(s => ({ ...s, confidence: parseFloat(e.target.value) }))}
                                className="w-full"
                            />
                            <span className="text-sm text-blue-400">{testInput.confidence.toFixed(2)}</span>
                        </div>
                    </div>
                </div>

                {/* Explanation Result */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Explanation</h3>
                    {explanation ? (
                        <div className="space-y-3">
                            <div className="text-sm text-gray-400">Top Features:</div>
                            {explanation.top_features?.map((f, i) => (
                                <div key={i} className="flex justify-between p-2 bg-gray-800/50 rounded">
                                    <span>{f.feature}</span>
                                    <span className={f.importance > 0 ? 'text-green-400' : 'text-red-400'}>
                                        {f.importance?.toFixed(4)}
                                    </span>
                                </div>
                            )) || <div className="text-gray-500">No features available</div>}
                        </div>
                    ) : (
                        <div className="text-center py-8 text-gray-400">
                            Click "Generate Explanation" to analyze
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

    // Reports Tab
    const renderReports = () => (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">📝 LLM Reports</h2>
                <button onClick={handleGenerateReport} className="btn-primary" disabled={loading}>
                    <FileText size={16} className="mr-2" /> Generate Report
                </button>
            </div>

            <div className="grid grid-cols-2 gap-6">
                {/* Report Settings */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Report Configuration</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Attack Type</label>
                            <select className="input-field w-full" value={testInput.attackType}
                                onChange={(e) => setTestInput(s => ({ ...s, attackType: e.target.value }))}>
                                {ATTACK_TYPES.map(t => <option key={t} value={t}>{t}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Template</label>
                            <select className="input-field w-full">
                                <option value="attack_summary">Attack Summary</option>
                                <option value="incident_report">Incident Report</option>
                                <option value="technical_analysis">Technical Analysis</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Language</label>
                            <select className="input-field w-full">
                                <option value="en">English</option>
                                <option value="tr">Türkçe</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Generated Report */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Generated Report</h3>
                    {generatedReport ? (
                        <div className="prose prose-invert max-w-none">
                            <pre className="whitespace-pre-wrap text-sm bg-gray-800/50 p-4 rounded-lg">
                                {generatedReport.content || JSON.stringify(generatedReport, null, 2)}
                            </pre>
                        </div>
                    ) : (
                        <div className="text-center py-8 text-gray-400">
                            Click "Generate Report" to create a report
                        </div>
                    )}
                </div>
            </div>
        </div>
    );

    // Playground Tab
    const renderPlayground = () => (
        <div className="space-y-6">
            <h2 className="text-xl font-bold">🎮 AI Playground</h2>

            <div className="grid grid-cols-2 gap-6">
                {/* Input */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">📥 Input</h3>
                    <div className="space-y-4">
                        <textarea
                            className="input-field w-full h-32"
                            placeholder="Paste traffic sample or use buttons below..."
                        />
                        <div className="flex gap-2">
                            <button className="btn-secondary flex-1">🎲 Random Sample</button>
                            <button className="btn-secondary flex-1">📁 Upload CSV</button>
                        </div>
                        <button className="btn-primary w-full">
                            <Rocket size={16} className="mr-2" /> Analyze
                        </button>
                    </div>
                </div>

                {/* Output */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">📤 Output</h3>
                    <div className="space-y-4">
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="text-sm text-gray-400">Prediction</div>
                            <div className="text-xl font-bold text-green-400">Normal (95.2%)</div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="text-sm text-gray-400">Zero-Day</div>
                            <div className="text-xl font-bold text-blue-400">Not Detected</div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="text-sm text-gray-400">RL Decision</div>
                            <div className="text-xl font-bold text-yellow-400">IGNORE</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );

    // Analytics Tab
    const renderAnalytics = () => (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">📈 Performance Analytics</h2>
                <button onClick={loadAnalyticsData} className="btn-secondary">
                    <RefreshCw size={16} className="mr-2" /> Refresh
                </button>
            </div>

            <div className="grid grid-cols-4 gap-4">
                <StatCard
                    title="Accuracy"
                    value={modelPerformance?.avg_accuracy ? `${(modelPerformance.avg_accuracy * 100).toFixed(1)}%` : 'N/A'}
                    color="green"
                />
                <StatCard
                    title="F1 Score"
                    value={modelPerformance?.avg_f1?.toFixed(2) || 'N/A'}
                    color="blue"
                />
                <StatCard
                    title="Total Models"
                    value={modelPerformance?.total_models || 0}
                    color="purple"
                />
                <StatCard
                    title="Best Model"
                    value={modelPerformance?.best_model || 'N/A'}
                    color="red"
                />
            </div>

            <div className="grid grid-cols-2 gap-6">
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Model Metrics</h3>
                    <div className="space-y-2">
                        {[
                            { label: 'Avg Accuracy', value: modelPerformance?.avg_accuracy, format: (v) => `${(v * 100).toFixed(1)}%` },
                            { label: 'Avg F1 Score', value: modelPerformance?.avg_f1, format: (v) => v?.toFixed(4) },
                            { label: 'Avg Precision', value: modelPerformance?.avg_precision, format: (v) => v?.toFixed(4) },
                            { label: 'Avg Recall', value: modelPerformance?.avg_recall, format: (v) => v?.toFixed(4) },
                        ].map((m, i) => (
                            <div key={i} className="flex justify-between p-2 bg-gray-800/50 rounded">
                                <span>{m.label}</span>
                                <span className="text-green-400">{m.value ? m.format(m.value) : 'N/A'}</span>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">System Resources (Live)</h3>
                    <div className="space-y-3">
                        {[
                            { label: 'CPU', value: systemMetrics?.cpu?.percent || 0, color: 'blue' },
                            { label: 'Memory', value: systemMetrics?.memory?.percent || 0, color: 'purple' },
                            { label: 'GPU', value: systemMetrics?.gpu?.percent || 0, color: 'green' },
                        ].map((r, i) => (
                            <div key={i}>
                                <div className="flex justify-between text-sm mb-1">
                                    <span className="text-gray-400">{r.label}</span>
                                    <span>{r.value.toFixed(1)}%</span>
                                </div>
                                <div className="w-full h-2 bg-gray-700 rounded">
                                    <div
                                        className={`h-full rounded bg-${r.color}-500`}
                                        style={{ width: `${r.value}%` }}
                                    />
                                </div>
                            </div>
                        ))}
                        {systemMetrics?.memory && (
                            <div className="text-xs text-gray-500 mt-2">
                                Memory: {systemMetrics.memory.used_gb}GB / {systemMetrics.memory.total_gb}GB
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );

    // Compare Tab
    const renderCompare = () => (
        <div className="space-y-6">
            <h2 className="text-xl font-bold">📊 Model Comparison</h2>

            {/* Model Selection */}
            <div className="card p-4">
                <h3 className="text-lg font-semibold mb-4">Select Models to Compare</h3>
                <div className="grid grid-cols-6 gap-3">
                    {MODEL_TYPES.map((type) => (
                        <label key={type} className="flex items-center gap-2 p-2 bg-gray-800/50 rounded-lg cursor-pointer hover:bg-gray-700/50">
                            <input type="checkbox" className="w-4 h-4" defaultChecked={['lstm', 'transformer'].includes(type)} />
                            <span className="text-sm uppercase">{type}</span>
                        </label>
                    ))}
                </div>
            </div>

            {/* Comparison Table */}
            <div className="card p-4">
                <h3 className="text-lg font-semibold mb-4">Performance Metrics</h3>
                {models.length > 0 ? (
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead>
                                <tr className="border-b border-gray-700">
                                    <th className="text-left p-2">Model</th>
                                    <th className="text-right p-2">Accuracy</th>
                                    <th className="text-right p-2">F1</th>
                                    <th className="text-right p-2">Precision</th>
                                    <th className="text-right p-2">Recall</th>
                                    <th className="text-right p-2">Samples</th>
                                </tr>
                            </thead>
                            <tbody>
                                {models.map((m, i) => (
                                    <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/30">
                                        <td className="p-2 font-medium">{m.name || m.model_type}</td>
                                        <td className="p-2 text-right text-green-400">
                                            {(m.accuracy * 100).toFixed(1)}%
                                        </td>
                                        <td className="p-2 text-right text-blue-400">
                                            {m.f1_score?.toFixed(2) || 'N/A'}
                                        </td>
                                        <td className="p-2 text-right text-purple-400">
                                            {m.precision?.toFixed(2) || 'N/A'}
                                        </td>
                                        <td className="p-2 text-right text-yellow-400">
                                            {m.recall?.toFixed(2) || 'N/A'}
                                        </td>
                                        <td className="p-2 text-right text-gray-400">
                                            {m.train_samples?.toLocaleString() || 'N/A'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                ) : (
                    <div className="text-center py-8 text-gray-400">
                        <p>No models found. Train some models first!</p>
                        <p className="text-xs mt-2">Model metrics will appear here after training.</p>
                    </div>
                )}
            </div>

            {/* Radar Chart Placeholder */}
            <div className="card p-4">
                <h3 className="text-lg font-semibold mb-4">Radar Comparison</h3>
                <div className="h-64 flex items-center justify-center bg-gray-800/30 rounded-lg">
                    <div className="text-center text-gray-400">
                        <BarChart3 size={48} className="mx-auto mb-2 opacity-50" />
                        <p>Radar chart visualization</p>
                    </div>
                </div>
            </div>
        </div>
    );

    // Ensemble Tab
    const renderEnsemble = () => (
        <div className="space-y-6">
            <h2 className="text-xl font-bold">🎯 Ensemble Builder</h2>

            <div className="grid grid-cols-2 gap-6">
                {/* Model Selection */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Select Base Models</h3>
                    <div className="space-y-3">
                        {MODEL_TYPES.filter(t => t !== 'ensemble').map((type) => (
                            <div key={type} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                                <label className="flex items-center gap-3">
                                    <input type="checkbox" className="w-4 h-4" defaultChecked />
                                    <span className="uppercase font-medium">{type}</span>
                                </label>
                                <div className="flex items-center gap-2">
                                    <span className="text-sm text-gray-400">Weight:</span>
                                    <input
                                        type="range" min="0" max="1" step="0.1" defaultValue="0.2"
                                        className="w-20"
                                    />
                                    <span className="text-sm text-blue-400 w-8">0.2</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Ensemble Config */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4">Ensemble Configuration</h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Voting Strategy</label>
                            <select className="input-field w-full">
                                <option value="soft">Soft Voting (Average)</option>
                                <option value="hard">Hard Voting (Majority)</option>
                                <option value="weighted">Weighted Voting</option>
                                <option value="stacking">Stacking</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Meta Learner</label>
                            <select className="input-field w-full">
                                <option value="none">None (Simple Voting)</option>
                                <option value="lr">Logistic Regression</option>
                                <option value="rf">Random Forest</option>
                                <option value="mlp">MLP</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Confidence Threshold</label>
                            <input type="range" min="0.5" max="0.99" step="0.01" defaultValue="0.8" className="w-full" />
                        </div>
                        <button className="btn-primary w-full">
                            <Layers size={16} className="mr-2" /> Create Ensemble
                        </button>
                    </div>
                </div>
            </div>

            {/* Ensemble Performance */}
            <div className="card p-4">
                <h3 className="text-lg font-semibold mb-4">Created Ensembles</h3>
                <div className="grid grid-cols-3 gap-4">
                    {[
                        { name: 'Ensemble-v1', models: 5, acc: 99.1 },
                        { name: 'Ensemble-v2', models: 3, acc: 98.5 },
                    ].map((e, i) => (
                        <div key={i} className="p-4 bg-gray-800/50 rounded-lg">
                            <h4 className="font-semibold">{e.name}</h4>
                            <div className="text-sm text-gray-400 mt-1">{e.models} models</div>
                            <div className="text-2xl font-bold text-green-400 mt-2">{e.acc}%</div>
                            <div className="flex gap-2 mt-3">
                                <button className="btn-sm btn-primary flex-1">Deploy</button>
                                <button className="btn-sm btn-secondary">Edit</button>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );

    // AutoML Tab
    const renderAutoML = () => (
        <div className="space-y-6">
            <h2 className="text-xl font-bold">🔧 AutoML & Advanced Features</h2>

            <div className="grid grid-cols-2 gap-6">
                {/* AutoML Search */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <Beaker className="text-green-400" size={20} />
                        AutoML Search
                    </h3>
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Search Space</label>
                            <select className="input-field w-full">
                                <option value="small">Small (Fast)</option>
                                <option value="medium">Medium (Balanced)</option>
                                <option value="large">Large (Thorough)</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Time Budget (minutes)</label>
                            <input type="number" className="input-field w-full" defaultValue={30} />
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Optimization Metric</label>
                            <select className="input-field w-full">
                                <option value="f1">F1 Score</option>
                                <option value="accuracy">Accuracy</option>
                                <option value="auc">AUC-ROC</option>
                            </select>
                        </div>
                        <button className="btn-primary w-full">
                            <Search size={16} className="mr-2" /> Start AutoML Search
                        </button>
                    </div>
                </div>

                {/* Model Drift */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <TrendingUp className="text-yellow-400" size={20} />
                        Model Drift Detection
                    </h3>
                    <div className="space-y-4">
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Data Drift</span>
                                <span className="text-green-400 font-bold">Low (0.02)</span>
                            </div>
                            <div className="w-full h-2 bg-gray-700 rounded mt-2">
                                <div className="h-full w-[8%] bg-green-500 rounded" />
                            </div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Concept Drift</span>
                                <span className="text-yellow-400 font-bold">Medium (0.15)</span>
                            </div>
                            <div className="w-full h-2 bg-gray-700 rounded mt-2">
                                <div className="h-full w-[30%] bg-yellow-500 rounded" />
                            </div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="flex justify-between items-center">
                                <span className="text-gray-400">Performance Drift</span>
                                <span className="text-green-400 font-bold">Stable</span>
                            </div>
                            <div className="w-full h-2 bg-gray-700 rounded mt-2">
                                <div className="h-full w-[5%] bg-green-500 rounded" />
                            </div>
                        </div>
                        <button className="btn-secondary w-full">
                            <RefreshCw size={16} className="mr-2" /> Run Drift Analysis
                        </button>
                    </div>
                </div>

                {/* A/B Testing */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <GitCompare className="text-blue-400" size={20} />
                        A/B Testing
                    </h3>
                    <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Model A</label>
                                <select className="input-field w-full">
                                    {MODEL_TYPES.map(t => <option key={t} value={t}>{t.toUpperCase()}</option>)}
                                </select>
                            </div>
                            <div>
                                <label className="block text-sm text-gray-400 mb-1">Model B</label>
                                <select className="input-field w-full" defaultValue="transformer">
                                    {MODEL_TYPES.map(t => <option key={t} value={t}>{t.toUpperCase()}</option>)}
                                </select>
                            </div>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Traffic Split</label>
                            <input type="range" min="10" max="90" defaultValue="50" className="w-full" />
                            <div className="flex justify-between text-xs text-gray-400">
                                <span>A: 50%</span>
                                <span>B: 50%</span>
                            </div>
                        </div>
                        <button className="btn-primary w-full">
                            <Play size={16} className="mr-2" /> Start A/B Test
                        </button>
                    </div>
                </div>

                {/* Federated Learning */}
                <div className="card p-4">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <Cpu className="text-purple-400" size={20} />
                        Federated Learning
                    </h3>
                    <div className="space-y-4">
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Connected Nodes</span>
                                <span className="text-green-400 font-bold">5 / 8</span>
                            </div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Global Rounds</span>
                                <span className="text-blue-400 font-bold">12</span>
                            </div>
                        </div>
                        <div className="p-3 bg-gray-800/50 rounded-lg">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Aggregation</span>
                                <span className="text-purple-400">FedAvg</span>
                            </div>
                        </div>
                        <button className="btn-secondary w-full">
                            <Activity size={16} className="mr-2" /> Sync Models
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );

    // Tab router
    const renderTabContent = () => {
        switch (activeTab) {
            case 'dashboard': return renderDashboard();
            case 'models': return renderModels();
            case 'training': return renderTraining();
            case 'intelligence': return renderIntelligence();
            case 'settings': return renderSettings();
            case 'explainability': return renderExplainability();
            case 'reports': return renderReports();
            case 'playground': return renderPlayground();
            case 'analytics': return renderAnalytics();
            case 'compare': return renderCompare();
            case 'ensemble': return renderEnsemble();
            case 'automl': return renderAutoML();
            default: return renderDashboard();
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-3xl font-bold flex items-center gap-3">
                    <Brain className="text-purple-400" />
                    AI/ML Hub
                    <span className="text-sm font-normal text-gray-400 ml-2">
                        Intelligent Control Center
                    </span>
                </h1>
            </div>

            {/* Error */}
            {error && (
                <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2">
                    <AlertCircle className="text-red-400" size={20} />
                    <span className="text-red-400">{error}</span>
                    <button onClick={() => setError(null)} className="ml-auto">✕</button>
                </div>
            )}

            {/* Tabs */}
            <div className="flex flex-wrap gap-2 mb-6 border-b border-gray-700 pb-4">
                {TABS.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-3 py-2 rounded-lg flex items-center gap-2 text-sm transition-all ${activeTab === tab.id
                            ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50'
                            : 'hover:bg-gray-800 text-gray-400'
                            }`}
                    >
                        <tab.icon size={16} />
                        {tab.label}
                    </button>
                ))}
            </div>

            {/* Content */}
            {loading ? (
                <div className="text-center py-12">
                    <RefreshCw size={32} className="animate-spin mx-auto text-purple-400" />
                    <p className="mt-2 text-gray-400">Loading...</p>
                </div>
            ) : (
                renderTabContent()
            )}
        </div>
    );
}

// Helper component
function StatCard({ title, value, subtitle, color }) {
    const colors = {
        blue: 'border-blue-500/30 bg-blue-500/5',
        purple: 'border-purple-500/30 bg-purple-500/5',
        red: 'border-red-500/30 bg-red-500/5',
        green: 'border-green-500/30 bg-green-500/5',
    };

    return (
        <div className={`card p-4 ${colors[color]}`}>
            <div className="text-sm text-gray-400">{title}</div>
            <div className="text-2xl font-bold mt-1">{value}</div>
            {subtitle && <div className="text-xs text-gray-500">{subtitle}</div>}
        </div>
    );
}
