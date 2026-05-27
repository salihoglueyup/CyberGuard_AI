/**
 * AI Hub - CyberGuard AI
 * ========================
 * 
 * Birleşik AI Merkezi - Tüm model ve AI özellikleri tek sayfada.
 * 
 * Sekmeler:
 * 1. Models - Model listesi, eğitim, karşılaştırma
 * 2. Intelligence - Zero-day, Meta-selector, RL
 * 3. Explainability - XAI, SHAP, Attention
 * 4. Reports - LLM raporlama
 * 5. Config - Tüm ayarlar
 */

import { useState, useEffect, useCallback } from 'react';
import {
    Brain, Zap, Shield, FileText, Settings, Activity, AlertTriangle,
    Eye, Target, Layers, Cpu, TrendingUp, BarChart3, RefreshCw,
    Play, Pause, CheckCircle, XCircle, Clock, Award, Search,
    Download, Upload, Trash2, GitCompare, Rocket, Archive,
    AlertCircle, Info, ChevronRight, Sparkles
} from 'lucide-react';
import {
    aiDecisionApi,
    modelsApi,
} from '../../services/api';
import { useToast } from '../../components/ui/Toast';

// Tab definitions
const TABS = [
    { id: 'models', label: '🔬 Models', icon: Brain, color: '#3B82F6' },
    { id: 'intelligence', label: '🧠 Intelligence', icon: Zap, color: '#8B5CF6' },
    { id: 'explainability', label: '📊 Explainability', icon: Eye, color: '#10B981' },
    { id: 'reports', label: '📝 Reports', icon: FileText, color: '#F59E0B' },
    { id: 'config', label: '⚙️ Config', icon: Settings, color: '#6B7280' },
];

// Attack types for selection
const ATTACK_TYPES = [
    'Normal', 'DDoS', 'DoS', 'Probe', 'PortScan',
    'BruteForce', 'WebAttack', 'Bot', 'Infiltration', 'ZERO_DAY'
];

export default function AIHub() {
    const [activeTab, setActiveTab] = useState('intelligence');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const toast = useToast();

    // Engine state
    const [engineHealth, setEngineHealth] = useState(null);
    const [engineStats, setEngineStats] = useState(null);

    // Intelligence state
    const [zeroDayStats, setZeroDayStats] = useState(null);
    const [thresholdStats, setThresholdStats] = useState(null);
    const [modelSelectStats, setModelSelectStats] = useState(null);

    // Explainability state
    const [featureInfo, setFeatureInfo] = useState(null);
    const [explanation, setExplanation] = useState(null);

    // Reports state
    const [reportHistory, setReportHistory] = useState([]);
    const [reportTemplates, setReportTemplates] = useState([]);
    const [generatedReport, setGeneratedReport] = useState(null);

    // Models state
    const [models, setModels] = useState([]);

    // Test input
    const [testInput, setTestInput] = useState({
        attackType: 'DDoS',
        confidence: 0.85,
        anomalyScore: 0.7,
    });

    // Load data on mount and tab change
    useEffect(() => {
        loadData();
    }, [activeTab, loadData]);

    const loadData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            // Always load engine health
            const healthRes = await aiDecisionApi.engine.health();
            setEngineHealth(healthRes.data);

            if (activeTab === 'intelligence') {
                const [zdRes, thRes, msRes] = await Promise.all([
                    aiDecisionApi.zeroDay.getStats().catch(() => ({ data: { success: false } })),
                    aiDecisionApi.threshold.getStats().catch(() => ({ data: { success: false } })),
                    aiDecisionApi.modelSelect.getStats().catch(() => ({ data: { success: false } })),
                ]);

                if (zdRes.data?.success) setZeroDayStats(zdRes.data.stats);
                if (thRes.data?.success) setThresholdStats(thRes.data.stats);
                if (msRes.data?.success) setModelSelectStats(msRes.data.stats);
            }

            if (activeTab === 'explainability') {
                const featRes = await aiDecisionApi.explain.getFeatures();
                if (featRes.data?.success) setFeatureInfo(featRes.data);
            }

            if (activeTab === 'reports') {
                const [histRes, templRes] = await Promise.all([
                    aiDecisionApi.report.getHistory(),
                    aiDecisionApi.report.getTemplates(),
                ]);

                if (histRes.data?.success) setReportHistory(histRes.data.history || []);
                if (templRes.data?.success) setReportTemplates(templRes.data.templates || []);
            }

            if (activeTab === 'models') {
                const modelsRes = await modelsApi.getAll().catch(() => ({ data: { models: [] } }));
                setModels(modelsRes.data?.models || []);
            }

            if (activeTab === 'config') {
                const statsRes = await aiDecisionApi.engine.getStats();
                if (statsRes.data?.success) setEngineStats(statsRes.data.stats);
            }
        } catch (err) {
            console.error('Load error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [activeTab]);

    // Initialize engine
    const handleInitialize = async () => {
        setLoading(true);
        try {
            await aiDecisionApi.engine.initialize();
            toast.success('Engine initialization started!');
            setTimeout(loadData, 2000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Train VAE
    const handleTrainVAE = async () => {
        setLoading(true);
        try {
            await aiDecisionApi.zeroDay.train(30, 3);
            toast.success('VAE training started!');
            setTimeout(loadData, 2000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Train RL
    const handleTrainRL = async () => {
        setLoading(true);
        try {
            await aiDecisionApi.threshold.train(100);
            toast.success('RL training started!');
            setTimeout(loadData, 2000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Get explanation
    const handleExplain = async () => {
        setLoading(true);
        try {
            // Generate dummy features for demo
            const features = Array(78).fill(0).map(() => Math.random());
            const res = await aiDecisionApi.explain.attack(features, testInput.attackType, 5);
            if (res.data?.success) {
                setExplanation(res.data.explanation);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Generate report
    const handleGenerateReport = async () => {
        setLoading(true);
        try {
            const res = await aiDecisionApi.report.generate(
                testInput.attackType,
                testInput.confidence,
                explanation,
                'attack_summary'
            );
            if (res.data?.success) {
                setGeneratedReport(res.data.report);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // RL Decision test
    const handleRLDecision = async () => {
        setLoading(true);
        try {
            const res = await aiDecisionApi.threshold.decide(
                testInput.confidence,
                testInput.anomalyScore
            );
            if (res.data?.success) {
                toast.info(`RL Decision: ${res.data.decision.action}`);
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    // Render functions for each tab
    const renderModelsTab = () => (
        <div className="space-y-6">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold">🔬 Model Manager</h2>
                <button
                    onClick={loadData}
                    className="btn-secondary flex items-center gap-2"
                >
                    <RefreshCw size={16} />
                    Refresh
                </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {models.map((model, idx) => (
                    <div key={idx} className="card p-4 hover:border-blue-500/50 transition-all">
                        <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 rounded-lg bg-blue-500/10">
                                <Cpu className="text-blue-400" size={20} />
                            </div>
                            <div>
                                <h3 className="font-semibold">{model.name || `Model ${idx + 1}`}</h3>
                                <span className="text-xs text-[var(--hud-text-muted)]">{model.type || 'Unknown'}</span>
                            </div>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-sm">
                            <div>
                                <span className="text-[var(--hud-text-muted)]">Accuracy:</span>
                                <span className="ml-2 text-green-400">{(model.accuracy * 100 || 95).toFixed(1)}%</span>
                            </div>
                            <div>
                                <span className="text-[var(--hud-text-muted)]">Status:</span>
                                <span className="ml-2 text-blue-400">{model.status || 'Active'}</span>
                            </div>
                        </div>
                    </div>
                ))}

                {models.length === 0 && (
                    <div className="col-span-3 text-center py-12 text-[var(--hud-text-muted)]">
                        No models loaded. Initialize the engine first.
                    </div>
                )}
            </div>
        </div>
    );

    const renderIntelligenceTab = () => (
        <div className="space-y-6">
            {/* Engine Status */}
            <div className="card p-4 border-purple-500/30">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <Sparkles className="text-purple-400" size={20} />
                        AI Engine Status
                    </h3>
                    <button
                        onClick={handleInitialize}
                        className="btn-primary flex items-center gap-2"
                        disabled={loading}
                    >
                        <Rocket size={16} />
                        Initialize
                    </button>
                </div>

                <div className="grid grid-cols-4 gap-4">
                    <div className="text-center p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className={`text-2xl mb-1 ${engineHealth?.status === 'healthy' ? 'text-green-400' : 'text-red-400'}`}>
                            {engineHealth?.status === 'healthy' ? '✅' : '❌'}
                        </div>
                        <div className="text-xs text-[var(--hud-text-muted)]">Engine</div>
                    </div>
                    <div className="text-center p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className={`text-2xl mb-1 ${engineHealth?.components?.zero_day ? 'text-green-400' : 'text-[var(--hud-amber)]'}`}>
                            {engineHealth?.components?.zero_day ? '✅' : '⏳'}
                        </div>
                        <div className="text-xs text-[var(--hud-text-muted)]">Zero-Day</div>
                    </div>
                    <div className="text-center p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className={`text-2xl mb-1 ${engineHealth?.components?.meta_selector ? 'text-green-400' : 'text-[var(--hud-amber)]'}`}>
                            {engineHealth?.components?.meta_selector ? '✅' : '⏳'}
                        </div>
                        <div className="text-xs text-[var(--hud-text-muted)]">Meta-Selector</div>
                    </div>
                    <div className="text-center p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className={`text-2xl mb-1 ${engineHealth?.components?.rl_agent ? 'text-green-400' : 'text-[var(--hud-amber)]'}`}>
                            {engineHealth?.components?.rl_agent ? '✅' : '⏳'}
                        </div>
                        <div className="text-xs text-[var(--hud-text-muted)]">RL Agent</div>
                    </div>
                </div>
            </div>

            {/* Zero-Day Detection */}
            <div className="card p-4 border-red-500/30">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-bold flex items-center gap-2">
                        <AlertTriangle className="text-red-400" size={20} />
                        Zero-Day Detection (VAE)
                    </h3>
                    <button
                        onClick={handleTrainVAE}
                        className="btn-secondary flex items-center gap-2"
                        disabled={loading}
                    >
                        <Play size={16} />
                        Train VAE
                    </button>
                </div>

                <div className="grid grid-cols-3 gap-4">
                    <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className="text-sm text-[var(--hud-text-muted)]">Status</div>
                        <div className={`text-lg font-bold ${zeroDayStats?.trained ? 'text-green-400' : 'text-[var(--hud-amber)]'}`}>
                            {zeroDayStats?.trained ? 'Trained' : 'Not Trained'}
                        </div>
                    </div>
                    <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className="text-sm text-[var(--hud-text-muted)]">Threshold</div>
                        <div className="text-lg font-bold text-blue-400">
                            {zeroDayStats?.threshold?.toFixed(4) || 'N/A'}
                        </div>
                    </div>
                    <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className="text-sm text-[var(--hud-text-muted)]">Sensitivity</div>
                        <div className="text-lg font-bold text-purple-400">
                            {zeroDayStats?.sensitivity || 3}/5
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
                    <div className="flex gap-2">
                        <button
                            onClick={handleTrainRL}
                            className="btn-secondary flex items-center gap-2"
                            disabled={loading}
                        >
                            <Play size={16} />
                            Train RL
                        </button>
                        <button
                            onClick={handleRLDecision}
                            className="btn-primary flex items-center gap-2"
                            disabled={loading}
                        >
                            <Zap size={16} />
                            Test Decision
                        </button>
                    </div>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <label className="text-sm text-[var(--hud-text-muted)]">Model Confidence</label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={testInput.confidence * 100}
                            onChange={(e) => setTestInput(prev => ({ ...prev, confidence: e.target.value / 100 }))}
                            className="w-full"
                        />
                        <span className="text-sm text-blue-400">{(testInput.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div>
                        <label className="text-sm text-[var(--hud-text-muted)]">Anomaly Score</label>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={testInput.anomalyScore * 100}
                            onChange={(e) => setTestInput(prev => ({ ...prev, anomalyScore: e.target.value / 100 }))}
                            className="w-full"
                        />
                        <span className="text-sm text-red-400">{(testInput.anomalyScore * 100).toFixed(0)}%</span>
                    </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                    <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className="text-sm text-[var(--hud-text-muted)]">Total Steps</div>
                        <div className="text-lg font-bold text-blue-400">
                            {thresholdStats?.total_steps || 0}
                        </div>
                    </div>
                    <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className="text-sm text-[var(--hud-text-muted)]">Episodes</div>
                        <div className="text-lg font-bold text-green-400">
                            {thresholdStats?.training_episodes || 0}
                        </div>
                    </div>
                    <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                        <div className="text-sm text-[var(--hud-text-muted)]">Epsilon</div>
                        <div className="text-lg font-bold text-[var(--hud-amber)]">
                            {thresholdStats?.current_epsilon?.toFixed(3) || 'N/A'}
                        </div>
                    </div>
                </div>
            </div>

            {/* Meta Model Selector */}
            <div className="card p-4 border-blue-500/30">
                <h3 className="text-lg font-bold flex items-center gap-2 mb-4">
                    <Layers className="text-blue-400" size={20} />
                    Meta-Learning Model Selector
                </h3>

                <div className="grid grid-cols-5 gap-2">
                    {['gru', 'lstm', 'bilstm', 'transformer', 'ensemble'].map(model => (
                        <div key={model} className="text-center p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                            <div className="text-xs text-[var(--hud-text-muted)] uppercase mb-1">{model}</div>
                            <div className="text-sm font-bold text-blue-400">
                                {modelSelectStats?.registered_models?.includes(model) ? '✅' : '⏳'}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );

    const renderExplainabilityTab = () => (
        <div className="space-y-6">
            <div className="flex gap-4 mb-4">
                <select
                    value={testInput.attackType}
                    onChange={(e) => setTestInput(prev => ({ ...prev, attackType: e.target.value }))}
                    className="input-field"
                >
                    {ATTACK_TYPES.map(type => (
                        <option key={type} value={type}>{type}</option>
                    ))}
                </select>
                <button
                    onClick={handleExplain}
                    className="btn-primary flex items-center gap-2"
                    disabled={loading}
                >
                    <Eye size={16} />
                    Explain Attack
                </button>
            </div>

            {explanation && (
                <div className="card p-4 border-green-500/30">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                        <Eye className="text-green-400" size={20} />
                        Attack Explanation: {explanation.attack_type}
                    </h3>

                    <p className="text-[var(--hud-text)] mb-4">{explanation.description}</p>

                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <h4 className="text-sm font-semibold text-[var(--hud-text-muted)] mb-2">Top Features</h4>
                            <div className="space-y-2">
                                {explanation.top_features?.map((f, i) => (
                                    <div key={i} className="flex justify-between items-center">
                                        <span className="text-sm">{f.name}</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-32 bg-[var(--hud-panel)] rounded-full h-2">
                                                <div
                                                    className="bg-blue-500 h-2 rounded-full"
                                                    style={{ width: `${f.importance * 100}%` }}
                                                />
                                            </div>
                                            <span className="text-xs text-[var(--hud-text-muted)]">
                                                {(f.importance * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        <div>
                            <h4 className="text-sm font-semibold text-[var(--hud-text-muted)] mb-2">Evidence</h4>
                            <ul className="space-y-1 text-sm">
                                {explanation.evidence?.map((e, i) => (
                                    <li key={i} className="flex items-center gap-2">
                                        <ChevronRight size={14} className="text-blue-400" />
                                        {e}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>

                    <div className="mt-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
                        <h4 className="text-sm font-semibold text-blue-400 mb-1">Recommendation</h4>
                        <p className="text-sm text-[var(--hud-text)]">{explanation.explanation}</p>
                    </div>
                </div>
            )}

            {featureInfo && (
                <div className="card p-4">
                    <h3 className="text-lg font-bold mb-4">Attack Patterns (Domain Knowledge)</h3>
                    <div className="grid grid-cols-2 gap-4">
                        {Object.entries(featureInfo.attack_patterns || {}).slice(0, 6).map(([type, info]) => (
                            <div key={type} className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                                <h4 className="font-semibold text-blue-400">{type}</h4>
                                <p className="text-sm text-[var(--hud-text-muted)]">{info.description}</p>
                                <div className="flex flex-wrap gap-1 mt-2">
                                    {info.indicators?.map((ind, i) => (
                                        <span key={i} className="text-xs px-2 py-1 bg-[var(--hud-panel)] rounded">
                                            {ind}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );

    const renderReportsTab = () => (
        <div className="space-y-6">
            <div className="flex gap-4 mb-4">
                <select
                    value={testInput.attackType}
                    onChange={(e) => setTestInput(prev => ({ ...prev, attackType: e.target.value }))}
                    className="input-field"
                >
                    {ATTACK_TYPES.map(type => (
                        <option key={type} value={type}>{type}</option>
                    ))}
                </select>
                <input
                    type="number"
                    min="0"
                    max="1"
                    step="0.01"
                    value={testInput.confidence}
                    onChange={(e) => setTestInput(prev => ({ ...prev, confidence: parseFloat(e.target.value) }))}
                    className="input-field w-24"
                    placeholder="Confidence"
                />
                <button
                    onClick={handleGenerateReport}
                    className="btn-primary flex items-center gap-2"
                    disabled={loading}
                >
                    <FileText size={16} />
                    Generate Report
                </button>
            </div>

            {generatedReport && (
                <div className="card p-4 border-yellow-500/30">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                        <FileText className="text-[var(--hud-amber)]" size={20} />
                        Generated Report
                    </h3>
                    <pre className="whitespace-pre-wrap text-sm bg-[var(--hud-bg)] p-4 rounded-lg overflow-auto max-h-96">
                        {generatedReport}
                    </pre>
                    <div className="flex gap-2 mt-4">
                        <button
                            onClick={() => navigator.clipboard.writeText(generatedReport)}
                            className="btn-secondary flex items-center gap-2"
                        >
                            <Download size={16} />
                            Copy to Clipboard
                        </button>
                    </div>
                </div>
            )}

            <div className="card p-4">
                <h3 className="text-lg font-bold mb-4">Report Templates</h3>
                <div className="grid grid-cols-3 gap-4">
                    {reportTemplates.map((template, i) => (
                        <div key={i} className="p-3 bg-[var(--hud-surface)]/50 rounded-lg text-center">
                            <FileText className="mx-auto mb-2 text-blue-400" size={24} />
                            <span className="text-sm">{template}</span>
                        </div>
                    ))}
                </div>
            </div>

            {reportHistory.length > 0 && (
                <div className="card p-4">
                    <h3 className="text-lg font-bold mb-4">Report History</h3>
                    <div className="space-y-2">
                        {reportHistory.map((report, i) => (
                            <div key={i} className="flex justify-between items-center p-2 bg-[var(--hud-surface)]/50 rounded">
                                <span>{report.attack_type}</span>
                                <span className="text-xs text-[var(--hud-text-muted)]">{report.timestamp}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );

    const renderConfigTab = () => (
        <div className="space-y-6">
            <div className="card p-4 border-purple-500/30">
                <h3 className="text-lg font-bold mb-4">Engine Configuration</h3>

                {engineStats && (
                    <div className="space-y-4">
                        <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                                <div className="text-sm text-[var(--hud-text-muted)]">Version</div>
                                <div className="text-lg font-bold text-blue-400">
                                    {engineStats.version || '1.0.0'}
                                </div>
                            </div>
                            <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg">
                                <div className="text-sm text-[var(--hud-text-muted)]">Initialized</div>
                                <div className="text-lg font-bold text-green-400">
                                    {engineStats.is_initialized ? 'Yes' : 'No'}
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-4 gap-4">
                            <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg text-center">
                                <div className="text-2xl font-bold text-blue-400">
                                    {engineStats.stats?.total_decisions || 0}
                                </div>
                                <div className="text-xs text-[var(--hud-text-muted)]">Total Decisions</div>
                            </div>
                            <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg text-center">
                                <div className="text-2xl font-bold text-red-400">
                                    {engineStats.stats?.zero_day_detections || 0}
                                </div>
                                <div className="text-xs text-[var(--hud-text-muted)]">Zero-Day</div>
                            </div>
                            <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg text-center">
                                <div className="text-2xl font-bold text-green-400">
                                    {engineStats.stats?.alerts_generated || 0}
                                </div>
                                <div className="text-xs text-[var(--hud-text-muted)]">Alerts</div>
                            </div>
                            <div className="p-3 bg-[var(--hud-surface)]/50 rounded-lg text-center">
                                <div className="text-2xl font-bold text-[var(--hud-text-muted)]">
                                    {engineStats.stats?.ignored || 0}
                                </div>
                                <div className="text-xs text-[var(--hud-text-muted)]">Ignored</div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] p-6">
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-xl font-semibold flex items-center gap-3 text-[var(--hud-text)]">
                    <Brain className="text-purple-400" />
                    AI Hub
                    <span className="text-[10px] font-normal text-[var(--hud-text-muted)] tracking-wide">
                        Intelligent Decision Center
                    </span>
                </h1>
            </div>

            {/* Error display */}
            {error && (
                <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex items-center gap-2">
                    <AlertCircle className="text-red-400" size={20} />
                    <span className="text-red-400">{error}</span>
                    <button onClick={() => setError(null)} className="ml-auto text-[var(--hud-text-muted)] hover:text-[var(--hud-text)]">
                        ✕
                    </button>
                </div>
            )}

            {/* Tabs */}
            <div className="flex gap-2 mb-6 border-b border-[var(--hud-border)] pb-4">
                {TABS.map(tab => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-all ${activeTab === tab.id
                            ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50'
                            : 'hover:bg-[var(--hud-surface)] text-[var(--hud-text-muted)]'
                            }`}
                    >
                        <tab.icon size={18} />
                        {tab.label}
                    </button>
                ))}

                <button
                    onClick={loadData}
                    className="ml-auto btn-secondary flex items-center gap-2"
                    disabled={loading}
                >
                    <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
                    Refresh
                </button>
            </div>

            {/* Loading */}
            {loading && (
                <div className="text-center py-12">
                    <RefreshCw size={32} className="animate-spin mx-auto text-purple-400" />
                    <p className="mt-2 text-[var(--hud-text-muted)]">Loading...</p>
                </div>
            )}

            {/* Tab content */}
            {!loading && (
                <div>
                    {activeTab === 'models' && renderModelsTab()}
                    {activeTab === 'intelligence' && renderIntelligenceTab()}
                    {activeTab === 'explainability' && renderExplainabilityTab()}
                    {activeTab === 'reports' && renderReportsTab()}
                    {activeTab === 'config' && renderConfigTab()}
                </div>
            )}
        </div>
    );
}
