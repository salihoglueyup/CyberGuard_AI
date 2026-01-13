import axios from 'axios';

// Full URL kullan - proxy bypass
const API_BASE = 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_BASE,
    headers: {
        'Content-Type': 'application/json'
    }
});

// Dashboard API
export const dashboardApi = {
    getStats: (hours = 24) => api.get(`/dashboard/stats?hours=${hours}`),
    getSummary: () => api.get('/dashboard/summary'),
    getHourlyTrend: (hours = 24) => api.get(`/dashboard/hourly-trend?hours=${hours}`),
    getRecentAttacks: (limit = 10) => api.get(`/dashboard/recent-attacks?limit=${limit}`)
};

// Attacks API
export const attacksApi = {
    getAll: (page = 1, limit = 20, hours = null) =>
        api.get(`/attacks?page=${page}&limit=${limit}${hours ? `&hours=${hours}` : ''}`),
    getStats: (hours = null) => api.get(`/attacks/stats${hours ? `?hours=${hours}` : ''}`),
    getByType: (hours = null) => api.get(`/attacks/by-type${hours ? `?hours=${hours}` : ''}`),
    getBySeverity: (hours = null) => api.get(`/attacks/by-severity${hours ? `?hours=${hours}` : ''}`),
    getTopIps: (limit = 10, hours = null) => api.get(`/attacks/top-ips?limit=${limit}${hours ? `&hours=${hours}` : ''}`),
    getTimeline: (hours = null) => api.get(`/attacks/timeline${hours ? `?hours=${hours}` : ''}`),
    searchByIp: (ip, limit = 50) => api.get(`/attacks/search/${ip}?limit=${limit}`)
};

// Models API
export const modelsApi = {
    getAll: () => api.get('/models'),
    getDeployed: () => api.get('/models/deployed'),
    getStats: () => api.get('/models/stats'),
    compare: () => api.get('/models/compare/all'),
    getById: (modelId) => api.get(`/models/${modelId}`),
    predict: (modelId, features) => api.post(`/models/${modelId}/predict`, features),
    predictAuto: (features) => api.post('/models/predict', features),
    delete: (modelId) => api.delete(`/models/${modelId}`),
    deploy: (modelId) => api.post(`/models/${modelId}/deploy`),
    archive: (modelId) => api.post(`/models/${modelId}/archive`)
};

// Chat API
export const chatApi = {
    send: (message, modelId = null, provider = 'groq') =>
        api.post('/chat', { message, model_id: modelId, provider }),
    getHistory: (limit = 20) => api.get(`/chat/history?limit=${limit}`),
    getProviders: () => api.get('/chat/providers'),
    clear: () => api.post('/chat/clear'),
    getStats: () => api.get('/chat/stats'),
    // Streaming chat - SSE endpoint
    sendStream: async (message, modelId, onChunk, onDone, onError) => {
        try {
            const response = await fetch(`${API_BASE}/chat/stream`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message, model_id: modelId }),
            });

            if (!response.ok) {
                // Streaming başarısızsa normal endpoint'e fallback
                const res = await api.post('/chat', { message, model_id: modelId });
                if (res.data.success) {
                    onChunk?.(res.data.response);
                    onDone?.();
                } else {
                    onError?.(res.data.error || 'Error');
                }
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            if (data.error) {
                                onError?.(data.error);
                            } else if (data.done) {
                                onDone?.();
                            } else if (data.text) {
                                onChunk?.(data.text);
                            }
                        } catch { /* JSON parse error, skip */ }
                    }
                }
            }
        } catch (error) {
            // Fallback to normal endpoint
            try {
                const res = await api.post('/chat', { message, model_id: modelId });
                if (res.data.success) {
                    onChunk?.(res.data.response);
                    onDone?.();
                } else {
                    onError?.(res.data.error);
                }
            } catch {
                onError?.(error.message);
            }
        }
    }
};

// Health check
export const healthCheck = () => api.get('/health');

// Training API
export const trainingApi = {
    start: (config) => api.post('/training/start', config),
    getStatus: (sessionId) => api.get(`/training/status/${sessionId}`),
    getSessions: () => api.get('/training/sessions'),
    stop: (sessionId) => api.post(`/training/stop/${sessionId}`),
    clearOld: (keepLast = 10) => api.delete(`/training/sessions/old?keep_last=${keepLast}`)
};

// Advanced Models API (BiLSTM, Transformer, GRU, Ensemble)
export const advancedModelsApi = {
    // Model listesi
    getModels: () => api.get('/advanced/models'),

    // Model karşılaştırma
    compare: () => api.get('/advanced/compare'),

    // Eğitim
    train: (config) => api.post('/advanced/train', config),
    getTrainingStatus: (trainingId) => api.get(`/advanced/train/${trainingId}`),
    getTrainings: () => api.get('/advanced/train'),

    // Hiperparametre optimizasyonu
    optimize: (config) => api.post('/advanced/optimize', config),
    getOptimizationStatus: (optId) => api.get(`/advanced/optimize/${optId}`),
    getOptimizations: () => api.get('/advanced/optimize'),

    // Ensemble
    createEnsemble: (config) => api.post('/advanced/ensemble', config),

    // Canlı metrikler
    getLiveMetrics: () => api.get('/advanced/metrics/live')
};

// Database API
export const databaseApi = {
    getStats: () => api.get('/database/stats'),
    getTables: () => api.get('/database/tables'),
    query: (sql) => api.post('/database/query', { sql }),
    backup: () => api.post('/database/backup'),
    restore: (file) => api.post('/database/restore', file),
    clear: (table) => api.delete(`/database/table/${table}`),
    getTableData: (table, page = 1, limit = 50) =>
        api.get(`/database/table/${table}?page=${page}&limit=${limit}`),
};

// Logs API
export const logsApi = {
    getAll: (page = 1, limit = 50, level = null) =>
        api.get(`/logs?page=${page}&limit=${limit}${level ? `&level=${level}` : ''}`),
    getByLevel: (level) => api.get(`/logs/level/${level}`),
    search: (query) => api.get(`/logs/search?q=${encodeURIComponent(query)}`),
    export: (format = 'json') => api.get(`/logs/export?format=${format}`, { responseType: 'blob' }),
    clear: () => api.delete('/logs'),
    getStats: () => api.get('/logs/stats'),
};

// Network API
export const networkApi = {
    getStatus: () => api.get('/network/status'),
    getConnections: () => api.get('/network/connections'),
    getTraffic: (hours = 24) => api.get(`/network/traffic?hours=${hours}`),
    getInterfaces: () => api.get('/network/interfaces'),
    startCapture: (iface) => api.post('/network/capture/start', { interface: iface }),
    stopCapture: () => api.post('/network/capture/stop'),
    getAlerts: (limit = 50) => api.get(`/network/alerts?limit=${limit}`),
    getPackets: (limit = 100) => api.get(`/network/packets?limit=${limit}`),
};

// Reports API
export const reportsApi = {
    generate: (type, params = {}) => api.post('/reports/generate', { type, ...params }),
    getAll: () => api.get('/reports'),
    getById: (id) => api.get(`/reports/${id}`),
    download: (id, format = 'pdf') =>
        api.get(`/reports/${id}/download?format=${format}`, { responseType: 'blob' }),
    schedule: (config) => api.post('/reports/schedule', config),
    delete: (id) => api.delete(`/reports/${id}`),
    getTemplates: () => api.get('/reports/templates'),
};

// Scanner API
export const scannerApi = {
    scanFile: (file) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/scanner/file', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    },
    scanUrl: (url) => api.post('/scanner/url', { url }),
    scanHash: (hash) => api.get(`/scanner/hash/${hash}`),
    getHistory: (limit = 50) => api.get(`/scanner/history?limit=${limit}`),
    getStats: () => api.get('/scanner/stats'),
    getResult: (scanId) => api.get(`/scanner/result/${scanId}`),
};

// Settings API  
export const settingsApi = {
    getAll: () => api.get('/settings'),
    get: (key) => api.get(`/settings/${key}`),
    update: (key, value) => api.put(`/settings/${key}`, { value }),
    updateBulk: (settings) => api.put('/settings', settings),
    reset: () => api.post('/settings/reset'),
    export: () => api.get('/settings/export'),
    import: (config) => api.post('/settings/import', config),
};

// Threat Analysis API
export const threatApi = {
    analyze: (data) => api.post('/threat/analyze', data),
    getIntel: (indicator) => api.get(`/threat/intel/${encodeURIComponent(indicator)}`),
    getMitre: (technique = null) => api.get(`/threat/mitre${technique ? `/${technique}` : ''}`),
    getIOCs: (limit = 100) => api.get(`/threat/iocs?limit=${limit}`),
    enrichIP: (ip) => api.get(`/threat/enrich/ip/${ip}`),
    enrichDomain: (domain) => api.get(`/threat/enrich/domain/${domain}`),
    getTimeline: (hours = 24) => api.get(`/threat/timeline?hours=${hours}`),
    getTactics: () => api.get('/threat/mitre/tactics'),
};

// Advanced ML API (AutoML, XAI, A/B Testing, Drift, Federated)
export const advancedMLApi = {
    // AutoML
    automlSearch: (config) => api.post('/ml/automl/search', config),
    automlStatus: () => api.get('/ml/automl/status'),
    automlResults: () => api.get('/ml/automl/results'),

    // XAI (Explainability)
    getFeatureImportance: (model = 'latest') =>
        api.get(`/ml/xai/feature-importance?model_name=${model}`),
    getShapValues: (model, sample) => api.post('/ml/xai/shap', { model, sample }),
    getLimeExplanation: (model, sample) => api.post('/ml/xai/lime', { model, sample }),

    // A/B Testing
    getABTests: () => api.get('/ml/ab/tests'),
    createABTest: (config) => api.post('/ml/ab/tests', config),
    getABResults: (testId) => api.get(`/ml/ab/tests/${testId}/results`),
    stopABTest: (testId) => api.post(`/ml/ab/tests/${testId}/stop`),

    // Drift Detection
    getDriftStatus: () => api.get('/ml/drift/status'),
    checkDrift: (data) => api.post('/ml/drift/check', data),
    getDriftHistory: () => api.get('/ml/drift/history'),

    // Federated Learning
    getFLStatus: () => api.get('/ml/federated/status'),
    startFLRound: () => api.post('/ml/federated/round'),
    getFLClients: () => api.get('/ml/federated/clients'),
    getFLHistory: () => api.get('/ml/federated/history'),
};

// Prediction API
export const predictionApi = {
    predict: (model, features) => api.post('/prediction/predict', { model_id: model, features }),
    predictBatch: (model, data) => api.post('/prediction/batch', { model_id: model, data }),
    analyzeFile: (file) => {
        const formData = new FormData();
        formData.append('file', file);
        return api.post('/prediction/analyze-file', formData, {
            headers: { 'Content-Type': 'multipart/form-data' }
        });
    },
    getHistory: (limit = 50) => api.get(`/prediction/history?limit=${limit}`),
    getStats: () => api.get('/prediction/stats'),
};

// Auth API
export const authApi = {
    login: (credentials) => api.post('/auth/login', credentials),
    register: (data) => api.post('/auth/register', data),
    logout: () => api.post('/auth/logout'),
    getProfile: () => api.get('/auth/profile'),
    updateProfile: (data) => api.put('/auth/profile', data),
    changePassword: (data) => api.post('/auth/change-password', data),
};

// AI Decision API - Full AI Pipeline
export const aiDecisionApi = {
    // Full pipeline
    decide: (features, sourceInfo = 'Unknown', targetInfo = 'Unknown') =>
        api.post('/ai/decide', { features, source_info: sourceInfo, target_info: targetInfo }),
    decideBatch: (samples) => api.post('/ai/decide/batch', { samples }),

    // Zero-Day Detection
    zeroDay: {
        detect: (features) => api.post('/ai/zero-day/detect', { features }),
        train: (epochs = 30, sensitivity = 3) =>
            api.post('/ai/zero-day/train', { epochs, sensitivity }),
        getStats: () => api.get('/ai/zero-day/stats'),
    },

    // Model Selection
    modelSelect: {
        select: (features) => api.post('/ai/model-select', { features }),
        getStats: () => api.get('/ai/model-select/stats'),
    },

    // RL Threshold
    threshold: {
        decide: (modelConfidence, anomalyScore, history = null) =>
            api.post('/ai/threshold/decide', {
                model_confidence: modelConfidence,
                anomaly_score: anomalyScore,
                history
            }),
        train: (episodes = 100) => api.post('/ai/threshold/train', { episodes }),
        getStats: () => api.get('/ai/threshold/stats'),
    },

    // Explainability
    explain: {
        attack: (features, attackType, topN = 5) =>
            api.post('/ai/explain', { features, attack_type: attackType, top_n: topN }),
        getFeatures: () => api.get('/ai/explain/features'),
    },

    // Report Generation
    report: {
        generate: (attackType, confidence, explanation = null, template = 'attack_summary') =>
            api.post('/ai/report/generate', {
                attack_type: attackType,
                confidence,
                explanation,
                template
            }),
        getHistory: () => api.get('/ai/report/history'),
        getTemplates: () => api.get('/ai/report/templates'),
    },

    // Engine Management
    engine: {
        getStats: () => api.get('/ai/engine/stats'),
        initialize: () => api.post('/ai/engine/initialize'),
        health: () => api.get('/ai/engine/health'),
    },
};

export default api;

