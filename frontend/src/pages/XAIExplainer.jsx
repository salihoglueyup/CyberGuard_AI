import React, { useState, useEffect } from 'react';
import api from '../services/api';

const XAIExplainer = () => {
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [method, setMethod] = useState('shap');
    const [explanation, setExplanation] = useState(null);
    const [featureImportance, setFeatureImportance] = useState(null);
    const [loading, setLoading] = useState(false);
    const [methods, setMethods] = useState([]);

    useEffect(() => {
        loadModels();
        loadMethods();
    }, []);

    const loadModels = async () => {
        try {
            const response = await api.get('/models');
            if (response.data.success) {
                setModels(response.data.data);
                if (response.data.data.length > 0) {
                    setSelectedModel(response.data.data[0].id);
                }
            }
        } catch (error) {
            console.error('Error loading models:', error);
        }
    };

    const loadMethods = async () => {
        try {
            const response = await api.get('/xai/explanation-methods');
            if (response.data.success) {
                setMethods(response.data.data.methods);
            }
        } catch (error) {
            console.error('Error loading methods:', error);
        }
    };

    const generateExplanation = async () => {
        setLoading(true);
        try {
            // Generate sample features
            const features = Array(78).fill(0).map(() => Math.random() * 100);

            const response = await api.post('/xai/explain', {
                model_id: selectedModel,
                features: features,
                num_features: 10,
                method: method
            });

            if (response.data.success) {
                setExplanation(response.data.data);
            }
        } catch (error) {
            console.error('Error generating explanation:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadFeatureImportance = async () => {
        setLoading(true);
        try {
            const response = await api.get(`/xai/feature-importance/${selectedModel}`);
            if (response.data.success) {
                setFeatureImportance(response.data.data);
            }
        } catch (error) {
            console.error('Error loading feature importance:', error);
        } finally {
            setLoading(false);
        }
    };

    const getSeverityColor = (contribution) => {
        return contribution === 'positive' ? '#10b981' : '#ef4444';
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
                        🔍 Explainable AI (XAI)
                    </h1>
                    <p className="text-gray-400 mt-2">
                        Model tahminlerini SHAP ve LIME ile açıklayın
                    </p>
                </div>

                {/* Controls */}
                <div className="bg-gray-800 rounded-xl p-6 mb-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Model Seç
                            </label>
                            <select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
                            >
                                {models.map(model => (
                                    <option key={model.id} value={model.id}>
                                        {model.name} ({(model.accuracy * 100).toFixed(2)}%)
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Açıklama Metodu
                            </label>
                            <select
                                value={method}
                                onChange={(e) => setMethod(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
                            >
                                <option value="shap">SHAP</option>
                                <option value="lime">LIME</option>
                            </select>
                        </div>

                        <div className="flex items-end gap-2">
                            <button
                                onClick={generateExplanation}
                                disabled={loading}
                                className="flex-1 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
                            >
                                {loading ? 'Yükleniyor...' : 'Açıklama Oluştur'}
                            </button>
                            <button
                                onClick={loadFeatureImportance}
                                disabled={loading}
                                className="flex-1 bg-pink-600 hover:bg-pink-700 px-4 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
                            >
                                Feature Importance
                            </button>
                        </div>
                    </div>
                </div>

                {/* Methods Info */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                    {methods.map(m => (
                        <div key={m.id} className="bg-gray-800 rounded-xl p-5">
                            <h3 className="font-bold text-lg text-purple-400">{m.name}</h3>
                            <p className="text-gray-400 text-sm mt-2">{m.description}</p>
                            <div className="mt-3 flex gap-2 flex-wrap">
                                {m.pros?.map((pro, i) => (
                                    <span key={i} className="bg-green-900/50 text-green-400 px-2 py-1 rounded text-xs">
                                        ✓ {pro}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>

                {/* Explanation Results */}
                {explanation && (
                    <div className="bg-gray-800 rounded-xl p-6 mb-6">
                        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                            <span className="text-2xl">📊</span>
                            {explanation.explanation.method} Sonuçları
                        </h2>

                        <div className="space-y-3">
                            {explanation.explanation.top_features?.map((feature, index) => (
                                <div key={index} className="bg-gray-700/50 rounded-lg p-4">
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="font-medium">{feature.feature}</span>
                                        <span
                                            className="px-2 py-1 rounded text-xs font-medium"
                                            style={{
                                                backgroundColor: getSeverityColor(feature.contribution) + '20',
                                                color: getSeverityColor(feature.contribution)
                                            }}
                                        >
                                            {feature.contribution === 'positive' ? '↑ Artırıcı' : '↓ Azaltıcı'}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-4">
                                        <div className="flex-1 bg-gray-600 rounded-full h-2">
                                            <div
                                                className="h-2 rounded-full"
                                                style={{
                                                    width: `${Math.min(Math.abs(feature.shap_value || feature.weight) * 100, 100)}%`,
                                                    backgroundColor: getSeverityColor(feature.contribution)
                                                }}
                                            />
                                        </div>
                                        <span className="text-sm text-gray-400">
                                            Değer: {feature.value?.toFixed(2)}
                                        </span>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Feature Importance */}
                {featureImportance && (
                    <div className="bg-gray-800 rounded-xl p-6">
                        <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
                            <span className="text-2xl">🎯</span>
                            Global Feature Importance
                        </h2>

                        <div className="space-y-2">
                            {featureImportance.feature_importance?.slice(0, 15).map((feature, index) => (
                                <div key={index} className="flex items-center gap-4">
                                    <span className="w-8 text-sm text-gray-500">#{feature.rank}</span>
                                    <span className="w-48 text-sm truncate">{feature.feature}</span>
                                    <div className="flex-1 bg-gray-700 rounded-full h-3">
                                        <div
                                            className="h-3 rounded-full bg-gradient-to-r from-purple-500 to-pink-500"
                                            style={{ width: `${feature.importance * 100}%` }}
                                        />
                                    </div>
                                    <span className="w-16 text-right text-sm text-gray-400">
                                        {(feature.importance * 100).toFixed(1)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default XAIExplainer;
