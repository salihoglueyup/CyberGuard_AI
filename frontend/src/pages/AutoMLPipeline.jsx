import React, { useState, useEffect } from 'react';
import api from '../services/api';

const AutoMLPipeline = () => {
    const [algorithms, setAlgorithms] = useState([]);
    const [recommendations, setRecommendations] = useState(null);
    const [jobResult, setJobResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [config, setConfig] = useState({
        dataset_name: 'cicids2017',
        task_type: 'classification',
        target_metric: 'accuracy',
        max_models: 5,
        time_limit_minutes: 30,
        include_deep_learning: true
    });

    useEffect(() => {
        loadAlgorithms();
        loadRecommendations();
    }, []);

    const loadAlgorithms = async () => {
        try {
            const response = await api.get('/automl/algorithms');
            if (response.data.success) {
                setAlgorithms(response.data.data.algorithms);
            }
        } catch (error) {
            console.error('Error loading algorithms:', error);
        }
    };

    const loadRecommendations = async () => {
        try {
            const response = await api.get('/automl/recommendations?dataset_type=network_traffic');
            if (response.data.success) {
                setRecommendations(response.data.data.recommendation);
            }
        } catch (error) {
            console.error('Error loading recommendations:', error);
        }
    };

    const startAutoML = async () => {
        setLoading(true);
        try {
            const response = await api.post('/automl/start', config);
            if (response.data.success) {
                setJobResult(response.data.data);
            }
        } catch (error) {
            console.error('Error starting AutoML:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            <div className="max-w-7xl mx-auto">
                {/* Header */}
                <div className="mb-8">
                    <h1 className="text-3xl font-bold bg-gradient-to-r from-orange-400 to-red-500 bg-clip-text text-transparent">
                        🤖 AutoML Pipeline
                    </h1>
                    <p className="text-gray-400 mt-2">
                        Otomatik model seçimi ve hiperparametre optimizasyonu
                    </p>
                </div>

                {/* Configuration */}
                <div className="bg-gray-800 rounded-xl p-6 mb-6">
                    <h2 className="text-lg font-medium mb-4">⚙️ Konfigürasyon</h2>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Dataset</label>
                            <select
                                value={config.dataset_name}
                                onChange={(e) => setConfig({ ...config, dataset_name: e.target.value })}
                                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
                            >
                                <option value="cicids2017">CICIDS2017</option>
                                <option value="nsl_kdd">NSL-KDD</option>
                                <option value="botiot">BoT-IoT</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Hedef Metrik</label>
                            <select
                                value={config.target_metric}
                                onChange={(e) => setConfig({ ...config, target_metric: e.target.value })}
                                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
                            >
                                <option value="accuracy">Accuracy</option>
                                <option value="f1_score">F1-Score</option>
                                <option value="precision">Precision</option>
                                <option value="recall">Recall</option>
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-1">Max Models</label>
                            <input
                                type="number"
                                value={config.max_models}
                                onChange={(e) => setConfig({ ...config, max_models: parseInt(e.target.value) })}
                                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
                                min={1}
                                max={10}
                            />
                        </div>
                    </div>

                    <div className="flex items-center gap-4 mt-4">
                        <label className="flex items-center gap-2 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={config.include_deep_learning}
                                onChange={(e) => setConfig({ ...config, include_deep_learning: e.target.checked })}
                                className="w-4 h-4 rounded"
                            />
                            <span>Deep Learning dahil et</span>
                        </label>

                        <button
                            onClick={startAutoML}
                            disabled={loading}
                            className="ml-auto bg-orange-600 hover:bg-orange-700 px-6 py-2 rounded-lg font-medium transition-colors disabled:opacity-50"
                        >
                            {loading ? '⏳ Çalışıyor...' : '🚀 AutoML Başlat'}
                        </button>
                    </div>
                </div>

                {/* Recommendations */}
                {recommendations && (
                    <div className="bg-gray-800 rounded-xl p-6 mb-6">
                        <h2 className="text-lg font-medium mb-4">💡 Öneriler</h2>
                        <div className="bg-gradient-to-r from-orange-900/50 to-red-900/50 rounded-lg p-4">
                            <div className="text-lg font-bold text-orange-400 mb-2">
                                Önerilen: {recommendations.top_pick?.algorithm?.toUpperCase()}
                            </div>
                            <p className="text-gray-300">{recommendations.top_pick?.why}</p>
                            <p className="text-green-400 mt-2">Beklenen: {recommendations.top_pick?.expected_accuracy}</p>
                        </div>
                    </div>
                )}

                {/* Available Algorithms */}
                <div className="bg-gray-800 rounded-xl p-6 mb-6">
                    <h2 className="text-lg font-medium mb-4">📚 Mevcut Algoritmalar</h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {algorithms.map(algo => (
                            <div key={algo.id} className="bg-gray-700/50 rounded-lg p-4">
                                <div className="font-bold text-orange-400">{algo.name}</div>
                                <div className="text-sm text-gray-400 mt-1">Type: {algo.type}</div>
                                <div className="flex gap-2 mt-2 flex-wrap">
                                    {algo.best_for?.map((tag, i) => (
                                        <span key={i} className="bg-gray-600 px-2 py-0.5 rounded text-xs">
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                                <div className="text-xs text-gray-500 mt-2">
                                    Complexity: {algo.complexity} | Time: {algo.training_time}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Results */}
                {jobResult && (
                    <div className="bg-gray-800 rounded-xl p-6">
                        <h2 className="text-lg font-medium mb-4">📊 Sonuçlar</h2>

                        {/* Best Model */}
                        {jobResult.best_model && (
                            <div className="bg-gradient-to-r from-green-900/50 to-emerald-900/50 rounded-lg p-5 mb-6">
                                <div className="text-sm text-green-400 mb-1">🏆 En İyi Model</div>
                                <div className="text-2xl font-bold">{jobResult.best_model.algorithm_name}</div>
                                <div className="grid grid-cols-4 gap-4 mt-4">
                                    {Object.entries(jobResult.best_model.metrics || {}).map(([key, value]) => (
                                        <div key={key} className="text-center">
                                            <div className="text-2xl font-bold text-green-400">{(value * 100).toFixed(2)}%</div>
                                            <div className="text-xs text-gray-400 capitalize">{key.replace('_', ' ')}</div>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Leaderboard */}
                        <h3 className="font-medium mb-3">📋 Leaderboard</h3>
                        <div className="overflow-x-auto">
                            <table className="w-full">
                                <thead>
                                    <tr className="text-left text-gray-400 text-sm">
                                        <th className="pb-3">Rank</th>
                                        <th className="pb-3">Algorithm</th>
                                        <th className="pb-3">Accuracy</th>
                                        <th className="pb-3">F1-Score</th>
                                        <th className="pb-3">Training Time</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {jobResult.leaderboard?.map((model, index) => (
                                        <tr key={index} className={`border-t border-gray-700 ${index === 0 ? 'bg-green-900/20' : ''}`}>
                                            <td className="py-3">
                                                {index === 0 ? '🥇' : index === 1 ? '🥈' : index === 2 ? '🥉' : index + 1}
                                            </td>
                                            <td className="py-3 font-medium">{model.algorithm_name}</td>
                                            <td className="py-3 text-green-400">{(model.metrics?.accuracy * 100).toFixed(2)}%</td>
                                            <td className="py-3">{(model.metrics?.f1_score * 100).toFixed(2)}%</td>
                                            <td className="py-3 text-gray-400">{model.training_time_seconds}s</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default AutoMLPipeline;
