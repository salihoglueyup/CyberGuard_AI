import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Trash2, Brain, Sparkles, AlertTriangle, Shield, TrendingUp, Search, ChevronDown, ChevronUp, Cpu, Zap, Target, Radio, BarChart3 } from 'lucide-react';
import { chatApi, modelsApi } from '../services/api';
import { useChatStore } from '../store';

// Hızlı Analiz Aksiyonları
const QUICK_ACTIONS = [
    {
        id: 'summary',
        label: 'Tehdit Özeti',
        icon: Sparkles,
        color: 'blue',
        message: 'Tüm veritabanındaki tehdit durumunu kısa özetle.'
    },
    {
        id: 'critical',
        label: 'Kritik Analiz',
        icon: AlertTriangle,
        color: 'red',
        message: 'Kritik ve yüksek seviyeli saldırıları kısa analiz et.'
    },
    {
        id: 'defense',
        label: 'Savunma Önerileri',
        icon: Shield,
        color: 'green',
        message: 'Mevcut tehditlere karşı en önemli 5 savunma önerisini listele.'
    },
    {
        id: 'trends',
        label: 'Trend Analizi',
        icon: TrendingUp,
        color: 'orange',
        message: 'Saldırı trendlerini ve en aktif saat dilimlerini analiz et.'
    },
    {
        id: 'investigate',
        label: 'IP Araştır',
        icon: Search,
        color: 'purple',
        message: 'En çok saldırı yapan 5 IP adresini ve risk durumlarını listele.'
    },
    // IDS Model Aksiyonları
    {
        id: 'model_compare',
        label: 'Model Karşılaştır',
        icon: BarChart3,
        color: 'blue',
        message: 'Mevcut IDS modellerinin performanslarını karşılaştır. Hangi model en iyi accuracy ve F1-score değerlerine sahip?'
    },
    {
        id: 'model_recommend',
        label: 'Model Öner',
        icon: Cpu,
        color: 'purple',
        message: 'Mevcut saldırı tiplerine göre hangi IDS modelini kullanmamı önerirsin? SSA-LSTMIDS mi, BiLSTM mi, yoksa Transformer mı?'
    },
    {
        id: 'realtime_status',
        label: 'Real-time IDS',
        icon: Radio,
        color: 'green',
        message: 'Real-time IDS sisteminin durumu nedir? Son tespit edilen saldırılar ve alert\'ler hakkında bilgi ver.'
    },
    {
        id: 'mitre_analysis',
        label: 'MITRE ATT&CK',
        icon: Target,
        color: 'red',
        message: 'Son saldırıları MITRE ATT&CK framework\'üne göre analiz et. Hangi taktik ve teknikler kullanılıyor?'
    }
];

const colorClasses = {
    blue: 'bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 border-blue-500/30',
    red: 'bg-red-600/20 text-red-400 hover:bg-red-600/30 border-red-500/30',
    green: 'bg-green-600/20 text-green-400 hover:bg-green-600/30 border-green-500/30',
    orange: 'bg-orange-600/20 text-orange-400 hover:bg-orange-600/30 border-orange-500/30',
    purple: 'bg-purple-600/20 text-purple-400 hover:bg-purple-600/30 border-purple-500/30'
};

export default function AIAssistant() {
    const [input, setInput] = useState('');
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedModelInfo, setSelectedModelInfo] = useState(null);
    const [showQuickActions, setShowQuickActions] = useState(true);

    // AI Provider state
    const [providers, setProviders] = useState([]);
    const [selectedProvider, setSelectedProvider] = useState('groq');

    const messagesEndRef = useRef(null);

    const { messages, isLoading, addMessage, setLoading, clearMessages } = useChatStore();

    useEffect(() => {
        loadModels();
        loadProviders();
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Seçili model değiştiğinde bilgilerini güncelle
    useEffect(() => {
        if (selectedModel && models.length > 0) {
            const model = models.find(m => m.id === selectedModel);
            setSelectedModelInfo(model);
        }
    }, [selectedModel, models]);

    const loadProviders = async () => {
        try {
            const res = await chatApi.getProviders();
            if (res.data.providers) {
                setProviders(res.data.providers);
                setSelectedProvider(res.data.default || 'groq');
            }
        } catch (error) {
            console.error('Providers load error:', error);
            // Fallback providers
            setProviders([
                { id: 'groq', name: '🦙 Groq (Llama 3.3)', available: true },
                { id: 'gemini', name: '🔮 Google Gemini', available: false },
            ]);
        }
    };

    const loadModels = async () => {
        try {
            const res = await modelsApi.getDeployed();
            if (res.data.success) {
                setModels(res.data.data);
                if (res.data.data.length > 0) {
                    setSelectedModel(res.data.data[0].id);
                    setSelectedModelInfo(res.data.data[0]);
                }
            }
        } catch (error) {
            console.error('Models load error:', error);
        }
    };

    const handleSend = async (messageToSend = null) => {
        const text = messageToSend || input.trim();
        console.log('🚀 handleSend called:', text, 'provider:', selectedProvider, 'isLoading:', isLoading);

        if (!text || isLoading) {
            console.log('⚠️ Skipping - empty text or loading');
            return;
        }

        setInput('');
        addMessage({ role: 'user', content: text });
        setLoading(true);

        try {
            console.log('📤 Sending to API...', selectedProvider, selectedModel);
            const res = await chatApi.send(text, selectedModel, selectedProvider);
            console.log('📥 API Response:', res.data);

            if (res.data.success) {
                addMessage({ role: 'assistant', content: res.data.response, provider: res.data.provider });
            } else {
                addMessage({ role: 'assistant', content: `❌ Hata: ${res.data.error}` });
            }
        } catch (error) {
            console.error('❌ API Error:', error);
            addMessage({ role: 'assistant', content: `❌ Bağlantı hatası: ${error.message}` });
        } finally {
            setLoading(false);
        }
    };

    const handleQuickAction = (action) => {
        handleSend(action.message);
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    return (
        <div className="h-[calc(100vh-6rem)] flex flex-col fade-in">
            {/* Header */}
            <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-blue-600 flex items-center justify-center">
                        <Brain className="w-7 h-7 text-white" />
                    </div>
                    <div>
                        <h1 className="text-2xl font-bold text-white">AI Assistant</h1>
                        <p className="text-slate-400 text-sm">Tehdit analizi ve güvenlik asistanı</p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    {/* AI Provider Selector */}
                    <div className="flex items-center gap-2 bg-slate-800/50 rounded-xl px-3 py-2 border border-slate-700">
                        <Cpu className="w-4 h-4 text-emerald-400" />
                        <select
                            value={selectedProvider}
                            onChange={(e) => setSelectedProvider(e.target.value)}
                            className="bg-transparent border-none text-white text-sm focus:outline-none cursor-pointer"
                        >
                            {providers.map((p) => (
                                <option
                                    key={p.id}
                                    value={p.id}
                                    className="bg-slate-800"
                                    disabled={!p.available}
                                >
                                    {p.name} {!p.available && '(N/A)'}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Model Selector */}
                    <div className="flex items-center gap-2 bg-slate-800/50 rounded-xl px-3 py-2 border border-slate-700">
                        <Bot className="w-4 h-4 text-purple-400" />
                        <select
                            value={selectedModel}
                            onChange={(e) => setSelectedModel(e.target.value)}
                            className="bg-transparent border-none text-white text-sm focus:outline-none cursor-pointer"
                        >
                            {models.map((m) => (
                                <option key={m.id} value={m.id} className="bg-slate-800">
                                    {m.name} ({(m.accuracy * 100).toFixed(1)}%)
                                </option>
                            ))}
                        </select>
                    </div>

                    <button
                        onClick={clearMessages}
                        className="p-2.5 bg-red-500/20 hover:bg-red-500/30 rounded-xl transition-colors"
                        title="Sohbeti Temizle"
                    >
                        <Trash2 className="w-5 h-5 text-red-400" />
                    </button>
                </div>
            </div>

            {/* Model Info Bar */}
            {selectedModelInfo && (
                <div className="mb-4 p-3 bg-gradient-to-r from-purple-900/20 to-blue-900/20 rounded-xl border border-purple-500/20">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <span className="text-sm text-slate-400">Aktif Model:</span>
                            <span className="text-white font-medium">{selectedModelInfo.name}</span>
                            <span className="px-2 py-0.5 rounded-full bg-green-500/20 text-green-400 text-xs">
                                {(selectedModelInfo.accuracy * 100).toFixed(1)}% Doğruluk
                            </span>
                        </div>
                        <button
                            onClick={() => setShowQuickActions(!showQuickActions)}
                            className="flex items-center gap-1 text-xs text-slate-400 hover:text-white transition-colors"
                        >
                            Hızlı Aksiyonlar
                            {showQuickActions ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                        </button>
                    </div>
                </div>
            )}

            {/* Quick Action Buttons */}
            {showQuickActions && (
                <div className="mb-4 p-4 bg-slate-800/30 rounded-xl border border-slate-700/50">
                    <p className="text-xs text-slate-500 mb-3 uppercase tracking-wide">⚡ Hızlı Tehdit Analizi</p>
                    <div className="flex flex-wrap gap-2">
                        {QUICK_ACTIONS.map(action => {
                            const Icon = action.icon;
                            return (
                                <button
                                    key={action.id}
                                    onClick={() => handleQuickAction(action)}
                                    disabled={isLoading}
                                    className={`
                                        flex items-center gap-2 px-4 py-2 rounded-xl font-medium border
                                        ${colorClasses[action.color]}
                                        ${isLoading ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:scale-105'}
                                        transition-all duration-200
                                    `}
                                >
                                    <Icon className="w-4 h-4" />
                                    <span className="text-sm">{action.label}</span>
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Chat Container */}
            <div className="flex-1 overflow-y-auto bg-slate-900/50 rounded-xl p-4 space-y-4">
                {messages.length === 0 && (
                    <div className="text-center py-12">
                        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-purple-500/20 to-blue-500/20 flex items-center justify-center mx-auto mb-4">
                            <Brain className="w-10 h-10 text-purple-400" />
                        </div>
                        <h3 className="text-xl font-semibold text-white mb-2">CyberGuard AI Assistant</h3>
                        <p className="text-slate-400 max-w-md mx-auto">
                            Tehdit analizi, saldırı tespiti ve güvenlik önerileri için yapay zeka destekli asistanınız.
                        </p>
                        <div className="flex flex-wrap justify-center gap-2 mt-6">
                            {['Sistemde kaç saldırı var?', 'En aktif saldırganlar kimler?', 'Güvenlik önerileri ver'].map((q) => (
                                <button
                                    key={q}
                                    onClick={() => setInput(q)}
                                    className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-xl text-sm text-slate-300 transition-colors border border-slate-700"
                                >
                                    {q}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {messages.map((msg, idx) => (
                    <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                        {msg.role === 'assistant' && (
                            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500/30 to-blue-500/30 flex items-center justify-center flex-shrink-0">
                                <Bot className="w-5 h-5 text-purple-400" />
                            </div>
                        )}
                        <div className={`max-w-[75%] px-4 py-3 rounded-2xl ${msg.role === 'user'
                            ? 'bg-gradient-to-r from-blue-600 to-blue-700 text-white'
                            : 'bg-slate-800 text-slate-200 border border-slate-700/50'
                            }`}>
                            <p className="whitespace-pre-wrap text-sm leading-relaxed">{msg.content}</p>
                        </div>
                        {msg.role === 'user' && (
                            <div className="w-9 h-9 rounded-xl bg-blue-600/30 flex items-center justify-center flex-shrink-0">
                                <User className="w-5 h-5 text-blue-400" />
                            </div>
                        )}
                    </div>
                ))}

                {isLoading && (
                    <div className="flex gap-3">
                        <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-purple-500/30 to-blue-500/30 flex items-center justify-center">
                            <Bot className="w-5 h-5 text-purple-400 animate-pulse" />
                        </div>
                        <div className="bg-slate-800 border border-slate-700/50 px-4 py-3 rounded-2xl">
                            <div className="flex items-center gap-2">
                                <div className="flex gap-1">
                                    <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                    <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                    <span className="w-2 h-2 bg-purple-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                                </div>
                                <span className="text-sm text-slate-400">Analiz ediliyor...</span>
                            </div>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <div className="mt-4 flex gap-3">
                <div className="flex-1 relative">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Bir soru sorun veya hızlı aksiyonları kullanın..."
                        className="w-full px-5 py-4 bg-slate-800 border border-slate-700 rounded-xl 
                                   text-white placeholder-slate-500 focus:outline-none focus:border-purple-500 
                                   focus:ring-2 focus:ring-purple-500/20 transition-all"
                    />
                </div>
                <button
                    onClick={() => handleSend()}
                    disabled={isLoading || !input.trim()}
                    className="px-6 py-4 bg-gradient-to-r from-purple-600 to-blue-600 
                               hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 
                               rounded-xl transition-all duration-200 flex items-center gap-2 
                               shadow-lg shadow-purple-500/20"
                >
                    <Send className="w-5 h-5 text-white" />
                </button>
            </div>
        </div>
    );
}
