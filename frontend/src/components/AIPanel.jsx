import { useState, useRef, useEffect } from 'react';
import { X, Send, Bot, User, Loader2, Sparkles, AlertTriangle, Shield, TrendingUp, Search } from 'lucide-react';
import { chatApi } from '../services/api';

/**
 * AIPanel - Yeniden kullanılabilir AI Chat Paneli
 * 
 * Props:
 * - isOpen: Panel açık mı?
 * - onClose: Panel kapatma fonksiyonu
 * - context: Gönderilecek ek context (attacks, model info vb.)
 * - modelId: Seçili model ID
 * - title: Panel başlığı
 */
export default function AIPanel({
    isOpen,
    onClose,
    context = null,
    modelId = null,
    title = "AI Analiz Asistanı"
}) {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Panel açıldığında hoş geldin mesajı
    useEffect(() => {
        if (isOpen && messages.length === 0) {
            setMessages([{
                role: 'assistant',
                content: '🤖 Merhaba! Ben CyberGuard AI asistanınızım. Size tehdit analizi, güvenlik önerileri ve saldırı tespiti konularında yardımcı olabilirim.\n\nNasıl yardımcı olabilirim?'
            }]);
        }
    }, [isOpen, messages.length]);

    const handleSend = async () => {
        if (!input.trim() || loading) return;

        const userMessage = input.trim();
        setInput('');
        setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
        setLoading(true);

        try {
            // Context varsa mesaja ekle
            let fullMessage = userMessage;
            if (context) {
                fullMessage = `[Context: ${JSON.stringify(context)}]\n\n${userMessage}`;
            }

            const res = await chatApi.send(fullMessage, modelId);

            if (res.data.success) {
                setMessages(prev => [...prev, { role: 'assistant', content: res.data.response }]);
            } else {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: '❌ Bir hata oluştu: ' + (res.data.error || 'Bilinmeyen hata')
                }]);
            }
        } catch (error) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: '❌ Bağlantı hatası: ' + error.message
            }]);
        } finally {
            setLoading(false);
        }
    };

    const handleQuickAction = async (action) => {
        if (loading) return;

        const actionMessages = {
            'summary': '📊 Son 24 saatteki tehdit durumunu özetle',
            'critical': '🔴 Kritik seviyedeki saldırıları analiz et ve öneriler sun',
            'defense': '🛡️ Mevcut tehditlere karşı savunma önerileri ver',
            'trends': '📈 Saldırı trendlerini analiz et ve tahminler yap',
            'investigate': '🔍 En çok saldırı yapan IP adreslerini araştır'
        };

        const message = actionMessages[action];
        if (message) {
            setInput(message);
            // Otomatik gönder
            const userMessage = message;
            setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
            setLoading(true);

            try {
                const res = await chatApi.send(userMessage, modelId);
                if (res.data.success) {
                    setMessages(prev => [...prev, { role: 'assistant', content: res.data.response }]);
                }
            } catch (error) {
                setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: '❌ Hata: ' + error.message
                }]);
            } finally {
                setLoading(false);
                setInput('');
            }
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-end">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/50 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Panel */}
            <div className="relative w-full max-w-lg h-full bg-slate-900 border-l border-slate-700 shadow-2xl flex flex-col animate-slide-in-right">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-slate-700 bg-gradient-to-r from-purple-900/30 to-blue-900/30">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                            <Bot className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <h2 className="text-lg font-semibold text-white">{title}</h2>
                            <p className="text-xs text-slate-400">Yapay Zeka Destekli Analiz</p>
                        </div>
                    </div>
                    <button
                        onClick={onClose}
                        className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Quick Actions */}
                <div className="p-3 border-b border-slate-700/50 bg-slate-800/30">
                    <div className="flex flex-wrap gap-2">
                        <button
                            onClick={() => handleQuickAction('summary')}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-blue-600/20 text-blue-400 text-xs font-medium hover:bg-blue-600/30 transition-colors"
                        >
                            <Sparkles className="w-3.5 h-3.5" />
                            Tehdit Özeti
                        </button>
                        <button
                            onClick={() => handleQuickAction('critical')}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-red-600/20 text-red-400 text-xs font-medium hover:bg-red-600/30 transition-colors"
                        >
                            <AlertTriangle className="w-3.5 h-3.5" />
                            Kritik Analiz
                        </button>
                        <button
                            onClick={() => handleQuickAction('defense')}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-green-600/20 text-green-400 text-xs font-medium hover:bg-green-600/30 transition-colors"
                        >
                            <Shield className="w-3.5 h-3.5" />
                            Savunma
                        </button>
                        <button
                            onClick={() => handleQuickAction('trends')}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-orange-600/20 text-orange-400 text-xs font-medium hover:bg-orange-600/30 transition-colors"
                        >
                            <TrendingUp className="w-3.5 h-3.5" />
                            Trend
                        </button>
                        <button
                            onClick={() => handleQuickAction('investigate')}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-purple-600/20 text-purple-400 text-xs font-medium hover:bg-purple-600/30 transition-colors"
                        >
                            <Search className="w-3.5 h-3.5" />
                            IP Araştır
                        </button>
                    </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {messages.map((msg, idx) => (
                        <div
                            key={idx}
                            className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
                        >
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${msg.role === 'user'
                                    ? 'bg-blue-600'
                                    : 'bg-gradient-to-br from-purple-500 to-blue-500'
                                }`}>
                                {msg.role === 'user'
                                    ? <User className="w-4 h-4 text-white" />
                                    : <Bot className="w-4 h-4 text-white" />
                                }
                            </div>
                            <div className={`max-w-[80%] p-3 rounded-2xl ${msg.role === 'user'
                                    ? 'bg-blue-600 text-white rounded-br-md'
                                    : 'bg-slate-800 text-slate-200 rounded-bl-md'
                                }`}>
                                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                            </div>
                        </div>
                    ))}

                    {loading && (
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-blue-500 flex items-center justify-center">
                                <Bot className="w-4 h-4 text-white" />
                            </div>
                            <div className="bg-slate-800 p-3 rounded-2xl rounded-bl-md">
                                <div className="flex items-center gap-2">
                                    <Loader2 className="w-4 h-4 text-purple-400 animate-spin" />
                                    <span className="text-sm text-slate-400">Analiz ediliyor...</span>
                                </div>
                            </div>
                        </div>
                    )}

                    <div ref={messagesEndRef} />
                </div>

                {/* Input */}
                <div className="p-4 border-t border-slate-700 bg-slate-800/50">
                    <div className="flex gap-2">
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleSend()}
                            placeholder="Bir soru sorun..."
                            className="flex-1 bg-slate-700 border border-slate-600 rounded-xl px-4 py-3 text-white placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                            disabled={loading}
                        />
                        <button
                            onClick={handleSend}
                            disabled={loading || !input.trim()}
                            className="px-4 py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white rounded-xl hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                        >
                            <Send className="w-5 h-5" />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
