import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    BookOpen, Search, ExternalLink, ChevronRight, ChevronDown,
    Shield, Brain, Network, Bot, Settings, BarChart3, FileText,
    Keyboard, HelpCircle, MessageCircle, Mail, Github, Globe
} from 'lucide-react';
import { Card, Button, Badge } from '../../components/ui';
import { SearchInput } from '../../components/ui/Input';
import { API_BASE } from '../../services/api';

const sections = [
    {
        id: 'getting-started',
        title: '🚀 Başlarken',
        icon: BookOpen,
        items: [
            { title: 'CyberGuard AI Nedir?', content: 'CyberGuard AI, makine öğrenimi tabanlı siber güvenlik platformudur. Ağ trafiğini analiz eder, zararlı yazılımları tespit eder ve yapay zeka destekli güvenlik önerileri sunar.' },
            { title: 'Kurulum', content: 'Projeyi başlatmak için run.bat dosyasını çalıştırın. Bu, hem backend (FastAPI) hem de frontend (React) sunucularını başlatır.' },
            { title: 'Sistem Gereksinimleri', content: 'Python 3.8+, Node.js 18+, 8GB RAM, GPU (isteğe bağlı, eğitim için önerilir)' },
        ]
    },
    {
        id: 'pages',
        title: '📄 Sayfalar',
        icon: Globe,
        items: [
            { title: 'Kontrol Paneli', content: 'Ana dashboard. Model istatistikleri, canlı tehdit haritası, sistem sağlığı ve hızlı işlemler burada.' },
            { title: 'Ağ İzleme', content: 'Gerçek zamanlı ağ trafiği analizi. Şüpheli aktiviteleri tespit eder.' },
            { title: 'Zararlı Tarayıcı', content: 'Dosya ve URL taraması. Virüs, trojan, ransomware tespiti.' },
            { title: 'AI Asistan', content: 'Gemini AI destekli chatbot. Güvenlik soruları sorun, analiz isteyin.' },
            { title: 'ML Modeller', content: 'Model yönetimi. Eğitim, deploy, karşılaştırma.' },
        ]
    },
    {
        id: 'features',
        title: '⚡ Özellikler',
        icon: Shield,
        items: [
            { title: 'WebSocket Real-time', content: 'Canlı tehdit akışı. Dashboard\'da anlık güncellemeler.' },
            { title: 'Tehdit Haritası', content: 'Dünya haritasında saldırı konumları. Leaflet.js ile görselleştirme.' },
            { title: 'PDF Raporlar', content: 'Dashboard ve tehdit raporlarını PDF olarak indirin.' },
            { title: 'Bildirim Sistemi', content: 'Toast bildirimleri ve notification bell ile anlık uyarılar.' },
        ]
    },
    {
        id: 'api',
        title: '🔌 API',
        icon: Network,
        items: [
            { title: 'API Dokümanı', content: `${API_BASE}/docs adresinden Swagger UI'a erişin.` },
            { title: 'Endpointler', content: '/api/dashboard, /api/attacks, /api/models, /api/chat, /api/training, /ws' },
            { title: 'WebSocket', content: `${API_BASE.replace('/api', '')}/ws - Real-time tehdit ve sistem verileri` },
        ]
    },
    {
        id: 'shortcuts',
        title: '⌨️ Kısayollar',
        icon: Keyboard,
        items: [
            { title: 'Cmd/Ctrl + K', content: 'Global arama' },
            { title: 'Cmd/Ctrl + B', content: 'Sidebar aç/kapat' },
            { title: 'Escape', content: 'Modal kapat' },
        ]
    },
];

const faqs = [
    { q: 'API anahtarımı nasıl eklerim?', a: 'Ayarlar > API Anahtarları bölümünden Gemini, OpenAI veya VirusTotal API anahtarlarınızı ekleyebilirsiniz.' },
    { q: 'Model nasıl eğitilir?', a: 'ML Modeller sayfasından "Yeni Model Eğit" butonuna tıklayın. Dataset, framework ve hiperparametreleri seçin.' },
    { q: 'WebSocket bağlantısı kopuyor?', a: 'Backend sunucusunun çalıştığından emin olun. Bağlantı otomatik olarak yeniden kurulur.' },
    { q: 'PDF rapor boş geliyor?', a: 'Dashboard\'da veri yüklenene kadar bekleyin, ardından rapor oluşturun.' },
];

export default function Help() {
    const navigate = useNavigate();
    const [searchQuery, setSearchQuery] = useState('');
    const [expandedSection, setExpandedSection] = useState('getting-started');
    const [expandedItem, setExpandedItem] = useState(null);

    const filteredSections = sections.map(section => ({
        ...section,
        items: section.items.filter(item =>
            item.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
            item.content.toLowerCase().includes(searchQuery.toLowerCase())
        )
    })).filter(section => section.items.length > 0);

    return (
        <div className="space-y-6 fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-xl font-semibold text-[var(--hud-text)] flex items-center gap-3">
                        <BookOpen className="w-6 h-6 text-[var(--hud-cyan)]" />
                        Yardım & Dokümantasyon
                    </h1>
                    <p className="text-slate-400 mt-1">CyberGuard AI kullanım kılavuzu</p>
                </div>
            </div>

            {/* Search */}
            <Card>
                <SearchInput
                    value={searchQuery}
                    onChange={setSearchQuery}
                    placeholder="Dokümantasyonda ara..."
                    className="w-full"
                />
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Documentation */}
                <div className="lg:col-span-2 space-y-4">
                    {filteredSections.map((section) => (
                        <Card key={section.id} className="p-0 overflow-hidden">
                            <button
                                onClick={() => setExpandedSection(expandedSection === section.id ? null : section.id)}
                                className="w-full flex items-center justify-between p-4 hover:bg-slate-800/50 transition-colors"
                            >
                                <div className="flex items-center gap-3">
                                    <section.icon className="w-5 h-5 text-blue-400" />
                                    <span className="font-semibold text-[var(--hud-text)]">{section.title}</span>
                                    <Badge variant="primary" size="sm">{section.items.length}</Badge>
                                </div>
                                {expandedSection === section.id ? (
                                    <ChevronDown className="w-5 h-5 text-slate-400" />
                                ) : (
                                    <ChevronRight className="w-5 h-5 text-slate-400" />
                                )}
                            </button>

                            {expandedSection === section.id && (
                                <div className="border-t border-slate-700/50">
                                    {section.items.map((item, idx) => (
                                        <div key={idx} className="border-b border-slate-800 last:border-0">
                                            <button
                                                onClick={() => setExpandedItem(expandedItem === `${section.id}-${idx}` ? null : `${section.id}-${idx}`)}
                                                className="w-full flex items-center justify-between p-4 hover:bg-slate-800/30 transition-colors"
                                            >
                                                <span className="text-slate-300">{item.title}</span>
                                                {expandedItem === `${section.id}-${idx}` ? (
                                                    <ChevronDown className="w-4 h-4 text-slate-500" />
                                                ) : (
                                                    <ChevronRight className="w-4 h-4 text-slate-500" />
                                                )}
                                            </button>
                                            {expandedItem === `${section.id}-${idx}` && (
                                                <div className="px-4 pb-4 text-slate-400 text-sm">
                                                    {item.content}
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </Card>
                    ))}
                </div>

                {/* Sidebar */}
                <div className="space-y-6">
                    {/* Quick Links */}
                    <Card>
                        <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">🔗 Hızlı Bağlantılar</h3>
                        <div className="space-y-2">
                            <a href="http://localhost:8000/api/docs" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 p-2 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-[var(--hud-text)] transition-colors">
                                <ExternalLink className="w-4 h-4" />
                                API Dokümanı
                            </a>
                            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="flex items-center gap-2 p-2 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-[var(--hud-text)] transition-colors">
                                <Github className="w-4 h-4" />
                                GitHub Repo
                            </a>
                            <a href="mailto:support@cyberguard.ai" className="flex items-center gap-2 p-2 rounded-lg hover:bg-slate-800 text-slate-300 hover:text-[var(--hud-text)] transition-colors">
                                <Mail className="w-4 h-4" />
                                Destek
                            </a>
                        </div>
                    </Card>

                    {/* FAQ */}
                    <Card>
                        <h3 className="text-lg font-semibold text-[var(--hud-text)] mb-4">❓ Sık Sorulan Sorular</h3>
                        <div className="space-y-3">
                            {faqs.map((faq, idx) => (
                                <div key={idx} className="p-3 bg-slate-800/50 rounded-lg">
                                    <p className="text-[var(--hud-text)] text-sm font-medium">{faq.q}</p>
                                    <p className="text-slate-400 text-xs mt-1">{faq.a}</p>
                                </div>
                            ))}
                        </div>
                    </Card>

                    {/* Contact */}
                    <Card className="bg-gradient-to-br from-blue-600/20 to-purple-600/20 border-blue-500/30">
                        <div className="text-center">
                            <MessageCircle className="w-10 h-10 text-blue-400 mx-auto mb-3" />
                            <h3 className="text-[var(--hud-text)] font-semibold">Yardıma mı ihtiyacın var?</h3>
                            <p className="text-slate-400 text-sm mt-1 mb-4">AI Asistan'a sor!</p>
                            <Button variant="primary" size="sm" onClick={() => navigate('/assistant')}>
                                AI Asistan'a Git
                            </Button>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
}
