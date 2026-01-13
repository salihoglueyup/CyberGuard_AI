import { useState, useEffect } from 'react';
import {
    Settings as SettingsIcon, Key, Palette, Globe, Bell, Info,
    Shield, Database, Save, Eye, EyeOff, Check, RefreshCw, Loader2
} from 'lucide-react';
import { Card, Button, Dropdown, Alert, Badge } from '../components/ui';
import { useThemeStore } from '../store';
import { useToast } from '../components/ui/Toast';

const API_BASE = 'http://localhost:8000/api';

export default function Settings() {
    const { isDark, toggleTheme } = useThemeStore();
    const toast = useToast();

    const [loading, setLoading] = useState(true);
    const [saving, setSaving] = useState(false);
    const [testing, setTesting] = useState({});

    const [apiKeys, setApiKeys] = useState({
        gemini: '',
        openai: '',
        virustotal: '',
    });
    const [showKeys, setShowKeys] = useState({});
    const [language, setLanguage] = useState('tr');
    const [notifications, setNotifications] = useState({
        threats: true,
        training: true,
        system: false,
        email: false,
    });
    const [activeTab, setActiveTab] = useState('api');

    // Ayarları yükle
    useEffect(() => {
        loadSettings();
    }, []);

    const loadSettings = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/settings`);
            const data = await res.json();

            if (data.success) {
                const settings = data.data;
                setApiKeys(settings.api_keys || {});
                setNotifications(settings.notifications || {});
                setLanguage(settings.language || 'tr');
            }
        } catch (error) {
            console.error('Settings load error:', error);
        } finally {
            setLoading(false);
        }
    };

    const handleSave = async () => {
        setSaving(true);
        try {
            const res = await fetch(`${API_BASE}/settings`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    api_keys: apiKeys,
                    notifications,
                    language,
                    appearance: { theme: isDark ? 'dark' : 'light' }
                })
            });
            const data = await res.json();

            if (data.success) {
                toast.success('Ayarlar başarıyla kaydedildi!');
            } else {
                toast.error(data.error || 'Kaydetme hatası');
            }
        } catch (error) {
            toast.error('Bağlantı hatası');
        } finally {
            setSaving(false);
        }
    };

    const testApiKey = async (keyName) => {
        setTesting(prev => ({ ...prev, [keyName]: true }));
        try {
            const res = await fetch(`${API_BASE}/settings/test-api-key/${keyName}`, {
                method: 'POST'
            });
            const data = await res.json();

            if (data.success) {
                toast.success(data.message);
            } else {
                toast.error(data.error);
            }
        } catch (error) {
            toast.error('Test hatası');
        } finally {
            setTesting(prev => ({ ...prev, [keyName]: false }));
        }
    };

    const tabs = [
        { id: 'api', label: 'API Anahtarları', icon: Key },
        { id: 'appearance', label: 'Görünüm', icon: Palette },
        { id: 'notifications', label: 'Bildirimler', icon: Bell },
        { id: 'system', label: 'Sistem', icon: Info },
    ];

    if (loading) {
        return (
            <div className="flex items-center justify-center h-96">
                <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
            </div>
        );
    }

    return (
        <div className="p-6 space-y-6 fade-in">
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                        <SettingsIcon className="w-7 h-7 text-blue-400" />
                        Ayarlar
                    </h1>
                    <p className="text-slate-400 mt-1">Uygulama ayarlarını ve tercihlerinizi yönetin</p>
                </div>
                <Button variant="primary" icon={Save} onClick={handleSave} loading={saving}>
                    Kaydet
                </Button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar Tabs */}
                <div className="space-y-1">
                    {tabs.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`
                                w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all
                                ${activeTab === tab.id
                                    ? 'bg-blue-600/20 text-blue-400 border border-blue-500/30'
                                    : 'text-slate-400 hover:bg-slate-800 hover:text-white'
                                }
                            `}
                        >
                            <tab.icon className="w-5 h-5" />
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Content Area */}
                <div className="lg:col-span-3 space-y-6">
                    {/* API Keys */}
                    {activeTab === 'api' && (
                        <Card>
                            <Card.Header>
                                <Card.Title icon={Key}>API Anahtarları</Card.Title>
                                <p className="text-sm text-slate-400 mt-1">Harici servis API anahtarlarınızı yapılandırın</p>
                            </Card.Header>
                            <Card.Body className="space-y-6">
                                <Alert type="warning">
                                    API anahtarlarınızı güvende tutun. Asla herkese açık paylaşmayın.
                                </Alert>

                                {/* Gemini API */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        Google Gemini API Anahtarı
                                        <Badge variant="success" size="sm" className="ml-2">Chatbot için Gerekli</Badge>
                                    </label>
                                    <div className="flex gap-2">
                                        <div className="flex-1 relative">
                                            <input
                                                type={showKeys.gemini ? 'text' : 'password'}
                                                value={apiKeys.gemini || ''}
                                                onChange={(e) => setApiKeys({ ...apiKeys, gemini: e.target.value })}
                                                placeholder="AIza..."
                                                className="input pr-10"
                                            />
                                            <button
                                                onClick={() => setShowKeys({ ...showKeys, gemini: !showKeys.gemini })}
                                                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                                            >
                                                {showKeys.gemini ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                            </button>
                                        </div>
                                        <Button
                                            variant="secondary"
                                            onClick={() => testApiKey('gemini')}
                                            disabled={testing.gemini}
                                        >
                                            {testing.gemini ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Test Et'}
                                        </Button>
                                    </div>
                                    <p className="text-xs text-slate-500 mt-1">
                                        Anahtarınızı <a href="https://makersuite.google.com/app/apikey" target="_blank" rel="noreferrer" className="text-blue-400 hover:underline">Google AI Studio</a>'dan alabilirsiniz
                                    </p>
                                </div>

                                {/* OpenAI API */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        OpenAI API Anahtarı
                                        <Badge variant="info" size="sm" className="ml-2">İsteğe Bağlı</Badge>
                                    </label>
                                    <div className="flex gap-2">
                                        <div className="flex-1 relative">
                                            <input
                                                type={showKeys.openai ? 'text' : 'password'}
                                                value={apiKeys.openai || ''}
                                                onChange={(e) => setApiKeys({ ...apiKeys, openai: e.target.value })}
                                                placeholder="sk-..."
                                                className="input pr-10"
                                            />
                                            <button
                                                onClick={() => setShowKeys({ ...showKeys, openai: !showKeys.openai })}
                                                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                                            >
                                                {showKeys.openai ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                            </button>
                                        </div>
                                        <Button variant="secondary" onClick={() => testApiKey('openai')} disabled={testing.openai}>
                                            {testing.openai ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Test Et'}
                                        </Button>
                                    </div>
                                </div>

                                {/* VirusTotal API */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">
                                        VirusTotal API Anahtarı
                                        <Badge variant="info" size="sm" className="ml-2">Zararlı Tarama için</Badge>
                                    </label>
                                    <div className="flex gap-2">
                                        <div className="flex-1 relative">
                                            <input
                                                type={showKeys.virustotal ? 'text' : 'password'}
                                                value={apiKeys.virustotal || ''}
                                                onChange={(e) => setApiKeys({ ...apiKeys, virustotal: e.target.value })}
                                                placeholder="API anahtarı girin..."
                                                className="input pr-10"
                                            />
                                            <button
                                                onClick={() => setShowKeys({ ...showKeys, virustotal: !showKeys.virustotal })}
                                                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                                            >
                                                {showKeys.virustotal ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                                            </button>
                                        </div>
                                        <Button variant="secondary" onClick={() => testApiKey('virustotal')} disabled={testing.virustotal}>
                                            {testing.virustotal ? <Loader2 className="w-4 h-4 animate-spin" /> : 'Test Et'}
                                        </Button>
                                    </div>
                                </div>
                            </Card.Body>
                        </Card>
                    )}

                    {/* Appearance */}
                    {activeTab === 'appearance' && (
                        <Card>
                            <Card.Header>
                                <Card.Title icon={Palette}>Görünüm</Card.Title>
                                <p className="text-sm text-slate-400 mt-1">Görünümü ve hissi özelleştirin</p>
                            </Card.Header>
                            <Card.Body className="space-y-6">
                                {/* Theme */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-3">Tema</label>
                                    <div className="grid grid-cols-2 gap-4">
                                        <button
                                            onClick={() => !isDark && toggleTheme()}
                                            className={`p-4 rounded-xl border-2 transition-all ${isDark ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-slate-600'
                                                }`}
                                        >
                                            <div className="w-full h-20 rounded-lg bg-slate-900 border border-slate-700 mb-3" />
                                            <p className="font-medium text-white">Koyu Tema</p>
                                            <p className="text-xs text-slate-400">Gözlere daha rahat</p>
                                            {isDark && <Check className="w-5 h-5 text-blue-400 mt-2" />}
                                        </button>
                                        <button
                                            onClick={() => isDark && toggleTheme()}
                                            className={`p-4 rounded-xl border-2 transition-all ${!isDark ? 'border-blue-500 bg-blue-500/10' : 'border-slate-700 hover:border-slate-600'
                                                }`}
                                        >
                                            <div className="w-full h-20 rounded-lg bg-white border border-slate-200 mb-3" />
                                            <p className="font-medium text-white">Açık Tema</p>
                                            <p className="text-xs text-slate-400">Klasik aydınlık tema</p>
                                            {!isDark && <Check className="w-5 h-5 text-blue-400 mt-2" />}
                                        </button>
                                    </div>
                                </div>

                                {/* Language */}
                                <div>
                                    <label className="block text-sm font-medium text-slate-300 mb-2">Dil</label>
                                    <Dropdown
                                        value={language}
                                        onChange={setLanguage}
                                        icon={Globe}
                                        options={[
                                            { value: 'tr', label: '🇹🇷 Türkçe' },
                                            { value: 'en', label: '🇬🇧 English' },
                                            { value: 'de', label: '🇩🇪 Deutsch' },
                                        ]}
                                    />
                                </div>
                            </Card.Body>
                        </Card>
                    )}

                    {/* Notifications */}
                    {activeTab === 'notifications' && (
                        <Card>
                            <Card.Header>
                                <Card.Title icon={Bell}>Bildirimler</Card.Title>
                                <p className="text-sm text-slate-400 mt-1">Hangi bildirimleri alacağınızı seçin</p>
                            </Card.Header>
                            <Card.Body className="space-y-4">
                                {[
                                    { key: 'threats', label: 'Tehdit Uyarıları', desc: 'Yeni tehditler tespit edildiğinde bildirim alın' },
                                    { key: 'training', label: 'Eğitim Güncellemeleri', desc: 'Model eğitimi tamamlandığında bildirim' },
                                    { key: 'system', label: 'Sistem Durumu', desc: 'Sistem sağlığı ve performans güncellemeleri' },
                                    { key: 'email', label: 'E-posta Bildirimleri', desc: 'Önemli uyarıları e-posta ile alın' },
                                ].map((item) => (
                                    <div key={item.key} className="flex items-center justify-between p-4 bg-slate-800/50 rounded-xl">
                                        <div>
                                            <p className="font-medium text-white">{item.label}</p>
                                            <p className="text-sm text-slate-400">{item.desc}</p>
                                        </div>
                                        <button
                                            onClick={() => setNotifications({ ...notifications, [item.key]: !notifications[item.key] })}
                                            className={`relative w-12 h-6 rounded-full transition-colors ${notifications[item.key] ? 'bg-blue-600' : 'bg-slate-700'
                                                }`}
                                        >
                                            <div className={`absolute top-1 w-4 h-4 rounded-full bg-white transition-transform ${notifications[item.key] ? 'translate-x-7' : 'translate-x-1'
                                                }`} />
                                        </button>
                                    </div>
                                ))}
                            </Card.Body>
                        </Card>
                    )}

                    {/* System */}
                    {activeTab === 'system' && (
                        <div className="space-y-6">
                            <Card>
                                <Card.Header>
                                    <Card.Title icon={Info}>Sistem Bilgisi</Card.Title>
                                </Card.Header>
                                <Card.Body>
                                    <div className="grid grid-cols-2 gap-4">
                                        {[
                                            { label: 'Versiyon', value: 'v2.0.0' },
                                            { label: 'Frontend', value: 'React 18 + Vite' },
                                            { label: 'Backend', value: 'FastAPI' },
                                            { label: 'Veritabanı', value: 'SQLite' },
                                            { label: 'ML Framework', value: 'TensorFlow / Keras' },
                                            { label: 'AI Model', value: 'Google Gemini Pro' },
                                        ].map((item) => (
                                            <div key={item.label} className="p-3 bg-slate-800/50 rounded-lg">
                                                <p className="text-xs text-slate-400">{item.label}</p>
                                                <p className="font-medium text-white">{item.value}</p>
                                            </div>
                                        ))}
                                    </div>
                                </Card.Body>
                            </Card>

                            <Card>
                                <Card.Header>
                                    <Card.Title icon={Database}>Veritabanı</Card.Title>
                                </Card.Header>
                                <Card.Body className="space-y-4">
                                    <div className="flex items-center justify-between p-4 bg-slate-800/50 rounded-xl">
                                        <div>
                                            <p className="font-medium text-white">Veritabanı Yolu</p>
                                            <p className="text-sm text-slate-400 font-mono">src/database/cyberguard.db</p>
                                        </div>
                                        <Badge variant="success" dot>Bağlı</Badge>
                                    </div>
                                    <div className="flex gap-4">
                                        <Button variant="secondary" icon={RefreshCw}>Veritabanını Optimize Et</Button>
                                        <Button variant="danger">Veritabanını Sıfırla</Button>
                                    </div>
                                </Card.Body>
                            </Card>

                            <Card>
                                <Card.Header>
                                    <Card.Title icon={Shield}>Güvenlik</Card.Title>
                                </Card.Header>
                                <Card.Body>
                                    <Alert type="success">
                                        Tüm sistemler güvenli bir şekilde çalışıyor. Son güvenlik taraması: Bugün
                                    </Alert>
                                </Card.Body>
                            </Card>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
