import { useState, useCallback } from 'react';
import { NavLink } from 'react-router-dom';
import {
    LayoutDashboard,
    Network,
    Shield,
    Bot,
    Brain,
    Moon,
    Sun,
    Menu,
    X,
    Settings,
    Activity,
    FileText,
    FileBarChart,
    BarChart3,
    Database,
    HelpCircle,
    Radar,
    Zap,
    Target,
    Sparkles,
    Globe,
    Bell,
    Eye,
    Box,
    FileSearch,
    Link2,
    Workflow,
    AlertTriangle,
    Fingerprint,
    Server,
    Layers,
    Crosshair,
    Lightbulb,
    ShieldCheck,
    BookOpen,
    ChevronDown,
    ChevronRight
} from 'lucide-react';
import { useThemeStore } from '../store';

// 1️⃣ Genel Bakış
const overviewItems = [
    { path: '/', icon: LayoutDashboard, label: 'Kontrol Paneli' },
    { path: '/network', icon: Network, label: 'Ağ İzleme' },
    { path: '/notifications', icon: Bell, label: 'Bildirimler' },
];

// 2️⃣ Tehdit İstihbaratı
const threatIntelItems = [
    { path: '/scanner', icon: Shield, label: 'Zararlı Tarayıcı' },
    { path: '/attack-map', icon: Globe, label: 'Saldırı Haritası', highlight: true },
    { path: '/incidents', icon: AlertTriangle, label: 'Olaylar' },
    { path: '/threat-intel', icon: Radar, label: 'Tehdit İstihbaratı' },
    { path: '/darkweb', icon: Eye, label: 'Dark Web İzleme', highlight: true },
];

// 3️⃣ AI & ML
const aimlItems = [
    { path: '/aiml-hub', icon: Sparkles, label: 'AI/ML Hub', highlight: true },
    { path: '/models', icon: Brain, label: 'ML Modeller' },
    { path: '/advanced-models', icon: Zap, label: 'Gelişmiş Modeller' },
    { path: '/xai', icon: Lightbulb, label: 'XAI Açıklayıcı' },
    { path: '/automl', icon: Workflow, label: 'AutoML Pipeline' },
    { path: '/attack-training', icon: Target, label: 'Saldırı Eğitimi' },
];

// 4️⃣ Güvenlik Araçları
const securityToolsItems = [
    { path: '/security-hub', icon: ShieldCheck, label: 'Güvenlik Merkezi' },
    { path: '/vuln-scanner', icon: FileSearch, label: 'Zafiyet Tarama' },
    { path: '/threat-hunting', icon: Crosshair, label: 'Tehdit Avı' },
    { path: '/sandbox', icon: Fingerprint, label: 'Malware Sandbox' },
    { path: '/playbooks', icon: BookOpen, label: 'Playbooks' },
];

// 5️⃣ Altyapı
const infrastructureItems = [
    { path: '/container', icon: Box, label: 'Container Güvenlik' },
    { path: '/siem', icon: Server, label: 'SIEM Entegrasyon' },
    { path: '/blockchain', icon: Link2, label: 'Blockchain Audit' },
    { path: '/deception', icon: Layers, label: 'Deception Tech' },
];

// 6️⃣ Analiz & Raporlar
const analyticsItems = [
    { path: '/analytics', icon: BarChart3, label: 'Analitik' },
    { path: '/reports', icon: FileBarChart, label: 'Raporlar' },
    { path: '/logs', icon: FileText, label: 'Loglar' },
    { path: '/database', icon: Database, label: 'Veritabanı' },
];

// 7️⃣ Sistem (Footer)
const systemItems = [
    { path: '/assistant', icon: Bot, label: 'AI Asistan' },
    { path: '/settings', icon: Settings, label: 'Ayarlar' },
    { path: '/help', icon: HelpCircle, label: 'Yardım' },
];

// NavItem component - dışarıda tanımlandı
const NavItem = ({ item, collapsed }) => (
    <NavLink
        to={item.path}
        className={({ isActive }) => `
            flex items-center gap-3 px-3 py-2 rounded-xl
            transition-all duration-200 group
            ${isActive
                ? 'bg-gradient-to-r from-blue-600/20 to-purple-600/20 text-blue-400 border border-blue-500/30 shadow-lg shadow-blue-500/10'
                : 'text-slate-400 hover:bg-slate-800/50 hover:text-slate-200'
            }
            ${item.highlight ? 'ring-1 ring-purple-500/30' : ''}
        `}
    >
        <item.icon className={`w-5 h-5 flex-shrink-0 ${collapsed ? 'mx-auto' : ''}`} />
        {!collapsed && <span className="font-medium text-sm">{item.label}</span>}
    </NavLink>
);

// CollapsibleSection component - dışarıda tanımlandı
const CollapsibleSection = ({ id, title, items, color = 'text-slate-500', emoji = '', isOpen, onToggle, collapsed }) => (
    <div>
        {!collapsed ? (
            <button
                onClick={() => onToggle(id)}
                className={`w-full px-3 py-1.5 flex items-center justify-between text-xs font-semibold ${color} uppercase tracking-wider hover:bg-slate-800/30 rounded-lg transition-colors`}
            >
                <span className="flex items-center gap-1">
                    {emoji} {title}
                </span>
                {isOpen ? (
                    <ChevronDown className="w-4 h-4" />
                ) : (
                    <ChevronRight className="w-4 h-4" />
                )}
            </button>
        ) : (
            <div className="w-full h-px bg-slate-700/50 my-2" />
        )}

        <div className={`space-y-1 overflow-hidden transition-all duration-300 ${isOpen || collapsed ? 'max-h-96 opacity-100 mt-1' : 'max-h-0 opacity-0'
            }`}>
            {items.map((item) => (
                <NavItem key={item.path} item={item} collapsed={collapsed} />
            ))}
        </div>
    </div>
);

export default function Sidebar() {
    const [collapsed, setCollapsed] = useState(false);
    const { isDark, toggleTheme } = useThemeStore();

    // Collapsible section states - varsayılan olarak sadece Genel Bakış açık
    const [openSections, setOpenSections] = useState({
        overview: true,
        threatIntel: false,
        aiml: false,
        securityTools: false,
        infrastructure: false,
        analytics: false,
    });

    const toggleSection = useCallback((section) => {
        setOpenSections(prev => ({
            ...prev,
            [section]: !prev[section]
        }));
    }, []);

    return (
        <aside className={`
            fixed left-0 top-0 h-screen
            ${collapsed ? 'w-20' : 'w-64'}
            bg-slate-900/95 backdrop-blur-xl
            border-r border-slate-700/50
            transition-all duration-300 z-50
            flex flex-col
        `}>
            {/* Header */}
            <div className="p-4 flex items-center justify-between border-b border-slate-700/50">
                {!collapsed && (
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                            <Shield className="w-6 h-6 text-white" />
                        </div>
                        <div>
                            <span className="text-lg font-bold gradient-text">CyberGuard</span>
                            <p className="text-[10px] text-slate-500 uppercase tracking-wider">AI Güvenlik</p>
                        </div>
                    </div>
                )}
                {collapsed && (
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mx-auto">
                        <Shield className="w-6 h-6 text-white" />
                    </div>
                )}
                <button
                    onClick={() => setCollapsed(!collapsed)}
                    className={`p-2 rounded-lg hover:bg-slate-800 transition-colors ${collapsed ? 'mx-auto mt-2' : ''}`}
                >
                    {collapsed ? <Menu className="w-5 h-5 text-slate-400" /> : <X className="w-5 h-5 text-slate-400" />}
                </button>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-3 space-y-3 overflow-y-auto">
                <CollapsibleSection
                    id="overview"
                    title="Genel Bakış"
                    items={overviewItems}
                    emoji="📊"
                    isOpen={openSections.overview}
                    onToggle={toggleSection}
                    collapsed={collapsed}
                />
                <CollapsibleSection
                    id="threatIntel"
                    title="Tehdit İstihbaratı"
                    items={threatIntelItems}
                    color="text-red-500"
                    emoji="🎯"
                    isOpen={openSections.threatIntel}
                    onToggle={toggleSection}
                    collapsed={collapsed}
                />
                <CollapsibleSection
                    id="aiml"
                    title="AI & ML"
                    items={aimlItems}
                    color="text-purple-500"
                    emoji="🧠"
                    isOpen={openSections.aiml}
                    onToggle={toggleSection}
                    collapsed={collapsed}
                />
                <CollapsibleSection
                    id="securityTools"
                    title="Güvenlik Araçları"
                    items={securityToolsItems}
                    color="text-orange-500"
                    emoji="🛡️"
                    isOpen={openSections.securityTools}
                    onToggle={toggleSection}
                    collapsed={collapsed}
                />
                <CollapsibleSection
                    id="infrastructure"
                    title="Altyapı"
                    items={infrastructureItems}
                    color="text-cyan-500"
                    emoji="🏗️"
                    isOpen={openSections.infrastructure}
                    onToggle={toggleSection}
                    collapsed={collapsed}
                />
                <CollapsibleSection
                    id="analytics"
                    title="Analiz & Raporlar"
                    items={analyticsItems}
                    emoji="📈"
                    isOpen={openSections.analytics}
                    onToggle={toggleSection}
                    collapsed={collapsed}
                />
            </nav>

            {/* Footer */}
            <div className="p-3 border-t border-slate-700/50 space-y-1">
                {/* System Nav Items */}
                {systemItems.map((item) => (
                    <NavItem key={item.path} item={item} collapsed={collapsed} />
                ))}

                {/* Theme Toggle */}
                <button
                    onClick={toggleTheme}
                    className={`
                        w-full flex items-center gap-3 px-3 py-2 rounded-xl
                        text-slate-400 hover:bg-slate-800 hover:text-slate-200
                        transition-all duration-200
                    `}
                >
                    {isDark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                    {!collapsed && <span className="text-sm">{isDark ? 'Açık Tema' : 'Koyu Tema'}</span>}
                </button>

                {/* Status */}
                {!collapsed && (
                    <div className="px-3 py-2 bg-gradient-to-r from-emerald-500/10 to-green-500/10 rounded-xl border border-emerald-500/20">
                        <div className="flex items-center gap-2">
                            <Activity className="w-4 h-4 text-emerald-400" />
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                                <span className="text-emerald-400 text-sm font-medium">Sistem Aktif</span>
                            </div>
                        </div>
                    </div>
                )}

                {collapsed && (
                    <div className="flex justify-center py-2">
                        <div className="w-3 h-3 bg-emerald-500 rounded-full animate-pulse" />
                    </div>
                )}
            </div>
        </aside>
    );
}
