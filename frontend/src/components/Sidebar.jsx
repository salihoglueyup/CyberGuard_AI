import { useState, useCallback, useMemo } from 'react';
import { NavLink } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import {
    LayoutDashboard, Network, Shield, Bot, Brain, Menu, X,
    Settings, Activity, FileText, FileBarChart, BarChart3,
    Database, HelpCircle, Radar, Zap, Target, Sparkles,
    Globe, Bell, Eye, Box, FileSearch, Link2, Workflow,
    AlertTriangle, Fingerprint, Server, Crosshair, Lightbulb,
    ShieldCheck, ChevronDown, ChevronRight, Search, Terminal,
    ClipboardCheck, Bug, Users, Lock
} from 'lucide-react';
import { useSidebarStore } from '../store';

// --- Navigation sections ---
const overviewItems = [
    { path: '/', icon: LayoutDashboard, label: 'Kontrol Paneli' },
    { path: '/network', icon: Network, label: 'Ag Izleme' },
    { path: '/notifications', icon: Bell, label: 'Bildirimler' },
    { path: '/globe', icon: Globe, label: '3D Globe', highlight: true },
    { path: '/topology', icon: Activity, label: 'Topoloji Haritasi', highlight: true },
];

const threatIntelItems = [
    { path: '/scanner', icon: Shield, label: 'Zararli Tarayici' },
    { path: '/attack-map', icon: Globe, label: 'Saldiri Haritasi', highlight: true },
    { path: '/incidents', icon: AlertTriangle, label: 'Olaylar' },
    { path: '/threat-intel', icon: Radar, label: 'Tehdit Istihbarati' },
    { path: '/darkweb', icon: Eye, label: 'Dark Web Izleme', highlight: true },
];

const aimlItems = [
    { path: '/aiml-hub', icon: Sparkles, label: 'AI/ML Hub', highlight: true },
    { path: '/models', icon: Brain, label: 'ML Modeller' },
    { path: '/advanced-models', icon: Zap, label: 'Gelismis Modeller' },
    { path: '/xai', icon: Lightbulb, label: 'XAI Aciklayici' },
    { path: '/automl', icon: Workflow, label: 'AutoML Pipeline' },
    { path: '/attack-training', icon: Target, label: 'Saldiri Egitimi' },
];

const securityToolsItems = [
    { path: '/security-hub', icon: ShieldCheck, label: 'Guvenlik Merkezi' },
    { path: '/vuln-scanner', icon: FileSearch, label: 'Zafiyet Tarama' },
    { path: '/threat-hunting', icon: Crosshair, label: 'Tehdit Avi' },
    { path: '/sandbox', icon: Fingerprint, label: 'Malware Sandbox' },
    { path: '/honeypot', icon: Bug, label: 'Honeypot', highlight: true },
    { path: '/api-security', icon: Lock, label: 'API Guvenlik', highlight: true },
    { path: '/user-behavior', icon: Users, label: 'UEBA', highlight: true },
    { path: '/pentest', icon: Target, label: 'Pentest', highlight: true },
    { path: '/compliance', icon: ClipboardCheck, label: 'Uyumluluk', highlight: true },
    { path: '/forensics', icon: FileSearch, label: 'Adli Bilisim', highlight: true },
];

const infrastructureItems = [
    { path: '/container', icon: Box, label: 'Container Guvenlik' },
    { path: '/siem', icon: Server, label: 'SIEM Entegrasyon' },
    { path: '/blockchain', icon: Link2, label: 'Blockchain Audit' },
];

const analyticsItems = [
    { path: '/analytics', icon: BarChart3, label: 'Analitik' },
    { path: '/reports', icon: FileBarChart, label: 'Raporlar' },
    { path: '/logs', icon: FileText, label: 'Loglar' },
    { path: '/database', icon: Database, label: 'Veritabani' },
];

const systemItems = [
    { path: '/assistant', icon: Bot, label: 'AI Asistan' },
    { path: '/settings', icon: Settings, label: 'Ayarlar' },
    { path: '/help', icon: HelpCircle, label: 'Yardim' },
];

// --- NavItem ---
const NavItem = ({ item, collapsed }) => (
    <NavLink
        to={item.path}
        className={({ isActive }) => `
            flex items-center gap-2.5 px-3 py-2 rounded-lg
            transition-all duration-150 group text-[13px]
            ${isActive
                ? 'bg-[var(--hud-cyan-ghost)] text-[var(--hud-cyan)] border border-[var(--hud-border-strong)]'
                : 'text-[var(--hud-text-muted)] hover:bg-[var(--hud-cyan-ghost)] hover:text-[var(--hud-text)] border border-transparent'
            }
        `}
    >
        <item.icon className={`w-4 h-4 flex-shrink-0 ${collapsed ? 'mx-auto' : ''}`} />
        {!collapsed && <span className="truncate">{item.label}</span>}
    </NavLink>
);

// --- CollapsibleSection ---
const CollapsibleSection = ({ id, title, items, isOpen, onToggle, collapsed }) => (
    <div>
        {!collapsed ? (
            <button
                onClick={() => onToggle(id)}
                className="w-full px-3 py-1.5 flex items-center justify-between text-[11px] font-semibold text-[var(--hud-text-dim)] uppercase tracking-wide hover:text-[var(--hud-text-muted)] rounded transition-colors"
            >
                <span>{title}</span>
                <motion.span animate={{ rotate: isOpen ? 180 : 0 }} transition={{ duration: 0.2 }}>
                    <ChevronDown className="w-3 h-3" />
                </motion.span>
            </button>
        ) : (
            <div className="w-6 h-px bg-[var(--hud-border)] mx-auto my-2" />
        )}
        <AnimatePresence initial={false}>
            {(isOpen || collapsed) && (
                <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2, ease: 'easeInOut' }}
                    className="space-y-0.5 overflow-hidden mt-0.5"
                >
                    {items.map(item => <NavItem key={item.path} item={item} collapsed={collapsed} />)}
                </motion.div>
            )}
        </AnimatePresence>
    </div>
);

export default function Sidebar() {
    const { collapsed, toggleCollapsed } = useSidebarStore();
    const [searchQuery, setSearchQuery] = useState('');

    const [openSections, setOpenSections] = useState({
        overview: true,
        threatIntel: false,
        aiml: false,
        securityTools: false,
        infrastructure: false,
        analytics: false,
    });

    const toggleSection = useCallback((section) => {
        setOpenSections(prev => ({ ...prev, [section]: !prev[section] }));
    }, []);

    const allSections = useMemo(() => [
        { id: 'overview', title: 'Genel Bakış', items: overviewItems },
        { id: 'threatIntel', title: 'Tehdit İstihbaratı', items: threatIntelItems },
        { id: 'aiml', title: 'AI & ML', items: aimlItems },
        { id: 'securityTools', title: 'Güvenlik Araçları', items: securityToolsItems },
        { id: 'infrastructure', title: 'Altyapı', items: infrastructureItems },
        { id: 'analytics', title: 'Analiz & Rapor', items: analyticsItems },
    ], []);

    const filteredSections = useMemo(() => {
        if (!searchQuery.trim()) return null;
        const q = searchQuery.toLowerCase();
        return allSections
            .map(section => ({
                ...section,
                items: section.items.filter(item => item.label.toLowerCase().includes(q)),
            }))
            .filter(section => section.items.length > 0);
    }, [searchQuery, allSections]);

    const isSearching = filteredSections !== null;

    return (
        <aside className={`
            fixed left-0 top-0 h-screen
            ${collapsed ? 'w-16' : 'w-56'}
            bg-[var(--hud-bg)] backdrop-blur-xl
            border-r border-[var(--hud-border)]
            transition-all duration-300 z-50
            flex flex-col
        `}>
            {/* Header */}
            <div className="p-3 flex items-center justify-between border-b border-[var(--hud-border)]">
                {!collapsed && (
                    <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-lg border border-[var(--hud-border-strong)] flex items-center justify-center bg-[var(--hud-cyan-ghost)]">
                            <Shield className="w-4 h-4 text-[var(--hud-cyan)]" />
                        </div>
                        <div>
                            <span className="text-sm font-bold text-[var(--hud-text-bright)] tracking-wide">CyberGuard</span>
                            <p className="text-[10px] text-[var(--hud-text-dim)] tracking-wide">Security Platform</p>
                        </div>
                    </div>
                )}
                {collapsed && (
                    <div className="w-8 h-8 rounded-lg border border-[var(--hud-border-strong)] flex items-center justify-center bg-[var(--hud-cyan-ghost)] mx-auto">
                        <Shield className="w-4 h-4 text-[var(--hud-cyan)]" />
                    </div>
                )}
                <button
                    onClick={toggleCollapsed}
                    className={`p-1.5 rounded-lg hover:bg-[var(--hud-cyan-ghost)] transition-colors ${collapsed ? 'mx-auto mt-1.5' : ''}`}
                >
                    {collapsed ? <Menu className="w-4 h-4 text-[var(--hud-text-muted)]" /> : <X className="w-4 h-4 text-[var(--hud-text-muted)]" />}
                </button>
            </div>

            {/* Search */}
            {!collapsed && (
                <div className="px-2 pt-2">
                    <div className="relative">
                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--hud-text-dim)]" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Ara..."
                            className="w-full pl-8 pr-3 py-1.5 bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg text-[12px] text-[var(--hud-text)] placeholder:text-[var(--hud-text-dim)] focus:outline-none focus:border-[var(--hud-cyan)] focus:ring-2 focus:ring-[rgba(56,189,248,0.1)] transition-all"
                        />
                        {searchQuery && (
                            <button onClick={() => setSearchQuery('')}
                                className="absolute right-2 top-1/2 -translate-y-1/2 text-[var(--hud-text-dim)] hover:text-[var(--hud-text)]">
                                <X className="w-3 h-3" />
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* Navigation */}
            <nav className="flex-1 p-2 space-y-2 overflow-y-auto">
                {isSearching ? (
                    filteredSections.length > 0 ? (
                        filteredSections.map(section => (
                            <div key={section.id}>
                                <div className="px-3 py-1 text-[11px] font-semibold text-[var(--hud-text-dim)] uppercase tracking-wide">
                                    {section.title}
                                </div>
                                <div className="space-y-0.5 mt-0.5">
                                    {section.items.map(item => <NavItem key={item.path} item={item} collapsed={collapsed} />)}
                                </div>
                            </div>
                        ))
                    ) : (
                        <div className="text-center py-6 text-[var(--hud-text-dim)] text-xs">
                            <Search className="w-6 h-6 mx-auto mb-2 opacity-30" />
                            <p>Sonuç bulunamadı</p>
                        </div>
                    )
                ) : (
                    <>
                        {allSections.map(section => (
                            <CollapsibleSection
                                key={section.id}
                                id={section.id}
                                title={section.title}
                                items={section.items}
                                isOpen={openSections[section.id]}
                                onToggle={toggleSection}
                                collapsed={collapsed}
                            />
                        ))}
                    </>
                )}
            </nav>

            {/* Footer */}
            <div className="p-2 border-t border-[var(--hud-border)] space-y-0.5">
                {systemItems.map(item => <NavItem key={item.path} item={item} collapsed={collapsed} />)}

                {/* Status */}
                {!collapsed && (
                    <div className="px-2 py-1.5 mt-1 bg-[rgba(16,185,129,0.04)] rounded-lg border border-[rgba(16,185,129,0.15)]">
                        <div className="flex items-center gap-2">
                            <div className="w-1.5 h-1.5 bg-[var(--hud-emerald)] rounded-full animate-pulse" />
                            <span className="text-[var(--hud-emerald)] text-[11px] font-medium">Sistem Aktif</span>
                        </div>
                    </div>
                )}
                {collapsed && (
                    <div className="flex justify-center py-1.5">
                        <div className="w-2 h-2 bg-[var(--hud-emerald)] rounded-full animate-pulse" />
                    </div>
                )}
            </div>
        </aside>
    );
}
