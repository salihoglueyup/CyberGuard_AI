import { useState, useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Search, Command, User, ChevronDown, LogOut, Settings, HelpCircle, Activity } from 'lucide-react';
import NotificationBell from './NotificationBell';
import ThemeToggle from './shared/ThemeToggle';
import LanguageSwitcher from './shared/LanguageSwitcher';

const pageTitles = {
    '/': 'Kontrol Paneli',
    '/network': 'Ağ İzleme',
    '/scanner': 'Zararlı Tarayıcı',
    '/assistant': 'AI Asistan',
    '/models': 'ML Modeller',
    '/settings': 'Ayarlar',
    '/attack-map': 'Saldırı Haritası',
    '/incidents': 'Olaylar',
    '/threat-intel': 'Tehdit İstihbaratı',
    '/analytics': 'Analitik',
    '/reports': 'Raporlar',
    '/aiml-hub': 'AI/ML Hub',
    '/security-hub': 'Güvenlik Merkezi',
    '/darkweb': 'Dark Web İzleme',
};

export default function Header() {
    const location = useLocation();
    const navigate = useNavigate();
    const [searchQuery, setSearchQuery] = useState('');
    const [showUserMenu, setShowUserMenu] = useState(false);
    const [clock, setClock] = useState('');
    const menuRef = useRef(null);

    const currentPage = pageTitles[location.pathname] || 'CyberGuard AI';

    // Live clock
    useEffect(() => {
        const tick = () => {
            const now = new Date();
            setClock(now.toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }));
        };
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, []);

    // Close menu on outside click
    useEffect(() => {
        const handleClickOutside = (e) => {
            if (menuRef.current && !menuRef.current.contains(e.target)) {
                setShowUserMenu(false);
            }
        };
        if (showUserMenu) document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [showUserMenu]);

    const handleLogout = () => {
        sessionStorage.removeItem('token');
        sessionStorage.removeItem('user');
        navigate('/login');
    };

    return (
        <header className="h-14 bg-[var(--hud-surface)] border-b border-[var(--hud-border)] flex items-center justify-between px-5 sticky top-0 z-40">
            {/* Left: Breadcrumb-style title */}
            <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 text-[13px]">
                    <span className="text-[var(--hud-text-dim)]">CyberGuard</span>
                    <span className="text-[var(--hud-border-strong)]">/</span>
                    <span className="text-[var(--hud-text-bright)] font-medium">{currentPage}</span>
                </div>
            </div>

            {/* Center: Compact search */}
            <div className="flex-1 max-w-xs mx-6">
                <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--hud-text-dim)]" />
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Ara..."
                        className="w-full pl-8 pr-14 py-1.5 bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg text-[13px] text-[var(--hud-text)] placeholder:text-[var(--hud-text-dim)] focus:outline-none focus:border-[var(--hud-cyan)] focus:ring-2 focus:ring-[rgba(56,189,248,0.1)] transition-all"
                    />
                    <div className="absolute right-2 top-1/2 -translate-y-1/2 hidden md:flex items-center gap-0.5 text-[var(--hud-text-dim)]">
                        <kbd className="text-[9px] px-1 py-0.5 bg-[rgba(255,255,255,0.04)] border border-[var(--hud-border)] rounded font-mono">Ctrl</kbd>
                        <kbd className="text-[9px] px-1 py-0.5 bg-[rgba(255,255,255,0.04)] border border-[var(--hud-border)] rounded font-mono">K</kbd>
                    </div>
                </div>
            </div>

            {/* Right: Status, notifications, user */}
            <div className="flex items-center gap-3">
                {/* Live clock */}
                <div className="hidden md:flex items-center gap-1.5 text-[12px] text-[var(--hud-text-muted)] font-mono">
                    <Activity className="w-3 h-3 text-[var(--hud-emerald)]" />
                    <span>{clock}</span>
                </div>

                <div className="w-px h-4 bg-[var(--hud-border)]" />

                {/* Theme Toggle */}
                <ThemeToggle />

                {/* Language */}
                <LanguageSwitcher compact />

                {/* Notifications */}
                <NotificationBell />

                {/* User Menu */}
                <div className="relative" ref={menuRef}>
                    <button
                        onClick={() => setShowUserMenu(!showUserMenu)}
                        className="flex items-center gap-1.5 px-2 py-1 rounded-lg hover:bg-[var(--hud-cyan-ghost)] transition-colors"
                    >
                        <div className="w-7 h-7 rounded-lg border border-[var(--hud-border)] flex items-center justify-center bg-[var(--hud-cyan-ghost)]">
                            <User className="w-3.5 h-3.5 text-[var(--hud-cyan)]" />
                        </div>
                        <div className="hidden md:block text-left">
                            <p className="text-[12px] text-[var(--hud-text)]">Admin</p>
                        </div>
                        <ChevronDown className={`w-3 h-3 text-[var(--hud-text-dim)] transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
                    </button>

                    {showUserMenu && (
                        <div className="absolute right-0 top-full mt-1 w-44 bg-[var(--hud-surface-elevated)] border border-[var(--hud-border)] rounded-lg shadow-[0_8px_32px_rgba(0,0,0,0.5)] overflow-hidden z-50 scale-in">
                            <div className="p-1">
                                <button
                                    onClick={() => { setShowUserMenu(false); navigate('/settings'); }}
                                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[13px] text-[var(--hud-text-muted)] hover:bg-[var(--hud-cyan-ghost)] hover:text-[var(--hud-text)] transition-colors"
                                >
                                    <Settings className="w-4 h-4" />
                                    <span>Ayarlar</span>
                                </button>
                                <button className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[13px] text-[var(--hud-text-muted)] hover:bg-[var(--hud-cyan-ghost)] hover:text-[var(--hud-text)] transition-colors">
                                    <HelpCircle className="w-4 h-4" />
                                    <span>Yardım</span>
                                </button>
                                <div className="my-1 border-t border-[var(--hud-border)]" />
                                <button
                                    onClick={handleLogout}
                                    className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-[13px] text-[var(--hud-red)] hover:bg-[rgba(239,68,68,0.08)] transition-colors"
                                >
                                    <LogOut className="w-4 h-4" />
                                    <span>Çıkış</span>
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
}
