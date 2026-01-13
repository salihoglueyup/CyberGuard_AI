import { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Search, Command, User, ChevronDown, LogOut, Settings, HelpCircle } from 'lucide-react';
import NotificationBell from './NotificationBell';
import { SearchInput } from './ui/Input';

const pageTitles = {
    '/': 'Kontrol Paneli',
    '/network': 'Ağ İzleme',
    '/scanner': 'Zararlı Tarayıcı',
    '/assistant': 'AI Asistan',
    '/models': 'ML Modeller',
    '/settings': 'Ayarlar',
};

export default function Header() {
    const location = useLocation();
    const [searchQuery, setSearchQuery] = useState('');
    const [showUserMenu, setShowUserMenu] = useState(false);

    const currentPage = pageTitles[location.pathname] || 'CyberGuard AI';

    return (
        <header className="h-16 bg-slate-900/80 backdrop-blur-xl border-b border-slate-700/50 flex items-center justify-between px-6 sticky top-0 z-40">
            {/* Left: Page Title */}
            <div>
                <h1 className="text-xl font-bold text-white">{currentPage}</h1>
                <p className="text-xs text-slate-400">Gerçek zamanlı siber güvenlik izleme</p>
            </div>

            {/* Center: Search */}
            <div className="flex-1 max-w-md mx-8">
                <div className="relative">
                    <SearchInput
                        value={searchQuery}
                        onChange={setSearchQuery}
                        placeholder="Tehdit, model, log ara..."
                    />
                    <div className="absolute right-10 top-1/2 -translate-y-1/2 hidden md:flex items-center gap-1 text-slate-500">
                        <Command className="w-3 h-3" />
                        <span className="text-xs">K</span>
                    </div>
                </div>
            </div>

            {/* Right: Actions */}
            <div className="flex items-center gap-3">
                {/* Notifications */}
                <NotificationBell />

                {/* User Menu */}
                <div className="relative">
                    <button
                        onClick={() => setShowUserMenu(!showUserMenu)}
                        className="flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-slate-800 transition-colors"
                    >
                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white font-semibold text-sm">
                            Y
                        </div>
                        <div className="hidden md:block text-left">
                            <p className="text-sm font-medium text-white">Yönetici</p>
                            <p className="text-xs text-slate-400">Admin</p>
                        </div>
                        <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${showUserMenu ? 'rotate-180' : ''}`} />
                    </button>

                    {showUserMenu && (
                        <div className="absolute right-0 top-full mt-2 w-48 bg-slate-900/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-xl overflow-hidden z-50 scale-in">
                            <div className="p-2">
                                <a
                                    href="/settings"
                                    className="flex items-center gap-3 px-3 py-2 rounded-lg text-slate-300 hover:bg-slate-800 hover:text-white transition-colors"
                                >
                                    <Settings className="w-4 h-4" />
                                    <span className="text-sm">Ayarlar</span>
                                </a>
                                <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-slate-300 hover:bg-slate-800 hover:text-white transition-colors">
                                    <HelpCircle className="w-4 h-4" />
                                    <span className="text-sm">Yardım</span>
                                </button>
                                <div className="my-2 border-t border-slate-700/50" />
                                <button className="w-full flex items-center gap-3 px-3 py-2 rounded-lg text-red-400 hover:bg-red-500/10 transition-colors">
                                    <LogOut className="w-4 h-4" />
                                    <span className="text-sm">Çıkış Yap</span>
                                </button>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
}
