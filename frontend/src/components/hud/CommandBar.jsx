import { useState, useEffect, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { Command, Search, ArrowRight, Globe, Shield, Brain, BarChart3, Settings, Zap, X } from 'lucide-react';

const COMMANDS = [
    { id: 'dashboard', label: 'Kontrol Paneli', path: '/', icon: BarChart3, section: 'Navigasyon' },
    { id: 'attack-map', label: 'Saldiri Haritasi', path: '/attack-map', icon: Globe, section: 'Navigasyon' },
    { id: 'scanner', label: 'Zararli Tarayici', path: '/scanner', icon: Shield, section: 'Navigasyon' },
    { id: 'models', label: 'ML Modeller', path: '/models', icon: Brain, section: 'Navigasyon' },
    { id: 'aiml-hub', label: 'AI/ML Hub', path: '/aiml-hub', icon: Zap, section: 'Navigasyon' },
    { id: 'analytics', label: 'Analitik', path: '/analytics', icon: BarChart3, section: 'Navigasyon' },
    { id: 'settings', label: 'Ayarlar', path: '/settings', icon: Settings, section: 'Sistem' },
    { id: 'darkweb', label: 'Dark Web Izleme', path: '/darkweb', icon: Globe, section: 'Navigasyon' },
    { id: 'threat-intel', label: 'Tehdit Istihbarati', path: '/threat-intel', icon: Shield, section: 'Navigasyon' },
    { id: 'security-hub', label: 'Guvenlik Merkezi', path: '/security-hub', icon: Shield, section: 'Navigasyon' },
];

export default function CommandBar() {
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState('');
    const [selectedIdx, setSelectedIdx] = useState(0);
    const inputRef = useRef(null);
    const navigate = useNavigate();

    // Ctrl+K toggle
    useEffect(() => {
        const handler = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                setOpen(prev => !prev);
                setQuery('');
                setSelectedIdx(0);
            }
            if (e.key === 'Escape') setOpen(false);
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    useEffect(() => {
        if (open && inputRef.current) inputRef.current.focus();
    }, [open]);

    const filtered = query.trim()
        ? COMMANDS.filter(c => c.label.toLowerCase().includes(query.toLowerCase()))
        : COMMANDS;

    const execute = useCallback((cmd) => {
        navigate(cmd.path);
        setOpen(false);
        setQuery('');
    }, [navigate]);

    const handleKeyDown = (e) => {
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            setSelectedIdx(prev => Math.min(prev + 1, filtered.length - 1));
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            setSelectedIdx(prev => Math.max(prev - 1, 0));
        } else if (e.key === 'Enter' && filtered[selectedIdx]) {
            execute(filtered[selectedIdx]);
        }
    };

    if (!open) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]">
            {/* Backdrop */}
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)} />

            {/* Panel */}
            <div className="relative w-full max-w-md bg-[var(--hud-bg)] border border-[var(--hud-border)] rounded-lg shadow-[0_0_40px_rgba(56,189,248,0.06)] overflow-hidden scale-in">
                {/* Input */}
                <div className="flex items-center gap-2 px-4 py-3 border-b border-[var(--hud-border)]">
                    <Search className="w-4 h-4 text-[var(--hud-cyan)]" />
                    <input
                        ref={inputRef}
                        type="text"
                        value={query}
                        onChange={(e) => { setQuery(e.target.value); setSelectedIdx(0); }}
                        onKeyDown={handleKeyDown}
                        placeholder="Komut veya sayfa ara..."
                        className="flex-1 bg-transparent text-sm text-[var(--hud-text)] placeholder:text-[var(--hud-text-dim)] outline-none font-mono"
                    />
                    <kbd className="text-[9px] px-1.5 py-0.5 bg-[rgba(255,255,255,0.04)] border border-[var(--hud-border)] rounded font-mono text-[var(--hud-text-dim)]">ESC</kbd>
                </div>

                {/* Results */}
                <div className="max-h-64 overflow-y-auto p-1">
                    {filtered.length === 0 && (
                        <div className="py-6 text-center text-[var(--hud-text-dim)] text-xs font-mono">SONUC YOK</div>
                    )}
                    {filtered.map((cmd, idx) => {
                        const Icon = cmd.icon;
                        return (
                            <button
                                key={cmd.id}
                                onClick={() => execute(cmd)}
                                className={`w-full flex items-center gap-3 px-3 py-2 rounded transition-colors font-mono text-[12px] ${
                                    idx === selectedIdx
                                        ? 'bg-[rgba(0,229,255,0.08)] text-[var(--hud-cyan)]'
                                        : 'text-[var(--hud-text-muted)] hover:bg-[rgba(0,229,255,0.04)]'
                                }`}
                            >
                                <Icon className="w-4 h-4 flex-shrink-0" />
                                <span className="flex-1 text-left">{cmd.label}</span>
                                <span className="text-[9px] text-[var(--hud-text-dim)]">{cmd.section}</span>
                                {idx === selectedIdx && <ArrowRight className="w-3 h-3" />}
                            </button>
                        );
                    })}
                </div>

                {/* Footer */}
                <div className="flex items-center justify-between px-3 py-2 border-t border-[var(--hud-border)] text-[9px] text-[var(--hud-text-dim)] font-mono">
                    <span>↑↓ gezin</span>
                    <span>↵ sec</span>
                    <span>esc kapat</span>
                </div>
            </div>
        </div>
    );
}
