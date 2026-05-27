import { useState, useRef, useEffect } from 'react';
import { Palette, Check } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useThemeStore, THEMES } from '../../hooks/useTheme';

const themePreviewColors = {
    gotham: { bg: '#060a14', accent: 'var(--hud-cyan)', border: 'rgba(56,189,248,0.25)' },
    midnight: { bg: '#0a1628', accent: '#448aff', border: 'rgba(68,138,255,0.25)' },
    matrix: { bg: '#050f05', accent: '#10b981', border: 'rgba(16,185,129,0.25)' },
    redteam: { bg: '#140808', accent: '#ef4444', border: 'rgba(239,68,68,0.25)' },
};

export default function ThemeToggle() {
    const { theme, setTheme } = useThemeStore();
    const [open, setOpen] = useState(false);
    const ref = useRef(null);

    useEffect(() => {
        const handler = (e) => {
            if (ref.current && !ref.current.contains(e.target)) setOpen(false);
        };
        document.addEventListener('mousedown', handler);
        return () => document.removeEventListener('mousedown', handler);
    }, []);

    return (
        <div className="relative" ref={ref}>
            <button
                onClick={() => setOpen(!open)}
                className="flex items-center gap-1.5 px-2 py-1 rounded border border-[var(--hud-border)] text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)] hover:border-[var(--hud-border-strong)] transition-all text-[10px]"
                title="Tema Değiştir"
            >
                <Palette className="w-3.5 h-3.5" />
                <span className="hidden sm:inline">{THEMES[theme]?.label || 'TEMA'}</span>
            </button>

            <AnimatePresence>
                {open && (
                    <motion.div
                        initial={{ opacity: 0, y: -8, scale: 0.95 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: -8, scale: 0.95 }}
                        transition={{ duration: 0.15 }}
                        className="absolute right-0 top-full mt-2 w-56 bg-[var(--hud-surface-elevated)] backdrop-blur-xl border border-[var(--hud-border)] rounded-lg shadow-2xl overflow-hidden z-50"
                    >
                        <div className="px-3 py-2 border-b border-[var(--hud-border)]">
                            <span className="text-[9px] text-[var(--hud-text-dim)] tracking-wide">TEMA SECIMI</span>
                        </div>
                        {Object.entries(THEMES).map(([key, t]) => {
                            const colors = themePreviewColors[key];
                            const isActive = theme === key;
                            return (
                                <button
                                    key={key}
                                    onClick={() => { setTheme(key); setOpen(false); }}
                                    className={`w-full flex items-center gap-3 px-3 py-2.5 text-left transition-all hover:bg-[rgba(255,255,255,0.03)] ${isActive ? 'bg-[rgba(255,255,255,0.05)]' : ''}`}
                                >
                                    {/* Color preview swatch */}
                                    <div
                                        className="w-7 h-7 rounded-md border flex items-center justify-center shrink-0"
                                        style={{ background: colors.bg, borderColor: colors.border }}
                                    >
                                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: colors.accent, boxShadow: `0 0 8px ${colors.accent}60` }} />
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <div className="text-[11px] text-[var(--hud-text)]">{t.name}</div>
                                        <div className="text-[9px] text-[var(--hud-text-dim)]">{t.description}</div>
                                    </div>
                                    {isActive && <Check className="w-3.5 h-3.5 text-[var(--hud-cyan)] shrink-0" />}
                                </button>
                            );
                        })}
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}
