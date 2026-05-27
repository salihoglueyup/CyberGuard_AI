import { useEffect, useState, useCallback } from 'react';
import { CheckCircle, AlertCircle, AlertTriangle, Info, X, Volume2, VolumeX, ShieldAlert } from 'lucide-react';
import { create } from 'zustand';
import { motion, AnimatePresence } from 'framer-motion';

// Sound effects (Web Audio API)
const audioCtx = typeof window !== 'undefined' ? new (window.AudioContext || window.webkitAudioContext)() : null;

function playSound(type) {
    if (!audioCtx) return;
    try {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        gain.gain.value = 0.08;

        if (type === 'success') {
            osc.frequency.value = 880;
            osc.type = 'sine';
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.3);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.3);
        } else if (type === 'error' || type === 'critical') {
            osc.frequency.value = 220;
            osc.type = 'square';
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.5);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.5);
        } else if (type === 'warning') {
            osc.frequency.value = 440;
            osc.type = 'triangle';
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.25);
        } else {
            osc.frequency.value = 660;
            osc.type = 'sine';
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.2);
            osc.start(audioCtx.currentTime);
            osc.stop(audioCtx.currentTime + 0.2);
        }
    } catch { /* ignore audio errors */ }
}

// Toast Store
export const useToastStore = create((set, get) => ({
    toasts: [],
    soundEnabled: true,
    toggleSound: () => set(s => ({ soundEnabled: !s.soundEnabled })),
    addToast: (toast) => {
        const id = crypto.randomUUID();
        const newToast = { id, createdAt: Date.now(), ...toast };
        set((state) => ({ toasts: [...state.toasts, newToast] }));

        // Play sound
        if (get().soundEnabled && toast.sound !== false) {
            playSound(toast.type);
        }

        // Auto-remove after duration
        if (toast.duration !== 0) {
            const dur = toast.type === 'critical' ? 15000 : (toast.duration || 5000);
            setTimeout(() => {
                get().removeToast(id);
            }, dur);
        }
        return id;
    },
    removeToast: (id) => {
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
    },
    clearAll: () => set({ toasts: [] }),
}));

// Convenience hook
export function useToast() {
    const { addToast, removeToast, clearAll } = useToastStore();

    return {
        success: (message, options = {}) => addToast({ type: 'success', message, ...options }),
        error: (message, options = {}) => addToast({ type: 'error', message, ...options }),
        warning: (message, options = {}) => addToast({ type: 'warning', message, ...options }),
        info: (message, options = {}) => addToast({ type: 'info', message, ...options }),
        critical: (message, options = {}) => addToast({ type: 'critical', message, duration: 0, ...options }),
        remove: removeToast,
        clearAll,
    };
}

const icons = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
    critical: ShieldAlert,
};

const borderColors = {
    success: 'border-l-emerald-500',
    error: 'border-l-red-500',
    warning: 'border-l-amber-500',
    info: 'border-l-cyan-500',
    critical: 'border-l-red-600',
};

const iconColors = {
    success: 'text-emerald-400',
    error: 'text-red-400',
    warning: 'text-amber-400',
    info: 'text-[var(--hud-cyan)]',
    critical: 'text-red-500',
};

const glowColors = {
    success: 'shadow-emerald-500/10',
    error: 'shadow-red-500/10',
    warning: 'shadow-amber-500/10',
    info: 'shadow-cyan-500/10',
    critical: 'shadow-red-500/20',
};

function ToastProgress({ duration, type }) {
    const [width, setWidth] = useState(100);

    useEffect(() => {
        if (!duration) return;
        const start = Date.now();
        const iv = setInterval(() => {
            const elapsed = Date.now() - start;
            const remaining = Math.max(0, 100 - (elapsed / duration * 100));
            setWidth(remaining);
            if (remaining <= 0) clearInterval(iv);
        }, 50);
        return () => clearInterval(iv);
    }, [duration]);

    if (!duration) return null;

    const barColor = type === 'critical' || type === 'error' ? 'bg-red-500/60' :
        type === 'warning' ? 'bg-amber-500/60' :
        type === 'success' ? 'bg-emerald-500/60' : 'bg-cyan-500/60';

    return (
        <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-[rgba(255,255,255,0.05)]">
            <div className={`h-full ${barColor} transition-all duration-100`} style={{ width: `${width}%` }} />
        </div>
    );
}

function Toast({ toast, onRemove }) {
    const Icon = icons[toast.type] || Info;
    const isCritical = toast.type === 'critical';
    const dur = isCritical ? 0 : (toast.duration || 5000);

    return (
        <motion.div
            layout
            initial={{ x: 320, opacity: 0, scale: 0.8 }}
            animate={{ x: 0, opacity: 1, scale: 1 }}
            exit={{ x: 320, opacity: 0, scale: 0.8 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className={`relative flex items-start gap-3 px-4 py-3 bg-[var(--hud-surface-elevated)] backdrop-blur-xl border border-[var(--hud-border)] border-l-4 ${borderColors[toast.type]} rounded-lg shadow-lg ${glowColors[toast.type]} overflow-hidden ${isCritical ? 'ring-1 ring-red-500/30' : ''}`}
        >
            <div className={`mt-0.5 shrink-0 ${isCritical ? 'animate-pulse' : ''}`}>
                <Icon className={`w-5 h-5 ${iconColors[toast.type]}`} />
            </div>
            <div className="flex-1 min-w-0">
                {toast.title && (
                    <p className="font-semibold text-[var(--hud-text-bright)] text-sm">{toast.title}</p>
                )}
                <p className="text-sm text-[var(--hud-text)]">{toast.message}</p>
                {toast.action && (
                    <button onClick={toast.action.onClick} className="text-[11px] text-[var(--hud-cyan)] hover:underline mt-1">
                        {toast.action.label}
                    </button>
                )}
            </div>
            <button
                onClick={() => onRemove(toast.id)}
                className="p-1 rounded hover:bg-[rgba(255,255,255,0.05)] text-[var(--hud-text-dim)] hover:text-[var(--hud-text)] transition-colors shrink-0"
            >
                <X className="w-3.5 h-3.5" />
            </button>
            <ToastProgress duration={dur} type={toast.type} />
        </motion.div>
    );
}

export function ToastContainer() {
    const { toasts, removeToast, soundEnabled, toggleSound } = useToastStore();

    return (
        <div className="fixed top-16 right-4 z-[9999] flex flex-col gap-2 w-80 pointer-events-none">
            {/* Sound toggle - only show when toasts exist */}
            {toasts.length > 0 && (
                <div className="flex justify-end pointer-events-auto">
                    <button onClick={toggleSound} className="text-[9px] text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)] flex items-center gap-1 transition-colors">
                        {soundEnabled ? <Volume2 className="w-3 h-3" /> : <VolumeX className="w-3 h-3" />}
                    </button>
                </div>
            )}
            <AnimatePresence mode="popLayout">
                {toasts.map((toast) => (
                    <div key={toast.id} className="pointer-events-auto">
                        <Toast toast={toast} onRemove={removeToast} />
                    </div>
                ))}
            </AnimatePresence>
        </div>
    );
}

// Critical Alert Modal
export function CriticalAlertModal() {
    const { toasts, removeToast } = useToastStore();
    const criticals = toasts.filter(t => t.type === 'critical' && t.modal);

    if (criticals.length === 0) return null;
    const alert = criticals[0];

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/60 backdrop-blur-sm"
            >
                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.8, opacity: 0 }}
                    className="bg-[var(--hud-surface-elevated)] border border-red-500/30 rounded-xl p-6 max-w-md w-full mx-4 shadow-2xl"
                >
                    <div className="flex items-center gap-3 mb-4">
                        <div className="w-10 h-10 rounded-full bg-red-500/20 flex items-center justify-center animate-pulse">
                            <ShieldAlert className="w-6 h-6 text-red-500" />
                        </div>
                        <div>
                            <h3 className="text-lg font-bold text-red-400">{alert.title || 'KRİTİK UYARI'}</h3>
                            <p className="text-[10px] text-[var(--hud-text-dim)] tracking-wider">ACIL MUEDAHALE GEREKLI</p>
                        </div>
                    </div>
                    <p className="text-[var(--hud-text)] mb-6">{alert.message}</p>
                    <div className="flex gap-3 justify-end">
                        {alert.action && (
                            <button onClick={() => { alert.action.onClick(); removeToast(alert.id); }} className="px-4 py-2 bg-red-500/20 border border-red-500/40 rounded-lg text-red-400 text-sm hover:bg-red-500/30 transition-colors">
                                {alert.action.label}
                            </button>
                        )}
                        <button onClick={() => removeToast(alert.id)} className="px-4 py-2 bg-[var(--hud-surface)] border border-[var(--hud-border)] rounded-lg text-[var(--hud-text)] text-sm hover:bg-[rgba(255,255,255,0.05)] transition-colors">
                            Kapat
                        </button>
                    </div>
                </motion.div>
            </motion.div>
        </AnimatePresence>
    );
}

export default Toast;
