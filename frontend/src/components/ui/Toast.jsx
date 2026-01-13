import { useEffect, useState } from 'react';
import { CheckCircle, AlertCircle, AlertTriangle, Info, X } from 'lucide-react';
import { create } from 'zustand';

// Toast Store
export const useToastStore = create((set, get) => ({
    toasts: [],
    addToast: (toast) => {
        const id = Date.now();
        const newToast = { id, ...toast };
        set((state) => ({ toasts: [...state.toasts, newToast] }));

        // Süre sonunda otomatik kaldır
        if (toast.duration !== 0) {
            setTimeout(() => {
                get().removeToast(id);
            }, toast.duration || 5000);
        }
        return id;
    },
    removeToast: (id) => {
        set((state) => ({ toasts: state.toasts.filter((t) => t.id !== id) }));
    },
    clearAll: () => set({ toasts: [] }),
}));

// Kolay kullanım için Toast hook
export function useToast() {
    const { addToast, removeToast, clearAll } = useToastStore();

    return {
        success: (message, options = {}) => addToast({ type: 'success', message, ...options }),
        error: (message, options = {}) => addToast({ type: 'error', message, ...options }),
        warning: (message, options = {}) => addToast({ type: 'warning', message, ...options }),
        info: (message, options = {}) => addToast({ type: 'info', message, ...options }),
        remove: removeToast,
        clearAll,
    };
}

const icons = {
    success: CheckCircle,
    error: AlertCircle,
    warning: AlertTriangle,
    info: Info,
};

const colors = {
    success: 'text-green-400',
    error: 'text-red-400',
    warning: 'text-yellow-400',
    info: 'text-blue-400',
};

function Toast({ toast, onRemove }) {
    const Icon = icons[toast.type] || Info;

    return (
        <div className={`toast toast-${toast.type}`}>
            <Icon className={`w-5 h-5 flex-shrink-0 ${colors[toast.type]}`} />
            <div className="flex-1 min-w-0">
                {toast.title && (
                    <p className="font-semibold text-white text-sm">{toast.title}</p>
                )}
                <p className="text-sm text-slate-300">{toast.message}</p>
            </div>
            <button
                onClick={() => onRemove(toast.id)}
                className="p-1 rounded-lg hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
            >
                <X className="w-4 h-4" />
            </button>
        </div>
    );
}

export function ToastContainer() {
    const { toasts, removeToast } = useToastStore();

    if (toasts.length === 0) return null;

    return (
        <div className="toast-container">
            {toasts.map((toast) => (
                <Toast key={toast.id} toast={toast} onRemove={removeToast} />
            ))}
        </div>
    );
}

export default Toast;
