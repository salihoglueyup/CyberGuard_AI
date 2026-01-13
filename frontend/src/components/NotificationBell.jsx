import { useState, useRef, useEffect } from 'react';
import { Bell, X, Check, Trash2 } from 'lucide-react';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Notification Store
export const useNotificationStore = create(
    persist(
        (set, get) => ({
            notifications: [],
            unreadCount: 0,

            addNotification: (notification) => {
                const newNotification = {
                    id: Date.now(),
                    timestamp: new Date().toISOString(),
                    read: false,
                    ...notification,
                };
                set((state) => ({
                    notifications: [newNotification, ...state.notifications].slice(0, 50),
                    unreadCount: state.unreadCount + 1,
                }));
            },

            markAsRead: (id) => {
                set((state) => ({
                    notifications: state.notifications.map((n) =>
                        n.id === id ? { ...n, read: true } : n
                    ),
                    unreadCount: Math.max(0, state.unreadCount - 1),
                }));
            },

            markAllAsRead: () => {
                set((state) => ({
                    notifications: state.notifications.map((n) => ({ ...n, read: true })),
                    unreadCount: 0,
                }));
            },

            removeNotification: (id) => {
                set((state) => {
                    const notification = state.notifications.find((n) => n.id === id);
                    return {
                        notifications: state.notifications.filter((n) => n.id !== id),
                        unreadCount: notification && !notification.read
                            ? Math.max(0, state.unreadCount - 1)
                            : state.unreadCount,
                    };
                });
            },

            clearAll: () => set({ notifications: [], unreadCount: 0 }),
        }),
        {
            name: 'cyberguard-notifications',
        }
    )
);

// Notify helper
export function notify(type, title, message) {
    useNotificationStore.getState().addNotification({ type, title, message });
}

const typeColors = {
    success: 'text-green-400 bg-green-500/10',
    warning: 'text-yellow-400 bg-yellow-500/10',
    error: 'text-red-400 bg-red-500/10',
    info: 'text-blue-400 bg-blue-500/10',
    threat: 'text-red-400 bg-red-500/10',
};

const typeIcons = {
    success: '✅',
    warning: '⚠️',
    error: '❌',
    info: 'ℹ️',
    threat: '🛡️',
};

export default function NotificationBell() {
    const [isOpen, setIsOpen] = useState(false);
    const ref = useRef(null);
    const { notifications, unreadCount, markAsRead, markAllAsRead, removeNotification, clearAll } = useNotificationStore();

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (ref.current && !ref.current.contains(e.target)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const formatTime = (timestamp) => {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;

        if (diff < 60000) return 'Az önce';
        if (diff < 3600000) return `${Math.floor(diff / 60000)} dk önce`;
        if (diff < 86400000) return `${Math.floor(diff / 3600000)} sa önce`;
        return date.toLocaleDateString('tr-TR');
    };

    return (
        <div ref={ref} className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="relative p-2 rounded-lg hover:bg-slate-800 transition-colors"
            >
                <Bell className="w-5 h-5 text-slate-400" />
                {unreadCount > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 w-5 h-5 bg-red-500 text-white text-xs font-bold rounded-full flex items-center justify-center animate-pulse">
                        {unreadCount > 9 ? '9+' : unreadCount}
                    </span>
                )}
            </button>

            {isOpen && (
                <div className="absolute right-0 top-full mt-2 w-80 bg-slate-900/95 backdrop-blur-xl border border-slate-700/50 rounded-xl shadow-xl overflow-hidden z-50 scale-in">
                    {/* Header */}
                    <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700/50">
                        <h3 className="font-semibold text-white">Bildirimler</h3>
                        <div className="flex items-center gap-2">
                            {unreadCount > 0 && (
                                <button
                                    onClick={markAllAsRead}
                                    className="text-xs text-blue-400 hover:text-blue-300"
                                >
                                    Tümünü okundu işaretle
                                </button>
                            )}
                            {notifications.length > 0 && (
                                <button
                                    onClick={clearAll}
                                    className="p-1 text-slate-400 hover:text-red-400 transition-colors"
                                >
                                    <Trash2 className="w-4 h-4" />
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Notifications List */}
                    <div className="max-h-96 overflow-y-auto">
                        {notifications.length === 0 ? (
                            <div className="py-8 text-center text-slate-400">
                                <Bell className="w-10 h-10 mx-auto mb-2 opacity-50" />
                                <p>Bildirim yok</p>
                            </div>
                        ) : (
                            notifications.map((notification) => (
                                <div
                                    key={notification.id}
                                    className={`
                    flex items-start gap-3 px-4 py-3 border-b border-slate-800
                    hover:bg-slate-800/50 transition-colors cursor-pointer
                    ${!notification.read ? 'bg-blue-500/5' : ''}
                  `}
                                    onClick={() => markAsRead(notification.id)}
                                >
                                    <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${typeColors[notification.type]}`}>
                                        {typeIcons[notification.type]}
                                    </div>
                                    <div className="flex-1 min-w-0">
                                        <p className={`text-sm font-medium ${!notification.read ? 'text-white' : 'text-slate-300'}`}>
                                            {notification.title}
                                        </p>
                                        {notification.message && (
                                            <p className="text-xs text-slate-400 truncate">{notification.message}</p>
                                        )}
                                        <p className="text-xs text-slate-500 mt-1">{formatTime(notification.timestamp)}</p>
                                    </div>
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            removeNotification(notification.id);
                                        }}
                                        className="p-1 text-slate-500 hover:text-red-400 transition-colors"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
