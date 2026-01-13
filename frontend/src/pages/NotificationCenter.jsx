import React, { useState, useEffect } from 'react';
import api from '../services/api';

const NotificationCenter = () => {
    const [notifications, setNotifications] = useState([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState('all'); // all, unread, priority
    const [preferences, setPreferences] = useState(null);

    useEffect(() => {
        loadNotifications();
        loadPreferences();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [filter]);

    const loadNotifications = async () => {
        try {
            const params = new URLSearchParams();
            if (filter === 'unread') params.append('unread_only', true);

            const response = await api.get(`/notifications?${params.toString()}`);
            setNotifications(response.data.data.notifications);
            setUnreadCount(response.data.data.unread_count);
        } catch (error) {
            console.error('Error loading notifications:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadPreferences = async () => {
        try {
            const response = await api.get('/notifications/preferences');
            setPreferences(response.data.data);
        } catch (error) {
            console.error('Error loading preferences:', error);
        }
    };

    const markAsRead = async (notificationId) => {
        try {
            await api.put(`/notifications/${notificationId}/read`);
            setNotifications(prev =>
                prev.map(n => n.id === notificationId ? { ...n, read: true } : n)
            );
            setUnreadCount(prev => Math.max(0, prev - 1));
        } catch (error) {
            console.error('Error marking as read:', error);
        }
    };

    const markAllRead = async () => {
        try {
            await api.put('/notifications/read-all');
            setNotifications(prev => prev.map(n => ({ ...n, read: true })));
            setUnreadCount(0);
        } catch (error) {
            console.error('Error marking all as read:', error);
        }
    };

    const deleteNotification = async (notificationId) => {
        try {
            await api.delete(`/notifications/${notificationId}`);
            setNotifications(prev => prev.filter(n => n.id !== notificationId));
        } catch (error) {
            console.error('Error deleting notification:', error);
        }
    };

    const getTypeIcon = (type) => {
        switch (type) {
            case 'attack_detected': return '🚨';
            case 'model_drift': return '📊';
            case 'system_update': return '🔄';
            case 'report_ready': return '📋';
            case 'credential_leak': return '🔑';
            case 'scan_complete': return '✅';
            default: return '🔔';
        }
    };

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'urgent': return 'bg-red-600';
            case 'high': return 'bg-orange-600';
            case 'normal': return 'bg-blue-600';
            case 'low': return 'bg-gray-600';
            default: return 'bg-gray-600';
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-cyan-500 mx-auto"></div>
                    <p className="mt-4 text-cyan-400">Loading Notifications...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-3xl font-bold text-cyan-400">🔔 Notification Center</h1>
                    <p className="text-gray-400">
                        {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
                    </p>
                </div>
                <div className="flex gap-3">
                    <button
                        onClick={markAllRead}
                        className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg font-medium"
                    >
                        ✓ Mark All Read
                    </button>
                    <button
                        onClick={loadNotifications}
                        className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg font-medium"
                    >
                        🔄 Refresh
                    </button>
                </div>
            </div>

            {/* Filters */}
            <div className="flex gap-2 mb-6">
                {['all', 'unread'].map((f) => (
                    <button
                        key={f}
                        onClick={() => setFilter(f)}
                        className={`px-4 py-2 rounded-lg font-medium transition ${filter === f
                            ? 'bg-cyan-600 text-white'
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                            }`}
                    >
                        {f === 'all' ? 'All' : 'Unread'}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Notifications List */}
                <div className="lg:col-span-2 space-y-3">
                    {notifications.length === 0 ? (
                        <div className="bg-gray-800 rounded-lg p-8 text-center border border-gray-700">
                            <p className="text-gray-400 text-lg">No notifications</p>
                        </div>
                    ) : (
                        notifications.map((notification) => (
                            <div
                                key={notification.id}
                                className={`bg-gray-800 rounded-lg p-4 border transition hover:border-cyan-500/50 ${notification.read ? 'border-gray-700 opacity-60' : 'border-cyan-500/30'
                                    }`}
                            >
                                <div className="flex items-start justify-between">
                                    <div className="flex items-start gap-3">
                                        <span className="text-2xl">{getTypeIcon(notification.type)}</span>
                                        <div>
                                            <h3 className={`font-semibold ${notification.read ? 'text-gray-400' : 'text-white'}`}>
                                                {notification.title}
                                            </h3>
                                            <p className="text-gray-400 text-sm mt-1">{notification.message}</p>
                                            <div className="flex items-center gap-3 mt-2">
                                                <span className={`text-xs px-2 py-1 rounded ${getPriorityColor(notification.priority)}`}>
                                                    {notification.priority?.toUpperCase()}
                                                </span>
                                                <span className="text-xs text-gray-500">
                                                    {new Date(notification.created_at).toLocaleString()}
                                                </span>
                                            </div>
                                        </div>
                                    </div>
                                    <div className="flex gap-2">
                                        {!notification.read && (
                                            <button
                                                onClick={() => markAsRead(notification.id)}
                                                className="p-2 hover:bg-gray-700 rounded-lg transition"
                                                title="Mark as read"
                                            >
                                                ✓
                                            </button>
                                        )}
                                        <button
                                            onClick={() => deleteNotification(notification.id)}
                                            className="p-2 hover:bg-red-900/50 rounded-lg transition text-red-400"
                                            title="Delete"
                                        >
                                            🗑️
                                        </button>
                                    </div>
                                </div>
                            </div>
                        ))
                    )}
                </div>

                {/* Preferences Panel */}
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 h-fit">
                    <h3 className="text-lg font-semibold text-cyan-400 mb-4">⚙️ Preferences</h3>

                    {preferences && (
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span>Email Notifications</span>
                                <span className={`px-2 py-1 rounded text-xs ${preferences.email_enabled ? 'bg-green-600' : 'bg-gray-600'}`}>
                                    {preferences.email_enabled ? 'ON' : 'OFF'}
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span>Slack Notifications</span>
                                <span className={`px-2 py-1 rounded text-xs ${preferences.slack_enabled ? 'bg-green-600' : 'bg-gray-600'}`}>
                                    {preferences.slack_enabled ? 'ON' : 'OFF'}
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span>In-App Notifications</span>
                                <span className={`px-2 py-1 rounded text-xs ${preferences.in_app_enabled ? 'bg-green-600' : 'bg-gray-600'}`}>
                                    {preferences.in_app_enabled ? 'ON' : 'OFF'}
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span>Digest Frequency</span>
                                <span className="text-cyan-400">{preferences.digest_frequency}</span>
                            </div>

                            <hr className="border-gray-700" />

                            <div>
                                <h4 className="font-medium mb-2">Quiet Hours</h4>
                                <p className="text-sm text-gray-400">
                                    {preferences.quiet_hours?.enabled
                                        ? `${preferences.quiet_hours.start} - ${preferences.quiet_hours.end}`
                                        : 'Disabled'
                                    }
                                </p>
                            </div>

                            <button className="w-full mt-4 px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg font-medium">
                                Edit Preferences
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default NotificationCenter;
