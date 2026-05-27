import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { useNotificationStore } from '../../components/NotificationBell';

// Mock framer-motion to avoid animation issues in tests
vi.mock('framer-motion', () => ({
    motion: {
        div: ({ children, ...props }) => <div {...props}>{children}</div>,
        button: ({ children, ...props }) => <button {...props}>{children}</button>,
    },
    AnimatePresence: ({ children }) => <>{children}</>,
}));

// Mock zustand persist
vi.mock('zustand/middleware', () => ({
    persist: (fn) => fn,
}));

describe('useNotificationStore', () => {
    beforeEach(() => {
        // Store'u sıfırla
        useNotificationStore.setState({ notifications: [], unreadCount: 0 });
    });

    it('starts with empty notifications', () => {
        const state = useNotificationStore.getState();
        expect(state.notifications).toHaveLength(0);
        expect(state.unreadCount).toBe(0);
    });

    it('adds a notification', () => {
        const store = useNotificationStore.getState();
        store.addNotification({ title: 'Test', message: 'Test message', type: 'info' });

        const state = useNotificationStore.getState();
        expect(state.notifications).toHaveLength(1);
        expect(state.unreadCount).toBe(1);
        expect(state.notifications[0].title).toBe('Test');
        expect(state.notifications[0].read).toBe(false);
        expect(state.notifications[0].id).toBeDefined();
    });

    it('marks a notification as read', () => {
        const store = useNotificationStore.getState();
        store.addNotification({ title: 'Test', message: 'msg' });

        const { notifications } = useNotificationStore.getState();
        const id = notifications[0].id;

        store.markAsRead(id);

        const updated = useNotificationStore.getState();
        expect(updated.notifications[0].read).toBe(true);
        expect(updated.unreadCount).toBe(0);
    });

    it('marks all notifications as read', () => {
        const store = useNotificationStore.getState();
        store.addNotification({ title: 'A', message: 'msg1' });
        store.addNotification({ title: 'B', message: 'msg2' });

        store.markAllAsRead();

        const state = useNotificationStore.getState();
        expect(state.unreadCount).toBe(0);
        expect(state.notifications.every((n) => n.read)).toBe(true);
    });

    it('removes a notification', () => {
        const store = useNotificationStore.getState();
        store.addNotification({ title: 'Remove Me', message: 'msg' });

        const { notifications } = useNotificationStore.getState();
        store.removeNotification(notifications[0].id);

        const state = useNotificationStore.getState();
        expect(state.notifications).toHaveLength(0);
        expect(state.unreadCount).toBe(0);
    });

    it('clears all notifications', () => {
        const store = useNotificationStore.getState();
        store.addNotification({ title: 'A', message: 'msg1' });
        store.addNotification({ title: 'B', message: 'msg2' });

        store.clearAll();

        const state = useNotificationStore.getState();
        expect(state.notifications).toHaveLength(0);
        expect(state.unreadCount).toBe(0);
    });

    it('limits notifications to 50', () => {
        const store = useNotificationStore.getState();
        for (let i = 0; i < 55; i++) {
            store.addNotification({ title: `Notification ${i}`, message: 'msg' });
        }

        const state = useNotificationStore.getState();
        expect(state.notifications.length).toBeLessThanOrEqual(50);
    });
});
