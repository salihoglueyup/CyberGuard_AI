import { useEffect, useCallback } from 'react';
import { create } from 'zustand';

const WS_URL = 'ws://localhost:8000/ws';
const RECONNECT_INTERVAL = 5000;
const HEARTBEAT_INTERVAL = 30000;
const MAX_RECONNECT_ATTEMPTS = 5;

// Global WebSocket state (singleton)
const useWebSocketStore = create((set) => ({
    ws: null,
    isConnected: false,
    threats: [],
    systemStats: null,
    analytics: null,
    lastMessage: null,
    reconnectAttempts: 0,
    isConnecting: false,

    setConnected: (val) => set({ isConnected: val }),
    setThreats: (threats) => set({ threats }),
    addThreat: (threat) => set((state) => ({
        threats: [threat, ...state.threats].slice(0, 100)
    })),
    setSystemStats: (stats) => set({ systemStats: stats }),
    setAnalytics: (data) => set({ analytics: data }),
    setLastMessage: (msg) => set({ lastMessage: msg }),
    setReconnectAttempts: (val) => set({ reconnectAttempts: val }),
    setIsConnecting: (val) => set({ isConnecting: val }),
    setWs: (ws) => set({ ws }),
    clearThreats: () => set({ threats: [] }),
}));

// Singleton connection manager
let wsInstance = null;
let heartbeatInterval = null;
let reconnectTimeout = null;
let connectionInitialized = false;

const initializeConnection = () => {
    const store = useWebSocketStore.getState();

    // Zaten bağlıysa veya bağlanıyorsa çık
    if (connectionInitialized || store.isConnecting || store.isConnected) {
        return;
    }

    if (store.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.log('[WS] Max deneme aşıldı');
        return;
    }

    connectionInitialized = true;
    store.setIsConnecting(true);

    try {
        console.log('[WS] Bağlanılıyor...');
        const ws = new WebSocket(WS_URL);

        ws.onopen = () => {
            console.log('[WS] ✅ Bağlantı kuruldu');
            store.setConnected(true);
            store.setReconnectAttempts(0);
            store.setIsConnecting(false);
            store.setWs(ws);
            wsInstance = ws;

            // Heartbeat başlat
            if (heartbeatInterval) clearInterval(heartbeatInterval);
            heartbeatInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ type: 'ping' }));
                }
            }, HEARTBEAT_INTERVAL);
        };

        ws.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                store.setLastMessage(data);

                switch (data.type) {
                    case 'threat':
                        store.addThreat(data);
                        break;
                    case 'system_stats':
                        store.setSystemStats(data);
                        break;
                    case 'analytics':
                        store.setAnalytics(data);
                        break;
                    case 'pong':
                        // Heartbeat yanıtı
                        break;
                    default:
                        break;
                }
            } catch (err) {
                console.error('[WS] Parse hatası:', err);
            }
        };

        ws.onclose = (event) => {
            console.log('[WS] Bağlantı kapandı, kod:', event.code);
            store.setConnected(false);
            store.setIsConnecting(false);
            store.setWs(null);
            wsInstance = null;
            connectionInitialized = false;

            if (heartbeatInterval) {
                clearInterval(heartbeatInterval);
                heartbeatInterval = null;
            }

            // Normal kapanış değilse yeniden bağlan
            if (event.code !== 1000 && store.reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                store.setReconnectAttempts(store.reconnectAttempts + 1);

                if (reconnectTimeout) clearTimeout(reconnectTimeout);
                reconnectTimeout = setTimeout(() => {
                    console.log('[WS] Yeniden bağlanılıyor...');
                    initializeConnection();
                }, RECONNECT_INTERVAL);
            }
        };

        ws.onerror = () => {
            console.error('[WS] Hata oluştu');
            store.setIsConnecting(false);
            connectionInitialized = false;
        };

    } catch (err) {
        console.error('[WS] Bağlantı hatası:', err);
        store.setIsConnecting(false);
        connectionInitialized = false;
    }
};

const disconnectWebSocket = () => {
    const store = useWebSocketStore.getState();

    if (reconnectTimeout) {
        clearTimeout(reconnectTimeout);
        reconnectTimeout = null;
    }

    if (heartbeatInterval) {
        clearInterval(heartbeatInterval);
        heartbeatInterval = null;
    }

    if (wsInstance) {
        wsInstance.close(1000, 'Manuel kapatma');
        wsInstance = null;
    }

    store.setConnected(false);
    store.setIsConnecting(false);
    connectionInitialized = false;
};

const sendMessage = (message) => {
    if (wsInstance?.readyState === WebSocket.OPEN) {
        wsInstance.send(JSON.stringify(message));
        return true;
    }
    return false;
};

// React Hook
export function useWebSocket() {
    const {
        isConnected,
        threats,
        systemStats,
        analytics,
        lastMessage,
        reconnectAttempts,
        clearThreats,
    } = useWebSocketStore();

    // İlk mount'ta bağlan
    useEffect(() => {
        // Sadece bir kez bağlan
        const timer = setTimeout(() => {
            initializeConnection();
        }, 100);

        return () => {
            clearTimeout(timer);
        };
    }, []);

    const ping = useCallback(() => sendMessage({ type: 'ping' }), []);
    const requestThreat = useCallback(() => sendMessage({ type: 'request_threat' }), []);
    const requestStats = useCallback(() => sendMessage({ type: 'request_stats' }), []);
    const requestAnalytics = useCallback(() => sendMessage({ type: 'request_analytics' }), []);

    const resetConnection = useCallback(() => {
        useWebSocketStore.getState().setReconnectAttempts(0);
        disconnectWebSocket();
        setTimeout(initializeConnection, 100);
    }, []);

    return {
        isConnected,
        threats,
        systemStats,
        analytics,
        lastMessage,
        reconnectAttempts,
        connect: initializeConnection,
        disconnect: disconnectWebSocket,
        sendMessage,
        ping,
        requestThreat,
        requestStats,
        requestAnalytics,
        clearThreats,
        resetConnection,
    };
}

export default useWebSocket;
