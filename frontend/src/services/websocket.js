/**
 * WebSocket Service - CyberGuard AI
 * 
 * Real-time updates için WebSocket bağlantısı.
 */

class WebSocketService {
    constructor() {
        this.ws = null;
        this.listeners = new Map();
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 3000;
        this.isConnected = false;
    }

    connect(url = 'ws://localhost:8000/ws') {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected');
            return;
        }

        try {
            this.ws = new WebSocket(url);

            this.ws.onopen = () => {
                console.log('✅ WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.notify('connection', { status: 'connected' });
            };

            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.notify(data.type, data.payload);
                } catch (err) {
                    console.error('WebSocket message parse error:', err);
                }
            };

            this.ws.onclose = (event) => {
                console.log('❌ WebSocket disconnected', event.code);
                this.isConnected = false;
                this.notify('connection', { status: 'disconnected' });
                this.handleReconnect();
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.notify('error', { error });
            };

        } catch (err) {
            console.error('WebSocket connection failed:', err);
        }
    }

    handleReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`🔄 Reconnecting... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
            setTimeout(() => this.connect(), this.reconnectDelay);
        } else {
            console.error('Max reconnection attempts reached');
            this.notify('error', { error: 'Max reconnection attempts reached' });
        }
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
            this.isConnected = false;
        }
    }

    subscribe(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);

        // Return unsubscribe function
        return () => {
            this.listeners.get(event)?.delete(callback);
        };
    }

    unsubscribe(event, callback) {
        this.listeners.get(event)?.delete(callback);
    }

    notify(event, data) {
        const callbacks = this.listeners.get(event);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (err) {
                    console.error('WebSocket callback error:', err);
                }
            });
        }

        // Also notify 'all' listeners
        const allCallbacks = this.listeners.get('*');
        if (allCallbacks) {
            allCallbacks.forEach(callback => {
                try {
                    callback({ type: event, payload: data });
                } catch (err) {
                    console.error('WebSocket callback error:', err);
                }
            });
        }
    }

    send(type, payload) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ type, payload }));
            return true;
        }
        console.warn('WebSocket not connected, cannot send message');
        return false;
    }

    getStatus() {
        return {
            isConnected: this.isConnected,
            reconnectAttempts: this.reconnectAttempts,
            readyState: this.ws?.readyState,
        };
    }
}

// Singleton instance
export const wsService = new WebSocketService();

// React hook for WebSocket
export function useWebSocket(event, callback) {
    const { useEffect } = require('react');

    useEffect(() => {
        // Connect if not connected
        if (!wsService.isConnected) {
            wsService.connect();
        }

        // Subscribe to event
        const unsubscribe = wsService.subscribe(event, callback);

        // Cleanup
        return () => {
            unsubscribe();
        };
    }, [event, callback]);

    return wsService;
}

// Event types
export const WS_EVENTS = {
    // Connection
    CONNECTION: 'connection',
    ERROR: 'error',

    // Attacks
    NEW_ATTACK: 'new_attack',
    ATTACK_BLOCKED: 'attack_blocked',

    // IDS
    IDS_STATUS: 'ids_status',
    IDS_ALERT: 'ids_alert',

    // Training
    TRAINING_PROGRESS: 'training_progress',
    TRAINING_COMPLETE: 'training_complete',

    // System
    STATS_UPDATE: 'stats_update',
    LOG_ENTRY: 'log_entry',
};

export default wsService;
