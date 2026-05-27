import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';

// WebSocket mock
class MockWebSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;

    constructor(url) {
        this.url = url;
        this.readyState = MockWebSocket.CONNECTING;
        this.onopen = null;
        this.onmessage = null;
        this.onerror = null;
        this.onclose = null;
        MockWebSocket.instances.push(this);
    }

    send(data) {
        this._sentMessages = this._sentMessages || [];
        this._sentMessages.push(data);
    }

    close() {
        this.readyState = MockWebSocket.CLOSED;
        if (this.onclose) this.onclose({ code: 1000, reason: 'test' });
    }

    // Test yardımcısı: bağlantıyı simüle et
    simulateOpen() {
        this.readyState = MockWebSocket.OPEN;
        if (this.onopen) this.onopen({});
    }

    // Test yardımcısı: mesaj simüle et
    simulateMessage(data) {
        if (this.onmessage) {
            this.onmessage({ data: typeof data === 'string' ? data : JSON.stringify(data) });
        }
    }

    static instances = [];
    static reset() {
        MockWebSocket.instances = [];
    }
}

describe('WebSocket Mock Infrastructure', () => {
    beforeEach(() => {
        MockWebSocket.reset();
        global.WebSocket = MockWebSocket;
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it('can create a WebSocket connection', () => {
        const ws = new WebSocket('ws://localhost:8000/ws');
        expect(ws).toBeDefined();
        expect(ws.url).toBe('ws://localhost:8000/ws');
        expect(ws.readyState).toBe(MockWebSocket.CONNECTING);
    });

    it('simulates open event', () => {
        const ws = new MockWebSocket('ws://localhost:8000/ws');
        const onopen = vi.fn();
        ws.onopen = onopen;

        ws.simulateOpen();

        expect(onopen).toHaveBeenCalled();
        expect(ws.readyState).toBe(MockWebSocket.OPEN);
    });

    it('simulates message event with JSON data', () => {
        const ws = new MockWebSocket('ws://localhost:8000/ws');
        const onmessage = vi.fn();
        ws.onmessage = onmessage;

        const payload = { type: 'threat', data: { severity: 'high' } };
        ws.simulateMessage(payload);

        expect(onmessage).toHaveBeenCalled();
        const received = JSON.parse(onmessage.mock.calls[0][0].data);
        expect(received.type).toBe('threat');
        expect(received.data.severity).toBe('high');
    });

    it('sends heartbeat message', () => {
        const ws = new MockWebSocket('ws://localhost:8000/ws');
        ws.simulateOpen();
        ws.send(JSON.stringify({ type: 'ping' }));

        expect(ws._sentMessages).toHaveLength(1);
        const msg = JSON.parse(ws._sentMessages[0]);
        expect(msg.type).toBe('ping');
    });

    it('closes connection cleanly', () => {
        const ws = new MockWebSocket('ws://localhost:8000/ws');
        const onclose = vi.fn();
        ws.onclose = onclose;

        ws.close();

        expect(ws.readyState).toBe(MockWebSocket.CLOSED);
        expect(onclose).toHaveBeenCalled();
    });
});
