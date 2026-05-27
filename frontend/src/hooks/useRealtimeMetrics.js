import { useEffect, useRef, useCallback } from 'react';
import { create } from 'zustand';

const MAX_POINTS = 60; // 60 data points = 60 seconds of history

// Realistic system metric simulation
function randomWalk(prev, min, max, volatility = 2) {
    const delta = (Math.random() - 0.5) * volatility * 2;
    return Math.min(max, Math.max(min, prev + delta));
}

function generateAttack() {
    const types = ['DDoS', 'SQL Injection', 'XSS', 'Brute Force', 'Port Scan', 'Ransomware', 'Phishing', 'Zero-Day', 'MITM', 'RCE'];
    const severities = ['critical', 'high', 'medium', 'low'];
    const severityWeights = [0.1, 0.25, 0.4, 0.25];
    const countries = ['CN', 'RU', 'US', 'BR', 'IR', 'KP', 'DE', 'IN', 'NG', 'UA', 'TR', 'VN'];
    const protocols = ['TCP', 'UDP', 'HTTP', 'HTTPS', 'SSH', 'DNS', 'ICMP'];

    const r = Math.random();
    let sevIdx = 0, cumulative = 0;
    for (let i = 0; i < severityWeights.length; i++) {
        cumulative += severityWeights[i];
        if (r <= cumulative) { sevIdx = i; break; }
    }

    const srcIP = `${Math.floor(Math.random() * 223) + 1}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}.${Math.floor(Math.random() * 256)}`;
    const dstIP = `10.0.${Math.floor(Math.random() * 5)}.${Math.floor(Math.random() * 254) + 1}`;

    return {
        id: crypto.randomUUID(),
        type: types[Math.floor(Math.random() * types.length)],
        severity: severities[sevIdx],
        source_ip: srcIP,
        dest_ip: dstIP,
        source_country: countries[Math.floor(Math.random() * countries.length)],
        protocol: protocols[Math.floor(Math.random() * protocols.length)],
        port: [22, 80, 443, 3306, 8080, 3389, 445, 53, 25][Math.floor(Math.random() * 9)],
        blocked: Math.random() > 0.2,
        confidence: Math.round(70 + Math.random() * 30),
        timestamp: new Date().toISOString(),
        time: new Date().toLocaleTimeString('tr-TR'),
    };
}

export const useRealtimeStore = create((set, get) => ({
    // Time-series metrics (arrays of {time, value})
    cpu: [],
    memory: [],
    network: [],
    disk: [],
    bandwidth: { inbound: [], outbound: [] },
    connections: [],
    requests: [],

    // Current values
    currentCpu: 35,
    currentMemory: 62,
    currentNetwork: 45,
    currentDisk: 38,
    currentBandwidthIn: 120,
    currentBandwidthOut: 85,
    currentConnections: 340,
    currentRps: 1200,

    // Live attack feed
    liveAttacks: [],
    attacksPerMinute: 0,
    totalBlocked: 0,
    totalAttacks: 0,

    // Traffic distribution
    trafficByProtocol: { HTTP: 45, HTTPS: 30, SSH: 8, DNS: 10, Other: 7 },

    pushMetric: (key, value) => set(state => {
        const arr = [...(state[key] || []), { time: new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' }), value }];
        return { [key]: arr.slice(-MAX_POINTS) };
    }),

    pushBandwidth: (inVal, outVal) => set(state => {
        const time = new Date().toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        return {
            bandwidth: {
                inbound: [...state.bandwidth.inbound, { time, value: inVal }].slice(-MAX_POINTS),
                outbound: [...state.bandwidth.outbound, { time, value: outVal }].slice(-MAX_POINTS),
            }
        };
    }),

    addLiveAttack: (attack) => set(state => ({
        liveAttacks: [attack, ...state.liveAttacks].slice(0, 50),
        totalAttacks: state.totalAttacks + 1,
        totalBlocked: state.totalBlocked + (attack.blocked ? 1 : 0),
    })),

    setAttacksPerMinute: (v) => set({ attacksPerMinute: v }),
}));

/**
 * useRealtimeMetrics — drives simulated real-time system metrics.
 * In production, replace simulation with actual WebSocket data.
 */
export function useRealtimeMetrics(interval = 1000) {
    const tickRef = useRef(null);
    const attackCountRef = useRef(0);
    const store = useRealtimeStore;

    const tick = useCallback(() => {
        const s = store.getState();

        // Walk system metrics
        const cpu = randomWalk(s.currentCpu, 8, 95, 3);
        const memory = randomWalk(s.currentMemory, 40, 92, 1.5);
        const network = randomWalk(s.currentNetwork, 10, 100, 4);
        const disk = randomWalk(s.currentDisk, 25, 80, 0.5);
        const bwIn = randomWalk(s.currentBandwidthIn, 20, 500, 15);
        const bwOut = randomWalk(s.currentBandwidthOut, 10, 300, 10);
        const conns = Math.round(randomWalk(s.currentConnections, 100, 800, 20));
        const rps = Math.round(randomWalk(s.currentRps, 200, 3000, 50));

        store.setState({
            currentCpu: cpu,
            currentMemory: memory,
            currentNetwork: network,
            currentDisk: disk,
            currentBandwidthIn: bwIn,
            currentBandwidthOut: bwOut,
            currentConnections: conns,
            currentRps: rps,
        });

        s.pushMetric('cpu', Math.round(cpu * 10) / 10);
        s.pushMetric('memory', Math.round(memory * 10) / 10);
        s.pushMetric('network', Math.round(network * 10) / 10);
        s.pushMetric('disk', Math.round(disk * 10) / 10);
        s.pushMetric('connections', conns);
        s.pushMetric('requests', rps);
        s.pushBandwidth(Math.round(bwIn), Math.round(bwOut));

        // Random attack generation (avg ~4 attacks per minute)
        if (Math.random() < 0.07) {
            const attack = generateAttack();
            s.addLiveAttack(attack);
            attackCountRef.current++;
        }
    }, []);

    // APM counter (attacks per minute)
    useEffect(() => {
        const apmInterval = setInterval(() => {
            store.getState().setAttacksPerMinute(attackCountRef.current);
            attackCountRef.current = 0;
        }, 60000);
        return () => clearInterval(apmInterval);
    }, []);

    useEffect(() => {
        tickRef.current = setInterval(tick, interval);
        return () => clearInterval(tickRef.current);
    }, [tick, interval]);
}
