import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import api from '../services/api';

/**
 * Custom hook to manage the state and WebSocket connection for the 3D Attack Map.
 * Handles live data feeds, historical stats, time-travel playback, and filtering.
 */
export default function useAttackMap() {
    const [attacks, setAttacks] = useState([]);
    const [stats, setStats] = useState(null);
    const [countries, setCountries] = useState([]);
    const [loading, setLoading] = useState(true);
    const [isLive, setIsLive] = useState(true);

    // Playback state
    const [timelineData, setTimelineData] = useState([]);
    const [playbackTime, setPlaybackTime] = useState(null); // null means showing live/latest

    // Filter state
    const [filters, setFilters] = useState({
        minSeverity: 'low', // 'low', 'medium', 'high', 'critical'
        threatType: 'all',
        showOnlyBlocked: false,
    });

    const [wsConnected, setWsConnected] = useState(false);
    const [mlStats, setMlStats] = useState({ predictions_made: 0, threats_detected: 0, accuracy: 0.94 });

    const wsRef = useRef(null);
    const attackCacheRef = useRef([]); // Store up to 500 recent attacks for playback

    // Load initial stats and countries
    useEffect(() => {
        const loadInitialData = async () => {
            try {
                const [attacksRes, statsRes, countriesRes] = await Promise.all([
                    api.get('/attack-map/live?limit=50'),
                    api.get('/attack-map/stats'),
                    api.get('/attack-map/countries')
                ]);

                const initialAttacks = attacksRes.data.data?.attacks || [];
                setAttacks(initialAttacks);
                attackCacheRef.current = initialAttacks;

                setStats(statsRes.data.data);
                setCountries(countriesRes.data.data?.countries || []);
            } catch (error) {
                console.error('Failed to load initial attack map data:', error);
            } finally {
                setLoading(false);
            }
        };

        loadInitialData();
    }, []);

    // WebSocket Connection Management
    useEffect(() => {
        if (!isLive) return;

        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        let reconnectTimeout = null;

        const connectWebSocket = () => {
            try {
                const wsUrl = import.meta.env.VITE_WS_URL || `ws://${window.location.hostname}:8000/ws`;
                const ws = new WebSocket(`${wsUrl}/attacks`);

                ws.onopen = () => {
                    console.log('[useAttackMap] WebSocket connected');
                    setWsConnected(true);
                    reconnectAttempts = 0;
                };

                ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);

                        if (message.type === 'attack') {
                            const newAttack = message.data;

                            // Update cache
                            attackCacheRef.current = [newAttack, ...attackCacheRef.current].slice(0, 500);

                            // Update live view if not in playback mode
                            if (!playbackTime) {
                                setAttacks(prev => [newAttack, ...prev].slice(0, 100)); // Keep max 100 in view for perf
                            }

                            // Update ML stats
                            if (newAttack.ml_prediction) {
                                setMlStats(prev => ({
                                    ...prev,
                                    predictions_made: prev.predictions_made + 1,
                                    threats_detected: newAttack.ml_prediction.is_threat
                                        ? prev.threats_detected + 1
                                        : prev.threats_detected
                                }));
                            }
                        }

                        if (message.type === 'heartbeat') {
                            ws.send(JSON.stringify({ type: 'ping' }));
                        }
                    } catch (e) {
                        console.error('WS message parse error:', e);
                    }
                };

                ws.onerror = () => {
                    setWsConnected(false);
                };

                ws.onclose = () => {
                    setWsConnected(false);
                    if (reconnectAttempts < maxReconnectAttempts) {
                        reconnectAttempts++;
                        const delay = Math.min(3000 * reconnectAttempts, 15000);
                        reconnectTimeout = setTimeout(connectWebSocket, delay);
                    }
                };

                wsRef.current = ws;
            } catch (e) {
                console.error('WS connection error:', e);
                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    reconnectTimeout = setTimeout(connectWebSocket, 5000);
                }
            }
        };

        connectWebSocket();

        return () => {
            if (reconnectTimeout) clearTimeout(reconnectTimeout);
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [isLive, playbackTime]);

    // Timeline Data generation (bucket attacks into time chunks)
    useEffect(() => {
        const updateTimeline = () => {
            const now = Date.now();
            const buckets = Array.from({ length: 30 }, (_, i) => ({
                id: i,
                time: now - (29 - i) * 2000, // 2-second buckets
                count: 0
            }));

            attackCacheRef.current.forEach(attack => {
                const attackTime = new Date(attack.timestamp).getTime();
                const bucketIndex = buckets.findIndex((b, i) => {
                    const nextTime = i < 29 ? buckets[i + 1].time : Infinity;
                    return attackTime >= b.time && attackTime < nextTime;
                });

                if (bucketIndex !== -1) {
                    buckets[bucketIndex].count++;
                }
            });

            setTimelineData(buckets);
        };

        const interval = setInterval(updateTimeline, 2000);
        return () => clearInterval(interval);
    }, [attacks]); // Re-run when attacks update to refresh timeline

    // Filter logic
    const filteredAttacks = useMemo(() => {
        let currentAttacks = attacks;

        // If in playback mode, filter by time
        if (playbackTime) {
            const timeWindow = 5000; // Show attacks within 5 seconds of playback time
            currentAttacks = attackCacheRef.current.filter(a => {
                const t = new Date(a.timestamp).getTime();
                return Math.abs(t - playbackTime) <= timeWindow;
            });
        }

        const severityLevels = { low: 1, medium: 2, high: 3, critical: 4 };
        const minLevel = severityLevels[filters.minSeverity] || 1;

        return currentAttacks.filter(attack => {
            const level = severityLevels[attack.severity] || 1;
            if (level < minLevel) return false;

            if (filters.threatType !== 'all' && attack.threat_type !== filters.threatType) return false;

            if (filters.showOnlyBlocked && !attack.blocked) return false;

            return true;
        });
    }, [attacks, filters, playbackTime]);

    // Helpers
    const toggleLive = useCallback(() => {
        setIsLive(prev => !prev);
        if (!isLive) {
            setPlaybackTime(null); // Reset playback when going live
            setAttacks([...attackCacheRef.current].slice(0, 100));
        }
    }, [isLive]);

    const setPlayback = useCallback((time) => {
        setIsLive(false);
        setPlaybackTime(time);
    }, []);

    const updateFilter = useCallback((key, value) => {
        setFilters(prev => ({ ...prev, [key]: value }));
    }, []);

    // Derived stats for UI
    const activeStats = useMemo(() => {
        if (!stats) return { total: 0, blocked: 0, critical: 0 };
        return {
            total: filteredAttacks.length,
            blocked: filteredAttacks.filter(a => a.blocked).length,
            critical: filteredAttacks.filter(a => a.severity === 'critical').length,
            mlThreats: filteredAttacks.filter(a => a.ml_prediction?.is_threat).length,
            avgConfidence: filteredAttacks.reduce((s, a) => s + (a.ml_prediction?.confidence || 0), 0) / (filteredAttacks.length || 1)
        };
    }, [filteredAttacks, stats]);

    return {
        attacks: filteredAttacks,
        rawAttacksLength: attacks.length,
        stats: activeStats,
        globalStats: stats,
        countries,
        loading,
        isLive,
        wsConnected,
        mlStats,
        timelineData,
        playbackTime,
        filters,
        toggleLive,
        setPlayback,
        updateFilter,
    };
}
