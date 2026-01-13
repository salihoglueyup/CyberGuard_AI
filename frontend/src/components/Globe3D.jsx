import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import Globe from 'react-globe.gl';

// Ülke verileri
const COUNTRIES_DATA = {
    TR: { name: 'Türkiye', lat: 39.0, lng: 35.0 },
    US: { name: 'ABD', lat: 38.0, lng: -97.0 },
    CN: { name: 'Çin', lat: 35.0, lng: 105.0 },
    RU: { name: 'Rusya', lat: 60.0, lng: 100.0 },
    DE: { name: 'Almanya', lat: 51.0, lng: 9.0 },
    GB: { name: 'İngiltere', lat: 54.0, lng: -2.0 },
    FR: { name: 'Fransa', lat: 46.0, lng: 2.0 },
    JP: { name: 'Japonya', lat: 36.0, lng: 138.0 },
    KR: { name: 'Güney Kore', lat: 36.0, lng: 128.0 },
    IN: { name: 'Hindistan', lat: 21.0, lng: 78.0 },
    BR: { name: 'Brezilya', lat: -10.0, lng: -55.0 },
    AU: { name: 'Avustralya', lat: -25.0, lng: 135.0 },
    CA: { name: 'Kanada', lat: 56.0, lng: -106.0 },
    IT: { name: 'İtalya', lat: 42.8, lng: 12.8 },
    ES: { name: 'İspanya', lat: 40.0, lng: -4.0 },
    NL: { name: 'Hollanda', lat: 52.5, lng: 5.7 },
    UA: { name: 'Ukrayna', lat: 49.0, lng: 32.0 },
    PL: { name: 'Polonya', lat: 52.0, lng: 19.0 },
    IR: { name: 'İran', lat: 32.0, lng: 53.0 },
    SA: { name: 'Suudi Arabistan', lat: 24.0, lng: 45.0 },
    EG: { name: 'Mısır', lat: 27.0, lng: 30.0 },
    ZA: { name: 'Güney Afrika', lat: -29.0, lng: 24.0 },
    NG: { name: 'Nijerya', lat: 10.0, lng: 8.0 },
    ID: { name: 'Endonezya', lat: -2.0, lng: 118.0 },
    PK: { name: 'Pakistan', lat: 30.0, lng: 70.0 },
    MX: { name: 'Meksika', lat: 23.0, lng: -102.0 },
    VN: { name: 'Vietnam', lat: 16.0, lng: 108.0 },
    TH: { name: 'Tayland', lat: 15.0, lng: 101.0 },
    AR: { name: 'Arjantin', lat: -34.0, lng: -64.0 },
    CL: { name: 'Şili', lat: -33.0, lng: -71.0 },
};

// Renk fonksiyonları
const getThreatColor = (count) => {
    if (count > 20) return '#ff0040';
    if (count > 10) return '#ff6600';
    if (count > 5) return '#ffcc00';
    if (count > 0) return '#00aaff';
    return '#22c55e';
};

const getHeatmapColor = (value) => {
    // 0-1 arası değer için renk gradyanı
    const r = Math.floor(255 * Math.min(1, value * 2));
    const g = Math.floor(255 * Math.max(0, 1 - value * 2));
    return `rgba(${r}, ${g}, 0, ${0.3 + value * 0.5})`;
};

const getSeverityColor = (severity) => {
    switch (severity) {
        case 'critical': return '#ff0040';
        case 'high': return '#ff6600';
        case 'medium': return '#ffcc00';
        case 'low': return '#00cc66';
        default: return '#00aaff';
    }
};

// Ses efekti
const playAlertSound = () => {
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        gainNode.gain.value = 0.1;

        oscillator.start();
        oscillator.stop(audioContext.currentTime + 0.15);
    } catch {
        console.log('Audio not supported');
    }
};

export default function Globe3D({ attacks = [], isLive = true, onToggleView }) {
    const globeRef = useRef();
    const containerRef = useRef();
    const [countries, setCountries] = useState({ features: [] });
    const [selectedCountry, setSelectedCountry] = useState(null);
    const [hoverD, setHoverD] = useState(null);
    const [ringPhase, setRingPhase] = useState(0);
    const [soundEnabled, setSoundEnabled] = useState(true);
    const [dimensions, setDimensions] = useState({ width: 800, height: 500 });
    const [timelineData, setTimelineData] = useState([]);
    const lastAttackCountRef = useRef(0);
    const timelineRef = useRef([]);

    // WebSocket state
    const [wsConnected, setWsConnected] = useState(false);
    const [wsAttacks, setWsAttacks] = useState([]);
    // eslint-disable-next-line no-unused-vars
    const [mlStats, setMlStats] = useState({ predictions_made: 0, threats_detected: 0, accuracy: 0.94 });
    const wsRef = useRef(null);

    // Combine props attacks with WebSocket attacks
    const allAttacks = [...wsAttacks, ...attacks].slice(0, 100);

    // WebSocket bağlantısı
    useEffect(() => {
        if (!isLive) return;

        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;
        let reconnectTimeout = null;

        const connectWebSocket = () => {
            try {
                const ws = new WebSocket('ws://localhost:8000/ws/attacks');

                ws.onopen = () => {
                    console.log('[Globe3D] WebSocket connected');
                    setWsConnected(true);
                    reconnectAttempts = 0; // Reset on successful connection
                };

                ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);

                        if (message.type === 'attack') {
                            const attack = message.data;
                            setWsAttacks(prev => [attack, ...prev].slice(0, 50));

                            // Update ML stats if available
                            if (attack.ml_prediction) {
                                setMlStats(prev => ({
                                    ...prev,
                                    predictions_made: prev.predictions_made + 1,
                                    threats_detected: attack.ml_prediction.is_threat
                                        ? prev.threats_detected + 1
                                        : prev.threats_detected
                                }));
                            }

                            // Sound for high threat
                            if (soundEnabled && attack.ml_prediction?.confidence > 0.85) {
                                playAlertSound();
                            }
                        }

                        if (message.type === 'heartbeat') {
                            ws.send(JSON.stringify({ type: 'ping' }));
                        }
                    } catch {
                        // Silent parse errors
                    }
                };

                ws.onerror = () => {
                    // Silent error - onclose will handle reconnection
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
            } catch {
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
    }, [isLive, soundEnabled]);

    // GeoJSON yükle
    useEffect(() => {
        fetch('https://raw.githubusercontent.com/vasturiano/react-globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson')
            .then(res => res.json())
            .then(setCountries)
            .catch(console.error);
    }, []);

    // Ring Pulse animasyonu
    useEffect(() => {
        const interval = setInterval(() => {
            setRingPhase(prev => (prev + 1) % 100);
        }, 50);
        return () => clearInterval(interval);
    }, []);

    // Timeline verisi güncelle (ref kullanarak cascade render önle)
    useEffect(() => {
        const now = new Date();
        timelineRef.current = [...timelineRef.current, { time: now, count: allAttacks.length }].slice(-30);
        setTimelineData([...timelineRef.current]);
    }, [allAttacks.length]);

    // Yeni saldırı geldiğinde auto-focus ve ses
    useEffect(() => {
        if (attacks.length > lastAttackCountRef.current && lastAttackCountRef.current > 0) {
            const newAttack = attacks[0];

            // Ses efekti
            if (soundEnabled && newAttack?.severity === 'critical') {
                playAlertSound();
            }

            // Auto-focus
            if (globeRef.current && newAttack?.source?.lat && newAttack?.source?.lng) {
                globeRef.current.pointOfView({
                    lat: newAttack.source.lat,
                    lng: newAttack.source.lng,
                    altitude: 2
                }, 1000);
            }
        }
        lastAttackCountRef.current = attacks.length;
    }, [attacks, soundEnabled]);

    // Ülke başına saldırı sayısı
    const attackCounts = useMemo(() => {
        const counts = {};
        allAttacks.forEach(attack => {
            const code = attack?.source?.country;
            if (code) counts[code] = (counts[code] || 0) + 1;
        });
        return counts;
    }, [allAttacks]);

    // İstatistikler
    const stats = useMemo(() => {
        const total = allAttacks.length;
        const blocked = allAttacks.filter(a => a.blocked).length;
        const critical = allAttacks.filter(a => a.severity === 'critical').length;
        const mlThreats = allAttacks.filter(a => a.ml_prediction?.is_threat).length;
        const avgConfidence = allAttacks.reduce((sum, a) => sum + (a.ml_prediction?.confidence || 0), 0) / (allAttacks.length || 1);
        const topCountries = Object.entries(attackCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 5);

        return { total, blocked, critical, mlThreats, avgConfidence, topCountries };
    }, [allAttacks, attackCounts]);

    // Hexbin verileri
    const hexbinData = useMemo(() => {
        return allAttacks.map(a => ({
            lat: a?.source?.lat || 0,
            lng: a?.source?.lng || 0,
            weight: a?.severity === 'critical' ? 3 : a?.severity === 'high' ? 2 : 1,
        })).filter(h => h.lat !== 0);
    }, [allAttacks]);

    // Arc verileri
    const arcsData = useMemo(() => {
        return allAttacks.slice(0, 40).map((attack, i) => ({
            id: attack.id || i,
            startLat: attack?.source?.lat || 0,
            startLng: attack?.source?.lng || 0,
            endLat: attack?.target?.lat || 39.0,
            endLng: attack?.target?.lng || 35.0,
            color: attack.ml_prediction?.is_threat
                ? [getSeverityColor(attack?.severity), '#ff0040']
                : [getSeverityColor(attack?.severity), 'rgba(0,255,136,0.3)'],
            stroke: attack?.severity === 'critical' ? 3 : attack.ml_prediction?.confidence > 0.8 ? 2.5 : 1.5,
        })).filter(arc => arc.startLat !== 0);
    }, [allAttacks]);

    // Ring verileri (Türkiye etrafında)
    const ringsData = useMemo(() => {
        const rings = [];
        for (let i = 0; i < 3; i++) {
            const phase = (ringPhase + i * 33) % 100;
            rings.push({
                lat: 39.0,
                lng: 35.0,
                maxR: 3 + phase * 0.05,
                propagationSpeed: 2,
                repeatPeriod: 1000,
                color: `rgba(0, 255, 136, ${0.8 - phase * 0.008})`,
            });
        }
        return rings;
    }, [ringPhase]);

    // Label verileri
    const labelsData = useMemo(() => {
        return Object.entries(attackCounts)
            .filter(([, count]) => count > 3)
            .map(([code, count]) => {
                const country = COUNTRIES_DATA[code];
                if (!country) return null;
                return {
                    lat: country.lat,
                    lng: country.lng,
                    text: `${country.name}: ${count}`,
                    color: getThreatColor(count),
                    size: Math.min(0.8 + count * 0.05, 1.5),
                };
            }).filter(Boolean);
    }, [attackCounts]);

    // Heatmap için polygon rengi
    const getPolygonColor = useCallback((d) => {
        const code = d?.properties?.ISO_A2;
        if (code === 'TR') return 'rgba(0, 255, 136, 0.7)';

        const count = attackCounts[code] || 0;
        const maxCount = Math.max(...Object.values(attackCounts), 1);
        const intensity = count / maxCount;

        return getHeatmapColor(intensity);
    }, [attackCounts]);

    // Polygon kenar rengi
    const getPolygonStrokeColor = useCallback((d) => {
        const code = d?.properties?.ISO_A2;
        if (code === 'TR') return '#00ff88';
        if (d === hoverD) return '#ffffff';
        const count = attackCounts[code] || 0;
        if (count > 10) return 'rgba(255, 100, 100, 0.6)';
        return 'rgba(100, 150, 200, 0.3)';
    }, [hoverD, attackCounts]);

    // Ülke tıklama
    const handlePolygonClick = useCallback((polygon) => {
        const code = polygon?.properties?.ISO_A2;
        const name = polygon?.properties?.NAME;
        setSelectedCountry({ code, name, count: attackCounts[code] || 0 });

        // Ülkeye zoom
        const countryData = COUNTRIES_DATA[code];
        if (countryData && globeRef.current) {
            globeRef.current.pointOfView({
                lat: countryData.lat,
                lng: countryData.lng,
                altitude: 1.5
            }, 1000);
        }
    }, [attackCounts]);

    // Container boyutunu izle
    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                const { offsetWidth, offsetHeight } = containerRef.current;
                setDimensions({ width: offsetWidth, height: offsetHeight });
            }
        };
        updateDimensions();
        window.addEventListener('resize', updateDimensions);
        return () => window.removeEventListener('resize', updateDimensions);
    }, []);

    // Globe başlangıç ayarları
    useEffect(() => {
        if (globeRef.current) {
            // Daha uzaktan başla - tüm dünya görünsün
            globeRef.current.pointOfView({ lat: 20, lng: 0, altitude: 2.2 }, 1500);
            globeRef.current.controls().autoRotate = isLive && !selectedCountry;
            globeRef.current.controls().autoRotateSpeed = 0.3;
        }
    }, [isLive, selectedCountry]);

    return (
        <div ref={containerRef} className="relative w-full h-full min-h-[500px] bg-black rounded-xl overflow-hidden">
            {/* Sol Panel - İstatistikler */}
            <div className="absolute top-3 left-3 z-20 w-56 bg-gray-900/95 backdrop-blur rounded-xl border border-gray-700 overflow-hidden">
                <div className="bg-gradient-to-r from-cyan-600 to-blue-600 px-3 py-2 flex justify-between items-center">
                    <h3 className="text-white font-bold text-sm">📊 Canlı İstatistikler</h3>
                    <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}
                        title={wsConnected ? 'WebSocket Bağlı' : 'Bağlantı Kesik'} />
                </div>
                <div className="p-3 space-y-3">
                    <div className="grid grid-cols-2 gap-2">
                        <div className="bg-gray-800 rounded-lg p-2 text-center">
                            <div className="text-2xl font-bold text-red-400">{stats.total}</div>
                            <div className="text-xs text-gray-400">Toplam Saldırı</div>
                        </div>
                        <div className="bg-gray-800 rounded-lg p-2 text-center">
                            <div className="text-2xl font-bold text-green-400">{stats.blocked}</div>
                            <div className="text-xs text-gray-400">Engellenen</div>
                        </div>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-2">
                        <div className="flex justify-between items-center mb-1">
                            <span className="text-xs text-gray-400">Kritik</span>
                            <span className="text-red-400 font-bold">{stats.critical}</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-1.5">
                            <div
                                className="bg-red-500 h-1.5 rounded-full transition-all"
                                style={{ width: `${Math.min((stats.critical / Math.max(stats.total, 1)) * 100, 100)}%` }}
                            />
                        </div>
                    </div>

                    {/* ML Stats Panel */}
                    <div className="bg-gradient-to-r from-purple-900/50 to-pink-900/50 rounded-lg p-2 border border-purple-500/30">
                        <div className="flex items-center gap-1 mb-2">
                            <span className="text-purple-400">🤖</span>
                            <span className="text-xs font-bold text-purple-300">ML Tahminler</span>
                        </div>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                            <div>
                                <div className="text-purple-400 font-bold">{stats.mlThreats || 0}</div>
                                <div className="text-gray-400">Tehdit</div>
                            </div>
                            <div>
                                <div className="text-cyan-400 font-bold">{((stats.avgConfidence || 0) * 100).toFixed(0)}%</div>
                                <div className="text-gray-400">Güven</div>
                            </div>
                        </div>
                        <div className="mt-2 w-full bg-gray-700 rounded-full h-1">
                            <div
                                className="bg-gradient-to-r from-purple-500 to-pink-500 h-1 rounded-full transition-all"
                                style={{ width: `${(stats.avgConfidence || 0) * 100}%` }}
                            />
                        </div>
                    </div>

                    <div className="space-y-1">
                        <div className="text-xs text-gray-400 mb-1">🏴‍☠️ En Çok Saldıran</div>
                        {stats.topCountries.map(([code, count]) => (
                            <div key={code} className="flex justify-between items-center text-xs">
                                <span className="text-gray-300">{COUNTRIES_DATA[code]?.name || code}</span>
                                <span className="text-red-400 font-mono">{count}</span>
                            </div>
                        ))}
                        {stats.topCountries.length === 0 && (
                            <div className="text-xs text-gray-500">Veri bekleniyor...</div>
                        )}
                    </div>
                </div>
            </div>

            {/* Globe */}
            <Globe
                ref={globeRef}
                globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
                bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
                backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"
                atmosphereColor="#4da6ff"
                atmosphereAltitude={0.2}

                // Ülke polygon'ları (Heatmap)
                polygonsData={countries.features}
                polygonAltitude={(d) => d?.properties?.ISO_A2 === 'TR' ? 0.02 : 0.01}
                polygonCapColor={getPolygonColor}
                polygonSideColor={() => 'rgba(30, 58, 92, 0.15)'}
                polygonStrokeColor={getPolygonStrokeColor}
                polygonLabel={(d) => `
                    <div style="background: rgba(0,0,0,0.9); padding: 10px 14px; border-radius: 10px; border: 1px solid #444; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
                        <div style="font-size: 15px; font-weight: bold; margin-bottom: 4px;">${d?.properties?.NAME || 'Bilinmiyor'}</div>
                        <div style="color: #888; font-size: 12px;">Kod: ${d?.properties?.ISO_A2 || '-'}</div>
                        <div style="color: #ff6b6b; font-size: 14px; margin-top: 4px;">🔴 Saldırı: <b>${attackCounts[d?.properties?.ISO_A2] || 0}</b></div>
                    </div>
                `}
                onPolygonClick={handlePolygonClick}
                onPolygonHover={setHoverD}

                // Saldırı Arc'ları
                arcsData={arcsData}
                arcColor="color"
                arcDashLength={0.5}
                arcDashGap={0.2}
                arcDashAnimateTime={1200}
                arcStroke="stroke"
                arcAltitudeAutoScale={0.5}

                // Ring Pulse (Türkiye)
                ringsData={ringsData}
                ringColor="color"
                ringMaxRadius="maxR"
                ringPropagationSpeed="propagationSpeed"
                ringRepeatPeriod="repeatPeriod"

                // Hexbin Layer
                hexBinPointsData={hexbinData}
                hexBinPointWeight="weight"
                hexBinResolution={3}
                hexAltitude={(d) => d.sumWeight * 0.01}
                hexTopColor={(d) => getHeatmapColor(d.sumWeight / 10)}
                hexSideColor={(d) => getHeatmapColor(d.sumWeight / 10)}
                hexBinMerge={true}

                // Label Markers
                labelsData={labelsData}
                labelLat="lat"
                labelLng="lng"
                labelText="text"
                labelSize="size"
                labelDotRadius={0.4}
                labelColor="color"
                labelResolution={2}
                labelAltitude={0.01}

                animateIn={true}
                width={dimensions.width}
                height={dimensions.height - 60}
            />

            {/* Sağ üst - Kontroller */}
            <div className="absolute top-3 right-3 z-20 flex flex-col gap-2">
                <button
                    onClick={onToggleView}
                    className="px-3 py-1.5 bg-gray-900/90 hover:bg-gray-800 text-white text-sm rounded-lg border border-gray-700 backdrop-blur"
                >
                    ← 2D
                </button>
                <button
                    onClick={() => setSoundEnabled(!soundEnabled)}
                    className={`px-3 py-1.5 text-sm rounded-lg border backdrop-blur ${soundEnabled
                        ? 'bg-green-900/90 border-green-700 text-green-300'
                        : 'bg-gray-900/90 border-gray-700 text-gray-400'
                        }`}
                >
                    {soundEnabled ? '🔊' : '🔇'}
                </button>
            </div>

            {/* Türkiye göstergesi */}
            <div className="absolute top-20 right-3 z-20 bg-gray-900/95 backdrop-blur rounded-lg p-2.5 border border-emerald-500/50">
                <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
                    <span className="text-emerald-400 text-sm font-medium">🇹🇷 Türkiye</span>
                </div>
                <div className="text-xs text-gray-400 mt-1">Hedef Konum</div>
            </div>

            {/* Renk skalası */}
            <div className="absolute top-44 right-3 z-20 bg-gray-900/90 backdrop-blur rounded-lg p-2 border border-gray-700 text-xs">
                <div className="text-gray-400 mb-1.5">Isı Haritası</div>
                <div className="flex items-center gap-1">
                    <div className="w-12 h-2 rounded" style={{ background: 'linear-gradient(to right, #22c55e, #ffcc00, #ff6600, #ff0040)' }}></div>
                </div>
                <div className="flex justify-between text-[10px] text-gray-500 mt-0.5">
                    <span>Düşük</span>
                    <span>Yüksek</span>
                </div>
            </div>

            {/* Seçili ülke paneli */}
            {selectedCountry && (
                <div className="absolute bottom-20 left-3 z-20 bg-gray-900/95 backdrop-blur rounded-xl p-4 border border-gray-600 min-w-[220px]">
                    <div className="flex justify-between items-start mb-3">
                        <div>
                            <div className="text-white font-bold">{selectedCountry.name}</div>
                            <div className="text-xs text-gray-400">Kod: {selectedCountry.code}</div>
                        </div>
                        <button onClick={() => setSelectedCountry(null)} className="text-gray-400 hover:text-white">✕</button>
                    </div>
                    <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                            <span className="text-gray-400">Saldırı Sayısı</span>
                            <span className="text-red-400 font-bold">{selectedCountry.count}</span>
                        </div>
                        <div className="flex justify-between text-sm">
                            <span className="text-gray-400">Tehdit Seviyesi</span>
                            <span style={{ color: getThreatColor(selectedCountry.count) }} className="font-bold">
                                {selectedCountry.count > 20 ? 'KRİTİK' : selectedCountry.count > 10 ? 'YÜKSEK' : selectedCountry.count > 5 ? 'ORTA' : 'DÜŞÜK'}
                            </span>
                        </div>
                        <div className="mt-2 bg-gray-700 rounded-full h-2 overflow-hidden">
                            <div
                                className="h-full rounded-full transition-all"
                                style={{ width: `${Math.min(selectedCountry.count * 4, 100)}%`, backgroundColor: getThreatColor(selectedCountry.count) }}
                            />
                        </div>
                    </div>
                </div>
            )}

            {/* Timeline - Alt */}
            <div className="absolute bottom-3 left-3 right-3 z-10 bg-gray-900/90 backdrop-blur rounded-lg p-2 border border-gray-700">
                <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">📊 Saldırı Zaman Çizelgesi</span>
                    <span className="text-xs text-cyan-400">{attacks.length} aktif</span>
                </div>
                <div className="flex items-end gap-0.5 h-8">
                    {timelineData.map((point, i) => {
                        const maxCount = Math.max(...timelineData.map(p => p.count), 1);
                        const height = (point.count / maxCount) * 100;
                        return (
                            <div
                                key={i}
                                className="flex-1 bg-gradient-to-t from-cyan-600 to-cyan-400 rounded-t transition-all"
                                style={{ height: `${Math.max(height, 5)}%`, opacity: 0.3 + (i / timelineData.length) * 0.7 }}
                                title={`${point.count} saldırı`}
                            />
                        );
                    })}
                    {timelineData.length === 0 && (
                        <div className="flex-1 text-center text-xs text-gray-500 py-2">Veri yükleniyor...</div>
                    )}
                </div>
            </div>
        </div>
    );
}
