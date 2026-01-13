import React, { useState, useEffect, useRef, Suspense, lazy } from 'react';
import api from '../services/api';

// 3D Globe'u lazy load ile yükle (performans için)
const Globe3D = lazy(() => import('../components/Globe3D'));

const AttackMap = () => {
    const [attacks, setAttacks] = useState([]);
    const [stats, setStats] = useState(null);
    const [countries, setCountries] = useState([]);
    const [loading, setLoading] = useState(true);
    const [isLive, setIsLive] = useState(true);
    const [viewMode, setViewMode] = useState('3d'); // '2d' veya '3d'
    const _canvasRef = useRef(null);

    useEffect(() => {
        loadData();
        const interval = setInterval(() => {
            if (isLive) {
                loadAttacks();
            }
        }, 3000);
        return () => clearInterval(interval);
    }, [isLive]);

    const loadData = async () => {
        try {
            const [attacksRes, statsRes, countriesRes] = await Promise.all([
                api.get('/attack-map/live?limit=30'),
                api.get('/attack-map/stats'),
                api.get('/attack-map/countries')
            ]);
            setAttacks(attacksRes.data.data?.attacks || []);
            setStats(statsRes.data.data);
            setCountries(countriesRes.data.data?.countries || []);
        } catch (error) {
            console.error('Saldırı haritası verisi yüklenirken hata:', error);
        } finally {
            setLoading(false);
        }
    };

    const loadAttacks = async () => {
        try {
            const response = await api.get('/attack-map/live?limit=10');
            const newAttacks = response.data.data?.attacks || [];
            setAttacks(prev => {
                const existingIds = new Set(prev.map(a => a.id));
                const uniqueNew = newAttacks.filter(a => !existingIds.has(a.id));
                return [...uniqueNew, ...prev].slice(0, 50);
            });
        } catch (error) {
            console.error('Saldırılar yüklenirken hata:', error);
        }
    };

    // Koordinat dönüşümü (2D için)
    const latLngToXY = (lat, lng, width, height) => {
        const x = ((lng + 180) / 360) * width;
        const y = ((90 - lat) / 180) * height;
        return { x, y };
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

    const getRiskText = (risk) => {
        switch (risk) {
            case 'high': return 'YÜKSEK';
            case 'medium': return 'ORTA';
            case 'low': return 'DÜŞÜK';
            default: return risk?.toUpperCase() || 'BİLİNMİYOR';
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen bg-gray-900">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-cyan-500 mx-auto"></div>
                    <p className="mt-4 text-cyan-400">Saldırı Haritası Yükleniyor...</p>
                </div>
            </div>
        );
    }

    // 2D Harita Render
    const render2DMap = () => (
        <div className="relative bg-gradient-to-b from-gray-900 via-slate-900 to-gray-900 rounded-lg overflow-hidden" style={{ height: '500px' }}>
            {/* Yıldız efekti */}
            <div className="absolute inset-0">
                {[...Array(50)].map((_, i) => (
                    <div
                        key={`star-${i}`}
                        className="absolute w-1 h-1 bg-white rounded-full opacity-30"
                        style={{
                            left: `${Math.random() * 100}%`,
                            top: `${Math.random() * 100}%`,
                            animation: `twinkle ${2 + Math.random() * 3}s infinite`
                        }}
                    />
                ))}
            </div>

            {/* SVG Dünya Haritası */}
            <svg viewBox="0 0 1000 500" className="absolute inset-0 w-full h-full">
                <defs>
                    <linearGradient id="oceanGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#0c1929" />
                        <stop offset="100%" stopColor="#0a1628" />
                    </linearGradient>
                    <linearGradient id="landGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#1e3a5f" />
                        <stop offset="100%" stopColor="#0f2744" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                <rect width="1000" height="500" fill="url(#oceanGradient)" />

                {/* Grid */}
                {[...Array(18)].map((_, i) => (
                    <line key={`h-${i}`} x1="0" y1={i * 28} x2="1000" y2={i * 28} stroke="#1e3a5f" strokeWidth="0.5" opacity="0.3" />
                ))}
                {[...Array(36)].map((_, i) => (
                    <line key={`v-${i}`} x1={i * 28} y1="0" x2={i * 28} y2="500" stroke="#1e3a5f" strokeWidth="0.5" opacity="0.3" />
                ))}

                {/* Kıtalar */}
                <path d="M120,80 Q180,70 220,90 Q280,100 300,140 Q320,180 280,220 Q240,250 180,260 Q120,250 100,200 Q90,150 120,80" fill="url(#landGradient)" stroke="#2d5a87" strokeWidth="1" />
                <path d="M200,280 Q240,260 260,300 Q280,360 260,420 Q240,450 200,440 Q160,420 170,360 Q180,300 200,280" fill="url(#landGradient)" stroke="#2d5a87" strokeWidth="1" />
                <path d="M440,100 Q500,80 540,100 Q560,130 540,160 Q500,180 460,170 Q430,150 440,100" fill="url(#landGradient)" stroke="#2d5a87" strokeWidth="1" />
                <path d="M460,200 Q520,180 560,220 Q580,280 560,350 Q520,400 480,390 Q440,360 450,300 Q450,240 460,200" fill="url(#landGradient)" stroke="#2d5a87" strokeWidth="1" />
                <path d="M560,80 Q680,60 800,100 Q880,140 900,200 Q880,260 800,280 Q720,290 640,260 Q580,220 560,160 Q550,120 560,80" fill="url(#landGradient)" stroke="#2d5a87" strokeWidth="1" />
                <path d="M780,340 Q840,320 880,360 Q900,400 860,430 Q820,450 780,430 Q750,400 780,340" fill="url(#landGradient)" stroke="#2d5a87" strokeWidth="1" />

                {/* Türkiye */}
                <ellipse cx="555" cy="175" rx="25" ry="12" fill="#00ff88" opacity="0.3" filter="url(#glow)">
                    <animate attributeName="opacity" values="0.3;0.6;0.3" dur="2s" repeatCount="indefinite" />
                </ellipse>
                <ellipse cx="555" cy="175" rx="8" ry="5" fill="#00ff88" />
                <text x="555" y="195" textAnchor="middle" fill="#00ff88" fontSize="10" fontWeight="bold">TR</text>
            </svg>

            {/* Saldırı çizgileri */}
            <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1000 500">
                {attacks.slice(0, 20).map((attack, index) => {
                    if (!attack?.source?.lat || !attack?.target?.lat) return null;
                    const source = latLngToXY(attack.source.lat, attack.source.lng, 1000, 500);
                    const target = latLngToXY(attack.target.lat, attack.target.lng, 1000, 500);
                    const color = getSeverityColor(attack.severity);
                    const midX = (source.x + target.x) / 2;
                    const midY = Math.min(source.y, target.y) - 50;

                    return (
                        <g key={`attack-${attack.id}-${index}`}>
                            <path
                                d={`M${source.x},${source.y} Q${midX},${midY} ${target.x},${target.y}`}
                                stroke={color}
                                strokeWidth="2"
                                fill="none"
                                opacity="0.7"
                                strokeDasharray="5,5"
                            >
                                <animate attributeName="stroke-dashoffset" from="10" to="0" dur="1s" repeatCount="indefinite" />
                            </path>
                            <circle cx={source.x} cy={source.y} r="6" fill={color} opacity="0.8">
                                <animate attributeName="r" values="4;8;4" dur="1.5s" repeatCount="indefinite" />
                            </circle>
                            <circle cx={target.x} cy={target.y} r="8" fill="#00ff88" opacity="0.9">
                                <animate attributeName="opacity" values="0.5;1;0.5" dur="1s" repeatCount="indefinite" />
                            </circle>
                        </g>
                    );
                })}
            </svg>

            {/* Legend */}
            <div className="absolute bottom-4 left-4 bg-gray-800/90 backdrop-blur rounded-lg p-3 border border-gray-700">
                <p className="text-xs text-gray-400 mb-2">Önem Derecesi</p>
                <div className="flex gap-3 text-xs">
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-red-500"></span> Kritik</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-orange-500"></span> Yüksek</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-yellow-500"></span> Orta</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-green-500"></span> Düşük</span>
                </div>
            </div>

            {/* Canlı gösterge */}
            <div className="absolute top-4 right-4 flex items-center gap-2 bg-gray-800/90 backdrop-blur rounded-lg px-3 py-2 border border-gray-700">
                <div className={`w-3 h-3 rounded-full ${isLive ? 'bg-red-500 animate-pulse' : 'bg-gray-500'}`}></div>
                <span className="text-xs font-medium text-white">{isLive ? 'CANLI' : 'DURAKLATILDI'}</span>
            </div>
        </div>
    );

    // 3D Harita Render
    const render3DMap = () => (
        <div style={{ height: '500px' }} className="rounded-lg overflow-hidden">
            <Suspense fallback={
                <div className="flex items-center justify-center h-full bg-gray-900">
                    <div className="text-center">
                        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500 mx-auto"></div>
                        <p className="mt-4 text-cyan-400 text-sm">3D Küre Yükleniyor...</p>
                    </div>
                </div>
            }>
                <Globe3D
                    attacks={attacks}
                    isLive={isLive}
                    onToggleView={() => setViewMode('2d')}
                />
            </Suspense>
        </div>
    );

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            {/* Başlık */}
            <div className="flex justify-between items-center mb-6">
                <div>
                    <h1 className="text-3xl font-bold text-cyan-400">🌍 Küresel Saldırı Haritası</h1>
                    <p className="text-gray-400">Gerçek zamanlı siber saldırı görselleştirmesi</p>
                </div>
                <div className="flex items-center gap-4">
                    {/* 2D/3D Toggle */}
                    <div className="flex bg-gray-800 rounded-lg p-1 border border-gray-700">
                        <button
                            onClick={() => setViewMode('2d')}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition ${viewMode === '2d'
                                    ? 'bg-cyan-600 text-white'
                                    : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            2D
                        </button>
                        <button
                            onClick={() => setViewMode('3d')}
                            className={`px-4 py-2 rounded-md text-sm font-medium transition ${viewMode === '3d'
                                    ? 'bg-cyan-600 text-white'
                                    : 'text-gray-400 hover:text-white'
                                }`}
                        >
                            3D 🌐
                        </button>
                    </div>

                    <button
                        onClick={() => setIsLive(!isLive)}
                        className={`px-4 py-2 rounded-lg font-medium transition ${isLive
                            ? 'bg-green-600 hover:bg-green-700'
                            : 'bg-gray-600 hover:bg-gray-700'
                            }`}
                    >
                        {isLive ? '🔴 CANLI' : '⏸️ DURAKLATILDI'}
                    </button>
                    <button
                        onClick={loadData}
                        className="px-4 py-2 bg-cyan-600 hover:bg-cyan-700 rounded-lg font-medium"
                    >
                        🔄 Yenile
                    </button>
                </div>
            </div>

            {/* İstatistik Kartları */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4 mb-6">
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Saldırılar (24s)</p>
                        <p className="text-2xl font-bold text-red-400">{stats.total_attacks_24h?.toLocaleString() || 0}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Engellenen</p>
                        <p className="text-2xl font-bold text-green-400">{stats.blocked_24h?.toLocaleString() || 0}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Aktif Tehditler</p>
                        <p className="text-2xl font-bold text-orange-400">{stats.active_threats || 0}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Etkilenen Ülkeler</p>
                        <p className="text-2xl font-bold text-cyan-400">{stats.countries_affected || 0}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">En Sık Saldırı</p>
                        <p className="text-xl font-bold text-purple-400">{stats.top_attack_type || '-'}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Saldırı/dk</p>
                        <p className="text-2xl font-bold text-yellow-400">{stats.avg_attacks_per_minute || 0}</p>
                    </div>
                </div>
            )}

            {/* Ana İçerik */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Harita (2D veya 3D) */}
                <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <h3 className="text-lg font-semibold text-cyan-400 mb-4">
                        {viewMode === '3d' ? '🌐 3D Dünya Küresi' : '🗺️ Dünya Haritası'}
                    </h3>
                    {viewMode === '3d' ? render3DMap() : render2DMap()}
                </div>

                {/* Canlı Saldırı Akışı */}
                <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                    <h3 className="text-lg font-semibold text-cyan-400 mb-4">⚡ Canlı Saldırı Akışı</h3>
                    <div className="space-y-2 max-h-[500px] overflow-y-auto">
                        {attacks.length === 0 ? (
                            <div className="text-center text-gray-500 py-8">
                                <p>Henüz saldırı kaydı yok</p>
                                <p className="text-sm mt-2">Honeypot verileri bekleniyor...</p>
                            </div>
                        ) : (
                            attacks.slice(0, 15).map((attack, index) => (
                                <div
                                    key={`feed-${attack.id}-${index}`}
                                    className="bg-gray-700/50 rounded-lg p-3 border-l-4 hover:bg-gray-700/70 transition-colors"
                                    style={{ borderColor: getSeverityColor(attack.severity) }}
                                >
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <p className="font-medium text-white">{attack.attack_type}</p>
                                            <p className="text-xs text-gray-400">
                                                {attack.source?.name || 'Bilinmiyor'} → {attack.target?.name || 'Türkiye'}
                                            </p>
                                        </div>
                                        <span className={`text-xs px-2 py-1 rounded ${attack.blocked ? 'bg-green-600' : 'bg-red-600'}`}>
                                            {attack.blocked ? 'ENGELLENDİ' : 'AKTİF'}
                                        </span>
                                    </div>
                                    <div className="flex justify-between mt-2 text-xs text-gray-500">
                                        <span className="font-mono">{attack.source?.ip || '-'}</span>
                                        <span>{attack.timestamp ? new Date(attack.timestamp).toLocaleTimeString('tr-TR') : '-'}</span>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* En Çok Saldıran Ülkeler Tablosu */}
            <div className="mt-6 bg-gray-800 rounded-lg p-4 border border-gray-700">
                <h3 className="text-lg font-semibold text-cyan-400 mb-4">🏴‍☠️ En Çok Saldıran Ülkeler</h3>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="text-left text-gray-400 border-b border-gray-700">
                                <th className="pb-3">Ülke</th>
                                <th className="pb-3">Gönderilen Saldırı</th>
                                <th className="pb-3">Alınan Saldırı</th>
                                <th className="pb-3">Engelleme %</th>
                                <th className="pb-3">Risk Seviyesi</th>
                            </tr>
                        </thead>
                        <tbody>
                            {countries.length === 0 ? (
                                <tr>
                                    <td colSpan="5" className="py-8 text-center text-gray-500">
                                        Ülke verisi bekleniyor...
                                    </td>
                                </tr>
                            ) : (
                                countries.slice(0, 10).map((country, index) => (
                                    <tr key={`country-${country.code}-${index}`} className="border-b border-gray-700/50 hover:bg-gray-700/30 transition-colors">
                                        <td className="py-3">
                                            <span className="font-medium">{country.name}</span>
                                            <span className="ml-2 text-gray-500">({country.code})</span>
                                        </td>
                                        <td className="py-3 text-red-400">{country.attacks_sent?.toLocaleString() || 0}</td>
                                        <td className="py-3 text-orange-400">{country.attacks_received?.toLocaleString() || 0}</td>
                                        <td className="py-3 text-green-400">{country.blocked || 0}%</td>
                                        <td className="py-3">
                                            <span className={`px-2 py-1 rounded text-xs ${country.risk_level === 'high' ? 'bg-red-600' :
                                                country.risk_level === 'medium' ? 'bg-yellow-600' : 'bg-green-600'
                                                }`}>
                                                {getRiskText(country.risk_level)}
                                            </span>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Veri Kaynağı Bilgisi */}
            <div className="mt-4 text-center text-xs text-gray-500">
                <p>📡 Veri Kaynakları: Honeypot Yakalama | Ağ Bağlantıları (psutil) | Güvenlik Duvarı Logları</p>
            </div>

            {/* CSS Animasyonları */}
            <style>{`
                @keyframes twinkle {
                    0%, 100% { opacity: 0.3; }
                    50% { opacity: 0.8; }
                }
            `}</style>
        </div>
    );
};

export default AttackMap;
