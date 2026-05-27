import React, { useState, Suspense, lazy, useEffect, useRef, useCallback } from 'react';
import useAttackMap from '../../hooks/useAttackMap';
import MapControls from '../../components/globe/MapControls';
import MapTooltip from '../../components/globe/MapTooltip';
import GlobeHUD from '../../components/globe/GlobeHUD';

const AdvancedGlobe3D = lazy(() => import('../../components/globe/AdvancedGlobe3D'));

export default function AttackMap() {
    const {
        attacks,
        stats,
        countries,
        loading,
        isLive,
        timelineData,
        playbackTime,
        filters,
        toggleLive,
        setPlayback,
        updateFilter
    } = useAttackMap();

    const [viewMode, setViewMode] = useState('3d');
    const [hoveredAttack, setHoveredAttack] = useState(null);
    const [hoveredCountry, setHoveredCountry] = useState(null);
    const [selectedCountry, setSelectedCountry] = useState(null);

    const handleCountryClick = useCallback((country) => {
        setSelectedCountry(country);
    }, []);

    // Auto-resize the globe container
    const containerRef = useRef(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    useEffect(() => {
        const updateDimensions = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.offsetWidth,
                    height: containerRef.current.offsetHeight
                });
            }
        };
        updateDimensions();
        // Allow a small delay for layout to settle
        const timeoutId = setTimeout(updateDimensions, 100);

        window.addEventListener('resize', updateDimensions);
        return () => {
            window.removeEventListener('resize', updateDimensions);
            clearTimeout(timeoutId);
        }
    }, [viewMode]); // Update dimensions when switching views

    // Severity colors for side feed
    const getSeverityColor = (severity) => {
        switch (severity) {
            case 'critical': return '#ef4444';
            case 'high': return '#f97316';
            case 'medium': return '#eab308';
            case 'low': return '#22c55e';
            default: return '#06b6d4';
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen bg-[#070b19] font-sans">
                <div className="text-center">
                    <div className="relative w-20 h-20 mx-auto mb-6">
                        <div className="absolute inset-0 border-4 border-[var(--hud-cyan)]/30 rounded-full"></div>
                        <div className="absolute inset-0 border-4 border-[var(--hud-cyan)] rounded-full border-t-transparent animate-spin"></div>
                        <div className="absolute inset-2 border-4 border-purple-500/30 rounded-full"></div>
                        <div className="absolute inset-2 border-4 border-purple-400 rounded-full border-b-transparent animate-spin-slow"></div>
                    </div>
                    <p className="text-[var(--hud-cyan)] font-mono tracking-widest text-sm animate-pulse">SİSTEM YÜKLENİYOR // INITIALIZING...</p>
                </div>
            </div>
        );
    }

    // Harita verileri (React Globe GL için özellikler dizisi eklememiz gerekiyor, eğer countriesGeoJSON'ı useAttackMap'ten almadıysanız)
    // Önceki Globe3D.jsx implementasyonunuzda `countries` state'inde `features` vardı, aynı yapıyı kullanıyoruz:
    const geoJsonData = { features: [] }; // Burayı uygulamanızdaki asıl datayla doldurabilirsiniz veya AdvancedGlobe içinde halledebiliriz

    return (
        <div className="min-h-screen bg-[#070b19] text-[var(--hud-text)] p-6 font-sans overflow-hidden">
            {/* Header */}
            <header className="flex justify-between items-center mb-6 relative z-10">
                <div>
                    <h1 className="text-3xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-purple-500 tracking-tight">
                        CYBER<span className="text-[var(--hud-text)]">SPACE</span> COMMAND
                    </h1>
                    <p className="text-slate-400 text-sm mt-1 font-mono">GELİŞMİŞ SİBER TEHDİT İZLEME AĞI // V2.0</p>
                </div>

                <div className="flex items-center gap-4">
                    {/* View Mode Toggle */}
                    <div className="flex bg-slate-900/80 backdrop-blur rounded-lg p-1 border border-slate-700/50 shadow-inner">
                        <button
                            onClick={() => setViewMode('2d')}
                            className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${viewMode === '2d' ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-[var(--hud-text)] shadow-lg' : 'text-slate-400 hover:text-[var(--hud-text)]'
                                }`}
                        >
                            🗺️ 2D KART
                        </button>
                        <button
                            onClick={() => setViewMode('3d')}
                            className={`px-4 py-1.5 rounded-md text-sm font-bold transition-all ${viewMode === '3d' ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-[var(--hud-text)] shadow-lg' : 'text-slate-400 hover:text-[var(--hud-text)]'
                                }`}
                        >
                            🌐 3D KÜRE
                        </button>
                    </div>
                </div>
            </header>

            {/* Top Stats Strip */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6 relative z-10">
                <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-700/50 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-cyan-500"></div>
                    <p className="text-slate-400 text-xs font-mono mb-1">GÖRÜNTÜLENEN SALDIRI</p>
                    <p className="text-3xl font-black text-[var(--hud-cyan)]">{stats.total}</p>
                </div>
                <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-700/50 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-green-500"></div>
                    <p className="text-slate-400 text-xs font-mono mb-1">BAŞARIYLA ENGELLENDİ</p>
                    <p className="text-3xl font-black text-green-400">{stats.blocked}</p>
                </div>
                <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-700/50 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-red-500"></div>
                    <p className="text-slate-400 text-xs font-mono mb-1">KRİTİK TEHDİT ALARMI</p>
                    <p className="text-3xl font-black text-red-500 animate-pulse">{stats.critical}</p>
                </div>
                <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-700/50 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-purple-500"></div>
                    <p className="text-slate-400 text-xs font-mono mb-1">AI TESPİT ORANI</p>
                    <p className="text-3xl font-black text-purple-400">{Math.round((stats.avgConfidence || 0) * 100)}%</p>
                </div>
                <div className="bg-slate-900/60 backdrop-blur-md rounded-xl p-4 border border-slate-700/50 relative overflow-hidden group">
                    <div className="absolute top-0 left-0 w-1 h-full bg-orange-500"></div>
                    <p className="text-slate-400 text-xs font-mono mb-1">ZAMAN ÇİZELGESİ</p>
                    <p className="text-lg font-bold text-orange-400 mt-2">{playbackTime ? new Date(playbackTime).toLocaleTimeString('tr-TR') : 'CANLI YAYIN'}</p>
                </div>
            </div>

            {/* Main Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-220px)] relative z-10">

                {/* 3D Map Area */}
                <div className="lg:col-span-3 relative bg-slate-900/40 backdrop-blur-sm rounded-2xl border border-slate-700/50 overflow-hidden flex flex-col" ref={containerRef}>

                    {/* UI Overlay Controls */}
                    <div className="absolute top-4 left-4 right-4 z-20 pointer-events-none">
                        <MapControls
                            isLive={isLive}
                            toggleLive={toggleLive}
                            timelineData={timelineData}
                            playbackTime={playbackTime}
                            setPlayback={setPlayback}
                            filters={filters}
                            updateFilter={updateFilter}
                        />
                    </div>

                    {/* Globe Canvas */}
                    <div className="flex-1 w-full h-full relative z-0">
                        {viewMode === '3d' ? (
                            <Suspense fallback={
                                <div className="absolute inset-0 flex items-center justify-center bg-black/80">
                                    <div className="text-cyan-500 animate-pulse font-mono tracking-widest text-sm">3D MOTORU BAŞLATILIYOR...</div>
                                </div>
                            }>
                                {dimensions.width > 0 && dimensions.height > 0 && (
                                    <AdvancedGlobe3D
                                        attacks={attacks}
                                        isLive={isLive}
                                        countriesGeoJSON={geoJsonData}
                                        setHoveredAttack={setHoveredAttack}
                                        setHoveredCountry={setHoveredCountry}
                                        containerDimensions={dimensions}
                                        onCountryClick={handleCountryClick}
                                    />
                                )}
                            </Suspense>
                        ) : (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-slate-900/80 rounded-xl m-4 border border-slate-800">
                                <span className="text-4xl mb-4">🗺️</span>
                                <div className="text-slate-400 font-mono text-center max-w-md">
                                    <p className="mb-2 text-cyan-500 font-bold">2D Taktiksel Görünüm</p>
                                    <p className="text-sm">Yüksek performanslı 3D motoru şu an aktif. 2D görünüm modülü sisteme yükleniyor...</p>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Palantir-style HUD Overlay */}
                    <GlobeHUD
                        stats={stats}
                        attacks={attacks}
                        isLive={isLive}
                        selectedCountry={selectedCountry}
                        onCloseCountry={() => setSelectedCountry(null)}
                    />

                    {/* Dynamic Floating Tooltip for Hovered Attack */}
                    {hoveredAttack && (
                        <div className="absolute z-50 transition-all duration-300 pointer-events-none" style={{
                            bottom: '2rem',
                            right: '2rem',
                        }}>
                            <MapTooltip attack={hoveredAttack} />
                        </div>
                    )}
                </div>

                {/* Right Side Feed */}
                <div className="bg-slate-900/60 backdrop-blur-md rounded-2xl border border-slate-700/50 p-4 flex flex-col h-full overflow-hidden">
                    <h3 className="text-sm font-bold text-[var(--hud-text)] mb-4 flex items-center gap-2 border-b border-slate-700/50 pb-3">
                        <span className="relative flex h-3 w-3">
                            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[var(--hud-cyan)] opacity-75"></span>
                            <span className="relative inline-flex rounded-full h-3 w-3 bg-cyan-500"></span>
                        </span>
                        SALDIRI AKIŞI
                    </h3>

                    <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar">
                        {attacks.length === 0 ? (
                            <div className="text-center text-slate-500 py-10 font-mono text-xs">
                                Bekleyen tehdit bulunamadı.
                            </div>
                        ) : (
                            attacks.slice(0, 50).map((attack, index) => (
                                <div
                                    key={attack.id || `feed-${index}`}
                                    className="bg-slate-800/40 hover:bg-slate-800/80 transition-colors p-3 rounded-xl border-l-2 relative overflow-hidden group cursor-pointer"
                                    style={{ borderLeftColor: getSeverityColor(attack.severity) }}
                                    onMouseEnter={() => setHoveredAttack(attack)}
                                    onMouseLeave={() => setHoveredAttack(null)}
                                >
                                    <div className="flex justify-between items-start mb-1 relative z-10">
                                        <p className="font-bold text-sm text-[var(--hud-text)] truncate max-w-[150px]">{attack.threat_type || attack.attack_type || 'Bilinmeyen Saldırı'}</p>
                                        <span className="text-[10px] text-slate-400 font-mono">
                                            {attack.timestamp ? new Date(attack.timestamp).toLocaleTimeString('tr-TR') : ''}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-2 mb-2 text-xs text-slate-300 relative z-10">
                                        <span className="truncate max-w-[80px]" title={attack.source?.country || attack.source?.name}>{attack.source?.country || attack.source?.name || 'UNK'}</span>
                                        <strong className="text-slate-500">→</strong>
                                        <span className="truncate max-w-[80px]" title={attack.target?.country || attack.target?.name}>{attack.target?.country || attack.target?.name || 'TR'}</span>
                                    </div>
                                    <div className="flex justify-between items-center mt-2 relative z-10">
                                        <span className={`text-[10px] font-bold px-2 py-0.5 rounded border ${attack.blocked ? 'border-green-500/50 text-green-400 bg-green-500/10' : 'border-red-500/50 text-red-400 bg-red-500/10'
                                            }`}>
                                            {attack.blocked ? 'ENGELLENDİ' : 'AKTİF'}
                                        </span>
                                        {(attack.ml_prediction || attack.confidence) && (
                                            <span className="text-[10px] text-purple-400 font-mono font-bold bg-purple-500/10 px-2 py-0.5 rounded border border-purple-500/30">
                                                AI: %{((attack.ml_prediction?.confidence || attack.confidence || 0) * 100).toFixed(0)}
                                            </span>
                                        )}
                                    </div>

                                    {/* Subtly colored background based on severity */}
                                    <div
                                        className="absolute inset-0 opacity-[0.03] group-hover:opacity-10 transition-opacity"
                                        style={{ backgroundColor: getSeverityColor(attack.severity) }}
                                    />
                                    {/* Hover sweep effect */}
                                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:animate-[shimmer_1.5s_infinite]" />
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            {/* Background design elements */}
            <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
                <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-purple-600/10 rounded-full blur-[120px]"></div>
                <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-cyan-600/10 rounded-full blur-[120px]"></div>
                {/* Subtle grid background */}
                <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:50px_50px] [mask-image:radial-gradient(ellipse_at_center,black_40%,transparent_80%)]"></div>
            </div>

            <style>{`
                .custom-scrollbar::-webkit-scrollbar {
                    width: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-track {
                    background: rgba(15, 23, 42, 0.5);
                    border-radius: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb {
                    background: rgba(51, 65, 85, 0.8);
                    border-radius: 4px;
                }
                .custom-scrollbar::-webkit-scrollbar-thumb:hover {
                    background: rgba(71, 85, 105, 1);
                }
                @keyframes shimmer {
                    100% { transform: translateX(100%); }
                }
                @keyframes spin-slow {
                    to { transform: rotate(-360deg); }
                }
            `}</style>
        </div>
    );
}
