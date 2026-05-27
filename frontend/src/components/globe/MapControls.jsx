import React from 'react';

/**
 * Modern floating control panel for the 3D Attack Map.
 * Includes time-travel playback controls and severity/type filters.
 */
export default function MapControls({
    isLive,
    toggleLive,
    timelineData,
    playbackTime,
    setPlayback,
    filters,
    updateFilter
}) {
    const formatTime = (ts) => new Date(ts).toLocaleTimeString('tr-TR', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

    return (
        <div className="absolute top-4 left-4 z-20 flex flex-col gap-4">
            {/* Playback & Live Status Panel */}
            <div className="bg-slate-900/80 backdrop-blur-md rounded-xl border border-slate-700/50 p-4 shadow-2xl w-80">
                <div className="flex items-center justify-between mb-4">
                    <h3 className="text-sm font-bold text-[var(--hud-text)] flex items-center gap-2">
                        <span className="text-[var(--hud-cyan)]">⏱️</span> Zaman Makinesi
                    </h3>
                    <button
                        onClick={toggleLive}
                        className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${isLive
                                ? 'bg-red-500/20 text-red-400 border border-red-500/50 shadow-[0_0_10px_rgba(239,68,68,0.3)] animate-pulse'
                                : 'bg-slate-800 text-slate-400 border border-slate-600 hover:text-[var(--hud-text)]'
                            }`}
                    >
                        {isLive ? '🔴 CANLI' : '▶️ CANLIYA DÖN'}
                    </button>
                </div>

                {/* Timeline Bar Chart */}
                <div className="h-12 flex items-end gap-1 mb-2 group cursor-pointer">
                    {timelineData.map((point) => {
                        const isPlaying = playbackTime && Math.abs(point.time - playbackTime) < 2000;
                        const maxCount = Math.max(...timelineData.map(p => p.count), 1);
                        const height = Math.max((point.count / maxCount) * 100, 4);

                        return (
                            <div
                                key={point.id}
                                onClick={() => setPlayback(point.time)}
                                className={`flex-1 rounded-t-sm transition-all duration-300 hover:bg-cyan-300 ${isPlaying ? 'bg-cyan-400 opacity-100 shadow-[0_0_6px_rgba(56,189,248,0.4)]' : 'bg-cyan-600/50 opacity-50'
                                    }`}
                                style={{ height: `${height}%` }}
                                title={`${formatTime(point.time)}: ${point.count} Saldırı`}
                            />
                        );
                    })}
                </div>

                <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                    <span>{timelineData[0] ? formatTime(timelineData[0].time) : '-'}</span>
                    <span className="text-[var(--hud-cyan)]">{playbackTime ? formatTime(playbackTime) : 'Şimdi'}</span>
                    <span>{timelineData[timelineData.length - 1] ? formatTime(timelineData[timelineData.length - 1].time) : '-'}</span>
                </div>
            </div>

            {/* Filters Panel */}
            <div className="bg-slate-900/80 backdrop-blur-md rounded-xl border border-slate-700/50 p-4 shadow-2xl w-80">
                <h3 className="text-sm font-bold text-[var(--hud-text)] mb-3 flex items-center gap-2">
                    <span className="text-purple-400">🛡️</span> Dinamik Filtreler
                </h3>

                <div className="space-y-3">
                    {/* Minimum Severity Filter */}
                    <div>
                        <label className="text-xs text-slate-400 mb-1 block">Min. Şiddet Seviyesi</label>
                        <select
                            value={filters.minSeverity}
                            onChange={(e) => updateFilter('minSeverity', e.target.value)}
                            className="w-full bg-slate-800 text-sm text-[var(--hud-text)] rounded-lg border border-slate-700 px-3 py-1.5 focus:border-cyan-500 outline-none"
                        >
                            <option value="low">Low (Tümü)</option>
                            <option value="medium">Medium ve Üzeri</option>
                            <option value="high">High ve Üzeri</option>
                            <option value="critical">Sadece Critical</option>
                        </select>
                    </div>

                    {/* Threat Type Filter */}
                    <div>
                        <label className="text-xs text-slate-400 mb-1 block">Saldırı Türü</label>
                        <select
                            value={filters.threatType}
                            onChange={(e) => updateFilter('threatType', e.target.value)}
                            className="w-full bg-slate-800 text-sm text-[var(--hud-text)] rounded-lg border border-slate-700 px-3 py-1.5 focus:border-purple-500 outline-none"
                        >
                            <option value="all">Tüm Türler</option>
                            <option value="DDoS">DDoS</option>
                            <option value="Brute Force">Brute Force</option>
                            <option value="SQL Injection">SQL Injection</option>
                            <option value="Malware">Malware</option>
                        </select>
                    </div>

                    {/* Blocked Only Toggle */}
                    <div className="flex items-center justify-between pt-1">
                        <span className="text-xs text-slate-300">Sadece Engellenenler</span>
                        <button
                            onClick={() => updateFilter('showOnlyBlocked', !filters.showOnlyBlocked)}
                            className={`w-10 h-5 rounded-full p-1 transition-colors ${filters.showOnlyBlocked ? 'bg-green-500' : 'bg-slate-700'}`}
                        >
                            <div className={`w-3 h-3 rounded-full bg-white transition-transform ${filters.showOnlyBlocked ? 'translate-x-5' : 'translate-x-0'}`} />
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
}
