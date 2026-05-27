import React from 'react';

/**
 * Advanced Tooltip displaying XAI data and ML insights when hovering over an attack arc.
 */
export default function MapTooltip({ attack }) {
    if (!attack) return null;

    const { ml_prediction, source, target, severity, threat_type, blocked } = attack;

    // Severity colors
    const sColors = {
        critical: 'text-red-400',
        high: 'text-orange-400',
        medium: 'text-yellow-400',
        low: 'text-green-400'
    };

    // Severity bg colors for badge
    const badgeColors = {
        critical: 'bg-red-500/20 text-red-400 border-red-500/50',
        high: 'bg-orange-500/20 text-orange-400 border-orange-500/50',
        medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50',
        low: 'bg-green-500/20 text-green-400 border-green-500/50'
    };

    return (
        <div
            className="bg-slate-900/95 backdrop-blur-xl border border-slate-700 shadow-2xl rounded-xl p-4 min-w-[300px]"
            style={{
                boxShadow: '0 10px 40px rgba(0,0,0,0.8), 0 0 20px rgba(6, 182, 212, 0.1)',
                pointerEvents: 'none' // Don't block map interactions
            }}
        >
            {/* Header */}
            <div className="flex justify-between items-start mb-3 pb-3 border-b border-slate-800">
                <div>
                    <h4 className="text-[var(--hud-text)] font-bold text-lg leading-tight">{threat_type || 'Unknown Attack'}</h4>
                    <span className="text-slate-400 text-xs mt-1 block">ID: {attack.id?.substring(0, 8) || 'N/A'}</span>
                </div>
                <div className={`px-2 py-1 rounded-md text-xs font-bold border uppercase tracking-wider ${badgeColors[severity] || badgeColors.low}`}>
                    {severity}
                </div>
            </div>

            {/* Source to Target Path */}
            <div className="flex items-center justify-between bg-slate-800/50 rounded-lg p-2 mb-3">
                <div className="text-center">
                    <span className="text-3xl" role="img" aria-label="source flag">
                        {source?.country === 'TR' ? '🇹🇷' : source?.country === 'US' ? '🇺🇸' : source?.country === 'RU' ? '🇷🇺' : source?.country === 'CN' ? '🇨🇳' : '🏴‍☠️'}
                    </span>
                    <p className="text-[10px] text-slate-300 font-mono mt-1">{source?.ip || '0.0.0.0'}</p>
                    <p className="text-xs font-bold text-[var(--hud-text)]">{source?.country || 'Unknown'}</p>
                </div>

                <div className="flex flex-col items-center px-2">
                    <span className="text-[10px] text-slate-500 mb-1">HEDEF</span>
                    <div className="w-16 h-[1px] bg-slate-600 relative">
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-2 border-t border-r border-cyan-400 rotate-45" />
                        {/* Animated dash traveling */}
                        <div className="absolute left-0 top-0 h-[1px] w-4 bg-cyan-400 shadow-[0_0_4px_rgba(56,189,248,0.4)] animate-[dash_1s_linear_infinite]" />
                    </div>
                </div>

                <div className="text-center">
                    <span className="text-3xl" role="img" aria-label="target flag">🇹🇷</span>
                    <p className="text-[10px] text-slate-300 font-mono mt-1">{target?.ip || '195.142.x.x'}</p>
                    <p className="text-xs font-bold text-[var(--hud-text)]">{target?.name || 'Türkiye'}</p>
                </div>
            </div>

            {/* Action Status */}
            <div className={`mb-4 px-3 py-2 rounded-lg text-sm font-bold flex items-center gap-2 ${blocked ? 'bg-green-900/40 text-green-400 border border-green-800/50' : 'bg-red-900/40 text-red-400 border border-red-800/50'}`}>
                {blocked ? '🛡️ BAŞARIYLA ENGELLENDİ' : '⚠️ AKTİF TEHDİT'}
            </div>

            {/* Machine Learning / XAI Block */}
            {ml_prediction && (
                <div className="pt-3 border-t border-slate-800">
                    <h5 className="text-xs font-bold text-purple-400 mb-2 flex items-center gap-1">
                        🤖 AI KARAR MEKANİZMASI (XAI)
                    </h5>

                    <div className="flex justify-between items-center mb-2">
                        <span className="text-xs text-slate-400">Güven Skoru (Confidence)</span>
                        <span className="text-xs font-bold text-[var(--hud-text)]">{(ml_prediction.confidence * 100).toFixed(1)}%</span>
                    </div>

                    {/* Confidence Progress Bar */}
                    <div className="w-full h-1.5 bg-slate-800 rounded-full mb-3 overflow-hidden">
                        <div
                            className="h-full bg-gradient-to-r from-purple-600 to-cyan-400"
                            style={{ width: `${ml_prediction.confidence * 100}%` }}
                        />
                    </div>

                    {/* SHAP Feature Importance (Mock data for visualization) */}
                    <div className="space-y-1">
                        <p className="text-[10px] text-slate-500 mb-1">Karar Faktörleri (SHAP)</p>
                        <div className="flex justify-between items-center text-[10px]">
                            <span className="text-slate-300">Payload Entropy</span>
                            <div className="flex-1 mx-2 h-1 bg-slate-800 rounded-full">
                                <div className="h-full bg-red-400 rounded-full" style={{ width: '85%' }} />
                            </div>
                            <span className="text-red-400">+3.2</span>
                        </div>
                        <div className="flex justify-between items-center text-[10px]">
                            <span className="text-slate-300">Req/Sec Rate</span>
                            <div className="flex-1 mx-2 h-1 bg-slate-800 rounded-full">
                                <div className="h-full bg-orange-400 rounded-full" style={{ width: '60%' }} />
                            </div>
                            <span className="text-orange-400">+1.8</span>
                        </div>
                    </div>
                </div>
            )}

            <style>{`
                @keyframes dash {
                    0% { left: 0; opacity: 1; }
                    50% { opacity: 1; }
                    100% { left: 100%; opacity: 0; }
                }
            `}</style>
        </div>
    );
}
