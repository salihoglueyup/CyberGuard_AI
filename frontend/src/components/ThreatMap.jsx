import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { Globe } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

// Tehdit renkleri
const severityColors = {
    low: '#22c55e',
    medium: '#f59e0b',
    high: '#f97316',
    critical: '#ef4444',
};

// Harita merkezleme componenti
function MapUpdater({ threats }) {
    const map = useMap();

    useEffect(() => {
        if (threats.length > 0) {
            const lastThreat = threats[0];
            if (lastThreat.lat && lastThreat.lng) {
                map.flyTo([lastThreat.lat, lastThreat.lng], 4, { duration: 1 });
            }
        }
    }, [threats, map]);

    return null;
}

export default function ThreatMap({ threats = [], height = '400px', className = '' }) {
    const mapRef = useRef(null);

    // Varsayılan konum (Türkiye)
    const defaultCenter = [39.9334, 32.8597];
    const defaultZoom = 2;

    return (
        <div className={`relative rounded-xl overflow-hidden border border-slate-700/50 ${className}`} style={{ height }}>
            {/* Başlık */}
            <div className="absolute top-3 left-3 z-[1000] bg-slate-900/90 backdrop-blur-sm px-3 py-2 rounded-lg border border-slate-700/50">
                <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-medium text-white">Canlı Tehdit Haritası</span>
                    {threats.length > 0 && (
                        <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">
                            {threats.length} tehdit
                        </span>
                    )}
                </div>
            </div>

            {/* Legend */}
            <div className="absolute bottom-3 left-3 z-[1000] bg-slate-900/90 backdrop-blur-sm px-3 py-2 rounded-lg border border-slate-700/50">
                <div className="flex items-center gap-3 text-xs">
                    {Object.entries(severityColors).map(([severity, color]) => (
                        <div key={severity} className="flex items-center gap-1">
                            <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
                            <span className="text-slate-400 capitalize">{severity}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Harita */}
            <MapContainer
                ref={mapRef}
                center={defaultCenter}
                zoom={defaultZoom}
                style={{ height: '100%', width: '100%' }}
                className="z-0"
                scrollWheelZoom={true}
            >
                {/* Dark theme tile layer */}
                <TileLayer
                    attribution='&copy; <a href="https://carto.com/">CARTO</a>'
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />

                {/* Tehdit marker'ları */}
                {threats.map((threat, idx) => (
                    threat.lat && threat.lng && (
                        <CircleMarker
                            key={threat.id || idx}
                            center={[threat.lat, threat.lng]}
                            radius={8 + (threat.severity === 'critical' ? 4 : threat.severity === 'high' ? 2 : 0)}
                            fillColor={severityColors[threat.severity] || severityColors.medium}
                            fillOpacity={0.7}
                            stroke={true}
                            color={severityColors[threat.severity] || severityColors.medium}
                            weight={2}
                            opacity={0.9}
                        >
                            <Popup className="threat-popup">
                                <div className="p-1 min-w-[180px]">
                                    <div className="flex items-center justify-between mb-2">
                                        <span className="font-semibold text-slate-900">{threat.threat_type}</span>
                                        <span className={`px-2 py-0.5 text-xs rounded-full capitalize ${threat.severity === 'critical' ? 'bg-red-100 text-red-700' :
                                                threat.severity === 'high' ? 'bg-orange-100 text-orange-700' :
                                                    threat.severity === 'medium' ? 'bg-yellow-100 text-yellow-700' :
                                                        'bg-green-100 text-green-700'
                                            }`}>
                                            {threat.severity}
                                        </span>
                                    </div>
                                    <div className="space-y-1 text-xs text-slate-600">
                                        <p><strong>Kaynak:</strong> {threat.source_ip}</p>
                                        <p><strong>Ülke:</strong> {threat.country}</p>
                                        <p><strong>Durum:</strong> {threat.blocked ? '✅ Engellendi' : '⚠️ Aktif'}</p>
                                        <p className="text-slate-400">{new Date(threat.timestamp).toLocaleTimeString('tr-TR')}</p>
                                    </div>
                                </div>
                            </Popup>
                        </CircleMarker>
                    )
                ))}

                <MapUpdater threats={threats} />
            </MapContainer>
        </div>
    );
}
