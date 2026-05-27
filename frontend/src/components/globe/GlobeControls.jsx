import { useState } from 'react';
import {
  Crosshair, RotateCw, ZoomIn, ZoomOut, Maximize2, Camera,
  Layers, Eye, EyeOff, Globe2
} from 'lucide-react';

const LAYERS = [
  { key: 'arcs', label: 'Saldiri Yollari', defaultOn: true },
  { key: 'hexbin', label: 'Heatmap', defaultOn: true },
  { key: 'rings', label: 'Hedef Sinyalleri', defaultOn: true },
  { key: 'points', label: 'AI Noktalar', defaultOn: true },
  { key: 'markers', label: 'Shockwave', defaultOn: true },
  { key: 'labels', label: 'Ulke Etiketleri', defaultOn: true },
];

export default function GlobeControls({
  globeRef,
  onResetView,
  onToggleAutoRotate,
  autoRotate = true,
  onScreenshot,
  layers = {},
  onToggleLayer,
  className = '',
}) {
  const [showLayers, setShowLayers] = useState(false);

  const handleZoomIn = () => {
    if (!globeRef?.current) return;
    const pov = globeRef.current.pointOfView();
    globeRef.current.pointOfView({ ...pov, altitude: Math.max(0.5, pov.altitude * 0.7) }, 400);
  };

  const handleZoomOut = () => {
    if (!globeRef?.current) return;
    const pov = globeRef.current.pointOfView();
    globeRef.current.pointOfView({ ...pov, altitude: Math.min(5, pov.altitude * 1.4) }, 400);
  };

  const handleReset = () => {
    if (globeRef?.current) {
      globeRef.current.pointOfView({ lat: 25, lng: 15, altitude: 2.2 }, 1200);
    }
    onResetView?.();
  };

  const handleFocusTR = () => {
    if (globeRef?.current) {
      globeRef.current.pointOfView({ lat: 39, lng: 35, altitude: 1.5 }, 800);
    }
  };

  return (
    <div className={`absolute right-3 top-1/2 -translate-y-1/2 z-10 flex flex-col gap-1 ${className}`}>
      <ControlBtn icon={<ZoomIn size={14} />} title="Zoom In" onClick={handleZoomIn} />
      <ControlBtn icon={<ZoomOut size={14} />} title="Zoom Out" onClick={handleZoomOut} />
      <div className="h-px my-0.5" style={{ background: 'var(--hud-border)' }} />
      <ControlBtn icon={<Crosshair size={14} />} title="Turkiye'ye Odakla" onClick={handleFocusTR} />
      <ControlBtn icon={<Globe2 size={14} />} title="Gorunumu Sifirla" onClick={handleReset} />
      <ControlBtn
        icon={<RotateCw size={14} />}
        title={autoRotate ? 'Dondurmey Durdur' : 'Otomatik Dondur'}
        onClick={onToggleAutoRotate}
        active={autoRotate}
      />
      <div className="h-px my-0.5" style={{ background: 'var(--hud-border)' }} />
      <ControlBtn
        icon={<Layers size={14} />}
        title="Katmanlar"
        onClick={() => setShowLayers(!showLayers)}
        active={showLayers}
      />
      <ControlBtn icon={<Camera size={14} />} title="Screenshot" onClick={onScreenshot} />

      {/* Layers panel */}
      {showLayers && (
        <div className="absolute right-10 top-0 w-48 p-2 rounded-lg"
          style={{
            background: 'var(--hud-surface-elevated)',
            border: '1px solid var(--hud-border-strong)',
            boxShadow: 'var(--hud-shadow-lg)',
          }}>
          <div className="font-mono text-[9px] tracking-wide text-[var(--hud-text-muted)] mb-2">
            Katmanlar
          </div>
          {LAYERS.map(l => {
            const isOn = layers[l.key] !== false;
            return (
              <button
                key={l.key}
                onClick={() => onToggleLayer?.(l.key)}
                className="flex items-center gap-2 w-full py-1.5 px-1 rounded hover:bg-white/5 transition-colors"
              >
                {isOn ?
                  <Eye size={12} style={{ color: 'var(--hud-cyan)' }} /> :
                  <EyeOff size={12} style={{ color: 'var(--hud-text-dim)' }} />
                }
                <span className="font-mono text-[10px]" style={{ color: isOn ? 'var(--hud-text)' : 'var(--hud-text-dim)' }}>
                  {l.label}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function ControlBtn({ icon, title, onClick, active = false }) {
  return (
    <button
      onClick={onClick}
      title={title}
      aria-label={title}
      className="w-8 h-8 flex items-center justify-center rounded-md transition-all duration-200"
      style={{
        background: active ? 'rgba(56,189,248,0.12)' : 'var(--hud-surface-elevated)',
        border: `1px solid ${active ? 'rgba(56,189,248,0.3)' : 'var(--hud-border)'}`,
        color: active ? 'var(--hud-cyan)' : 'var(--hud-text-muted)',
        boxShadow: active ? '0 0 12px rgba(56,189,248,0.15)' : 'var(--hud-shadow)',
      }}
    >
      {icon}
    </button>
  );
}
