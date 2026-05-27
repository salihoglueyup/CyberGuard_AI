import { useState, useEffect } from 'react';
import { MemoryStick } from 'lucide-react';

const MEMORY_BLOCKS = 64;

export default function MemoryGrid({ totalGB = 32, usedGB = 0, className = '' }) {
  const [blocks, setBlocks] = useState(() => generateBlocks(usedGB || 18, totalGB));

  useEffect(() => {
    if (usedGB > 0) {
      setBlocks(generateBlocks(usedGB, totalGB));
      return;
    }
    const iv = setInterval(() => {
      const simUsed = 12 + Math.random() * 14;
      setBlocks(generateBlocks(simUsed, totalGB));
    }, 3000);
    return () => clearInterval(iv);
  }, [usedGB, totalGB]);

  const used = blocks.filter(b => b > 0).length;
  const usedPct = (used / MEMORY_BLOCKS) * 100;
  const usedGBVal = usedGB || (usedPct / 100) * totalGB;

  const getBlockColor = (v) => {
    if (v === 0) return 'rgba(0,229,255,0.03)';
    if (v > 0.8) return 'var(--hud-red)';
    if (v > 0.5) return 'var(--hud-amber)';
    return 'var(--hud-cyan)';
  };

  return (
    <div className={`${className}`} style={{
      background: 'var(--hud-surface)',
      border: '1px solid var(--hud-border)',
      borderRadius: 8,
    }}>
      <div className="flex items-center justify-between px-3 py-2 border-b" style={{ borderColor: 'var(--hud-border)' }}>
        <div className="flex items-center gap-2">
          <MemoryStick size={12} style={{ color: usedPct > 85 ? 'var(--hud-red)' : 'var(--hud-cyan)' }} />
          <span className="font-mono text-[9px] tracking-widest" style={{ color: 'var(--hud-text-muted)' }}>MEMORY</span>
        </div>
        <span className="font-mono text-[10px]" style={{ color: 'var(--hud-text)' }}>
          {usedGBVal.toFixed(1)} / {totalGB} GB
        </span>
      </div>
      <div className="p-3">
        <div className="grid gap-[2px]" style={{ gridTemplateColumns: 'repeat(16, 1fr)' }}>
          {blocks.map((v, i) => (
            <div key={i} className="aspect-square rounded-[2px] transition-colors duration-500"
              style={{
                background: getBlockColor(v),
                boxShadow: v > 0.5 ? `0 0 4px ${getBlockColor(v)}40` : 'none',
              }} />
          ))}
        </div>
        <div className="flex justify-between mt-2">
          <span className="font-mono text-[8px]" style={{ color: 'var(--hud-text-dim)' }}>
            {used}/{MEMORY_BLOCKS} blocks
          </span>
          <span className="font-mono text-[9px] font-bold"
            style={{ color: usedPct > 85 ? 'var(--hud-red)' : usedPct > 60 ? 'var(--hud-amber)' : 'var(--hud-emerald)' }}>
            {usedPct.toFixed(0)}%
          </span>
        </div>
      </div>
    </div>
  );
}

function generateBlocks(usedGB, totalGB) {
  const ratio = Math.min(1, usedGB / totalGB);
  const usedCount = Math.floor(ratio * MEMORY_BLOCKS);
  const arr = Array(MEMORY_BLOCKS).fill(0);
  const indices = Array.from({ length: MEMORY_BLOCKS }, (_, i) => i);
  // Shuffle
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  for (let i = 0; i < usedCount; i++) {
    arr[indices[i]] = 0.2 + Math.random() * 0.8;
  }
  return arr;
}
