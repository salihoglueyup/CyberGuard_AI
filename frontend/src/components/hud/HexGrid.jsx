import { useEffect, useRef, useState } from 'react';

export default function HexGrid({ cols = 8, rows = 4, className = '' }) {
    const [cells, setCells] = useState([]);

    useEffect(() => {
        const initial = Array.from({ length: cols * rows }, (_, i) => ({
            id: i,
            active: Math.random() > 0.7,
            threat: Math.random() > 0.9,
        }));
        setCells(initial);

        const id = setInterval(() => {
            setCells(prev =>
                prev.map(cell => ({
                    ...cell,
                    active: Math.random() > 0.6,
                    threat: Math.random() > 0.92,
                }))
            );
        }, 3000);
        return () => clearInterval(id);
    }, [cols, rows]);

    const hexSize = 14;
    const hexW = hexSize * 2;
    const hexH = hexSize * Math.sqrt(3);

    return (
        <div className={`font-mono ${className}`}>
            <div className="text-[9px] text-[var(--hud-text-dim)] tracking-[0.15em] mb-1">NETWORK GRID</div>
            <svg
                width={cols * hexW * 0.76 + hexSize}
                height={rows * hexH + hexH / 2}
                className="mx-auto"
            >
                {cells.map((cell, i) => {
                    const col = i % cols;
                    const row = Math.floor(i / cols);
                    const x = col * hexW * 0.76 + (row % 2 ? hexSize * 0.76 : 0) + hexSize;
                    const y = row * hexH * 0.88 + hexSize;

                    const points = Array.from({ length: 6 }, (_, k) => {
                        const a = (Math.PI / 3) * k - Math.PI / 6;
                        return `${x + hexSize * Math.cos(a)},${y + hexSize * Math.sin(a)}`;
                    }).join(' ');

                    const fill = cell.threat
                        ? 'rgba(255,0,60,0.25)'
                        : cell.active
                            ? 'rgba(0,229,255,0.08)'
                            : 'rgba(255,255,255,0.01)';
                    const stroke = cell.threat
                        ? 'rgba(255,0,60,0.4)'
                        : cell.active
                            ? 'rgba(0,229,255,0.2)'
                            : 'rgba(255,255,255,0.05)';

                    return (
                        <polygon
                            key={cell.id}
                            points={points}
                            fill={fill}
                            stroke={stroke}
                            strokeWidth={0.5}
                            className="transition-all duration-500"
                        />
                    );
                })}
            </svg>
        </div>
    );
}
