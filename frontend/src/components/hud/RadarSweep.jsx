import { useEffect, useRef } from 'react';

export default function RadarSweep({ size = 120, className = '' }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        canvas.width = size * dpr;
        canvas.height = size * dpr;
        ctx.scale(dpr, dpr);

        const cx = size / 2;
        const cy = size / 2;
        const r = size / 2 - 4;
        let angle = 0;

        // Random blips
        const blips = Array.from({ length: 6 }, () => ({
            a: Math.random() * Math.PI * 2,
            d: 0.2 + Math.random() * 0.7,
            fade: 0,
        }));

        const draw = () => {
            ctx.clearRect(0, 0, size, size);

            // Grid circles
            ctx.strokeStyle = 'rgba(0,229,255,0.08)';
            ctx.lineWidth = 0.5;
            for (let i = 1; i <= 3; i++) {
                ctx.beginPath();
                ctx.arc(cx, cy, (r * i) / 3, 0, Math.PI * 2);
                ctx.stroke();
            }

            // Cross lines
            ctx.beginPath();
            ctx.moveTo(cx - r, cy);
            ctx.lineTo(cx + r, cy);
            ctx.moveTo(cx, cy - r);
            ctx.lineTo(cx, cy + r);
            ctx.stroke();

            // Sweep gradient
            const sweepGrad = ctx.createConicalGradient
                ? null // Not yet widely supported
                : null;

            // Sweep line
            const sx = cx + Math.cos(angle) * r;
            const sy = cy + Math.sin(angle) * r;
            ctx.strokeStyle = 'rgba(0,229,255,0.6)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(sx, sy);
            ctx.stroke();

            // Sweep trail (arc fill)
            ctx.fillStyle = 'rgba(0,229,255,0.04)';
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.arc(cx, cy, r, angle - 0.6, angle);
            ctx.closePath();
            ctx.fill();

            // Blips
            blips.forEach(blip => {
                const diff = ((angle - blip.a) % (Math.PI * 2) + Math.PI * 2) % (Math.PI * 2);
                if (diff < 0.3) blip.fade = 1;
                else blip.fade = Math.max(0, blip.fade - 0.008);

                if (blip.fade > 0) {
                    const bx = cx + Math.cos(blip.a) * r * blip.d;
                    const by = cy + Math.sin(blip.a) * r * blip.d;
                    ctx.fillStyle = `rgba(0,229,255,${blip.fade * 0.8})`;
                    ctx.beginPath();
                    ctx.arc(bx, by, 2, 0, Math.PI * 2);
                    ctx.fill();

                    // Glow
                    ctx.fillStyle = `rgba(0,229,255,${blip.fade * 0.2})`;
                    ctx.beginPath();
                    ctx.arc(bx, by, 5, 0, Math.PI * 2);
                    ctx.fill();
                }
            });

            // Outer ring
            ctx.strokeStyle = 'rgba(0,229,255,0.15)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.stroke();

            // Center dot
            ctx.fillStyle = 'rgba(0,229,255,0.5)';
            ctx.beginPath();
            ctx.arc(cx, cy, 2, 0, Math.PI * 2);
            ctx.fill();

            angle += 0.02;
            if (angle > Math.PI * 2) angle -= Math.PI * 2;
        };

        const id = setInterval(draw, 1000 / 30);
        draw();
        return () => clearInterval(id);
    }, [size]);

    return (
        <div className={`font-mono ${className}`}>
            <div className="text-[9px] text-[var(--hud-text-dim)] tracking-[0.15em] mb-1">RADAR</div>
            <canvas
                ref={canvasRef}
                style={{ width: size, height: size }}
                className="mx-auto"
            />
        </div>
    );
}
