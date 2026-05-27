import { useState, useEffect, useRef } from 'react';

export default function NetworkGraph({ width = 200, height = 120, nodeCount = 12, className = '' }) {
    const canvasRef = useRef(null);
    const nodesRef = useRef([]);
    const edgesRef = useRef([]);

    useEffect(() => {
        // Init nodes
        const nodes = Array.from({ length: nodeCount }, (_, i) => ({
            x: 20 + Math.random() * (width - 40),
            y: 15 + Math.random() * (height - 30),
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            r: 2 + Math.random() * 2,
            threat: i < 2, // first 2 are threat nodes
        }));
        nodesRef.current = nodes;

        // Init edges
        const edges = [];
        for (let i = 0; i < nodeCount; i++) {
            const connections = 1 + Math.floor(Math.random() * 2);
            for (let c = 0; c < connections; c++) {
                const j = Math.floor(Math.random() * nodeCount);
                if (j !== i) edges.push([i, j]);
            }
        }
        edgesRef.current = edges;

        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        ctx.scale(dpr, dpr);

        const animate = () => {
            ctx.clearRect(0, 0, width, height);

            // Update positions
            nodes.forEach(n => {
                n.x += n.vx;
                n.y += n.vy;
                if (n.x < 10 || n.x > width - 10) n.vx *= -1;
                if (n.y < 10 || n.y > height - 10) n.vy *= -1;
            });

            // Draw edges
            edges.forEach(([i, j]) => {
                const a = nodes[i], b = nodes[j];
                const dx = b.x - a.x, dy = b.y - a.y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > 100) return;
                const alpha = (1 - dist / 100) * 0.2;
                const isThreatEdge = a.threat || b.threat;
                ctx.strokeStyle = isThreatEdge
                    ? `rgba(255,0,60,${alpha})`
                    : `rgba(0,229,255,${alpha})`;
                ctx.lineWidth = 0.5;
                ctx.beginPath();
                ctx.moveTo(a.x, a.y);
                ctx.lineTo(b.x, b.y);
                ctx.stroke();
            });

            // Draw nodes
            nodes.forEach(n => {
                const color = n.threat ? 'rgba(255,0,60,0.7)' : 'rgba(0,229,255,0.5)';
                const glow = n.threat ? 'rgba(255,0,60,0.15)' : 'rgba(0,229,255,0.1)';

                ctx.fillStyle = glow;
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r + 3, 0, Math.PI * 2);
                ctx.fill();

                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
                ctx.fill();
            });
        };

        const id = setInterval(animate, 1000 / 24);
        animate();
        return () => clearInterval(id);
    }, [width, height, nodeCount]);

    return (
        <div className={`font-mono ${className}`}>
            <div className="text-[9px] text-[var(--hud-text-dim)] tracking-[0.15em] mb-1">NETWORK TOPOLOGY</div>
            <canvas
                ref={canvasRef}
                style={{ width, height }}
            />
        </div>
    );
}
