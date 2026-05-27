import { useEffect, useMemo, useRef } from 'react';

export default function DataMatrix({ className = '', rows = 12, cols = 32, speed = 60 }) {
    const canvasRef = useRef(null);
    const chars = useMemo(() => '01アイウエオカキクケコ█▓░▒', []);

    useEffect(() => {
        const cvs = canvasRef.current;
        if (!cvs) return;
        const ctx = cvs.getContext('2d');
        const fontSize = 12;
        cvs.width = cols * fontSize * 0.65;
        cvs.height = rows * fontSize;

        const drops = Array.from({ length: cols }, () => Math.random() * rows);

        const draw = () => {
            ctx.fillStyle = 'rgba(6, 10, 20, 0.15)';
            ctx.fillRect(0, 0, cvs.width, cvs.height);

            ctx.fillStyle = 'rgba(0, 229, 255, 0.35)';
            ctx.font = `${fontSize}px "JetBrains Mono", monospace`;

            for (let i = 0; i < drops.length; i++) {
                const ch = chars[Math.floor(Math.random() * chars.length)];
                const x = i * fontSize * 0.65;
                const y = drops[i] * fontSize;
                ctx.fillText(ch, x, y);

                if (y > cvs.height && Math.random() > 0.975) {
                    drops[i] = 0;
                }
                drops[i] += 0.5;
            }
        };

        const id = setInterval(draw, speed);
        return () => clearInterval(id);
    }, [rows, cols, speed, chars]);

    return (
        <canvas
            ref={canvasRef}
            className={`opacity-60 ${className}`}
            style={{ imageRendering: 'pixelated' }}
        />
    );
}
