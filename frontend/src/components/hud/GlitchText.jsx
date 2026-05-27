import { useState, useEffect, useRef } from 'react';

export default function GlitchText({ text, className = '', tag = 'span', interval = 8000, duration = 200 }) {
    const [display, setDisplay] = useState(text);
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*';
    const rafRef = useRef(null);

    useEffect(() => {
        setDisplay(text);
        const id = setInterval(() => {
            let step = 0;
            const maxSteps = text.length;
            const tick = () => {
                step++;
                setDisplay(
                    text.split('').map((ch, i) =>
                        i < step ? ch : chars[Math.floor(Math.random() * chars.length)]
                    ).join('')
                );
                if (step < maxSteps) {
                    rafRef.current = setTimeout(tick, duration / maxSteps);
                }
            };
            tick();
        }, interval);

        return () => {
            clearInterval(id);
            if (rafRef.current) clearTimeout(rafRef.current);
        };
    }, [text, interval, duration]);

    const Tag = tag;
    return <Tag className={`font-mono ${className}`}>{display}</Tag>;
}
