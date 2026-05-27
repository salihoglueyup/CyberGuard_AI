import { useState, useEffect, useRef } from 'react';

export default function TypewriterText({ text, speed = 40, className = '', onDone }) {
    const [displayed, setDisplayed] = useState('');
    const idx = useRef(0);

    useEffect(() => {
        idx.current = 0;
        setDisplayed('');
        const interval = setInterval(() => {
            idx.current++;
            setDisplayed(text.slice(0, idx.current));
            if (idx.current >= text.length) {
                clearInterval(interval);
                onDone?.();
            }
        }, speed);
        return () => clearInterval(interval);
    }, [text, speed]);

    return (
        <span className={`font-mono ${className}`}>
            {displayed}
            <span className="inline-block w-[2px] h-[1em] bg-[var(--hud-cyan)] ml-0.5 align-middle"
                  style={{ animation: 'edge-flash 1s step-end infinite' }} />
        </span>
    );
}
