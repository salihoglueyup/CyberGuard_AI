import { useEffect, useState, useRef } from 'react';

export default function EdgeFlash({ active = false }) {
    const [show, setShow] = useState(false);
    const timeoutRef = useRef(null);

    useEffect(() => {
        if (active) {
            setShow(true);
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
            timeoutRef.current = setTimeout(() => setShow(false), 1500);
        }
        return () => { if (timeoutRef.current) clearTimeout(timeoutRef.current); };
    }, [active]);

    if (!show) return null;

    return (
        <div className="fixed inset-0 pointer-events-none z-[90]">
            {/* Top edge */}
            <div className="absolute top-0 left-0 right-0 h-1" style={{
                background: 'linear-gradient(180deg, rgba(255,0,60,0.6), transparent)',
                animation: 'edge-flash 0.5s ease-out 3',
            }} />
            {/* Bottom edge */}
            <div className="absolute bottom-0 left-0 right-0 h-1" style={{
                background: 'linear-gradient(0deg, rgba(255,0,60,0.6), transparent)',
                animation: 'edge-flash 0.5s ease-out 3',
            }} />
            {/* Left edge */}
            <div className="absolute top-0 bottom-0 left-0 w-1" style={{
                background: 'linear-gradient(90deg, rgba(255,0,60,0.6), transparent)',
                animation: 'edge-flash 0.5s ease-out 3',
            }} />
            {/* Right edge */}
            <div className="absolute top-0 bottom-0 right-0 w-1" style={{
                background: 'linear-gradient(270deg, rgba(255,0,60,0.6), transparent)',
                animation: 'edge-flash 0.5s ease-out 3',
            }} />
        </div>
    );
}
