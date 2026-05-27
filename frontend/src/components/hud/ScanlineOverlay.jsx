export default function ScanlineOverlay() {
    return (
        <div className="fixed inset-0 pointer-events-none z-[100]">
            {/* Horizontal scan lines */}
            <div className="absolute inset-0 bg-[repeating-linear-gradient(0deg,transparent,transparent_2px,rgba(0,229,255,0.012)_2px,rgba(0,229,255,0.012)_4px)]" />
            {/* Moving scan beam */}
            <div
                className="absolute left-0 right-0 h-[1px]"
                style={{
                    background: 'linear-gradient(90deg, transparent 0%, rgba(0,229,255,0.15) 20%, rgba(0,229,255,0.3) 50%, rgba(0,229,255,0.15) 80%, transparent 100%)',
                    animation: 'scanline 6s linear infinite',
                }}
            />
        </div>
    );
}
