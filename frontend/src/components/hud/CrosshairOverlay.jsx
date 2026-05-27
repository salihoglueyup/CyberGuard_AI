export default function CrosshairOverlay({ className = '' }) {
    return (
        <div className={`fixed inset-0 pointer-events-none z-[80] flex items-center justify-center ${className}`}>
            {/* Horizontal line */}
            <div className="absolute left-0 right-0 h-px"
                 style={{ background: 'linear-gradient(90deg, transparent 30%, rgba(0,229,255,0.08) 50%, transparent 70%)' }} />
            {/* Vertical line */}
            <div className="absolute top-0 bottom-0 w-px"
                 style={{ background: 'linear-gradient(180deg, transparent 30%, rgba(0,229,255,0.08) 50%, transparent 70%)' }} />
            {/* Center dot */}
            <div className="w-1 h-1 rounded-full bg-[rgba(0,229,255,0.2)]" />
        </div>
    );
}
