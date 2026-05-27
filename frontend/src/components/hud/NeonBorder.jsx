export default function NeonBorder({ color = 'var(--hud-cyan)', className = '', children }) {
    return (
        <div className={`relative rounded-lg overflow-hidden ${className}`}
             style={{
                 border: `1px solid ${color}`,
                 boxShadow: `0 0 10px ${color}33, inset 0 0 10px ${color}11`,
             }}>
            {/* Top edge glow */}
            <div className="absolute top-0 left-0 right-0 h-px"
                 style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)`, opacity: 0.6 }} />
            {children}
        </div>
    );
}
