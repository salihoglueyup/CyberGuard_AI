export default function PulseRing({ color = 'var(--hud-cyan)', size = 40, className = '' }) {
    return (
        <div className={`relative inline-flex items-center justify-center ${className}`}
             style={{ width: size, height: size }}>
            <div className="absolute rounded-full"
                 style={{
                     width: '100%', height: '100%',
                     border: `2px solid ${color}`,
                     animation: 'sonar-ring 2s ease-out infinite',
                 }} />
            <div className="absolute rounded-full"
                 style={{
                     width: '100%', height: '100%',
                     border: `2px solid ${color}`,
                     animation: 'sonar-ring 2s ease-out infinite 0.6s',
                 }} />
            <div className="rounded-full"
                 style={{
                     width: size * 0.3, height: size * 0.3,
                     background: color,
                     boxShadow: `0 0 12px ${color}`,
                 }} />
        </div>
    );
}
