export default function CornerBrackets({ className = '', color = 'var(--hud-cyan)', size = 16, children }) {
    const style = {
        '--cb-color': color,
        '--cb-size': `${size}px`,
    };
    return (
        <div className={`relative ${className}`} style={style}>
            {/* Top-left */}
            <span className="absolute top-0 left-0 pointer-events-none" style={{
                width: 'var(--cb-size)', height: 'var(--cb-size)',
                borderTop: '2px solid var(--cb-color)',
                borderLeft: '2px solid var(--cb-color)',
                opacity: 0.4,
            }} />
            {/* Top-right */}
            <span className="absolute top-0 right-0 pointer-events-none" style={{
                width: 'var(--cb-size)', height: 'var(--cb-size)',
                borderTop: '2px solid var(--cb-color)',
                borderRight: '2px solid var(--cb-color)',
                opacity: 0.4,
            }} />
            {/* Bottom-left */}
            <span className="absolute bottom-0 left-0 pointer-events-none" style={{
                width: 'var(--cb-size)', height: 'var(--cb-size)',
                borderBottom: '2px solid var(--cb-color)',
                borderLeft: '2px solid var(--cb-color)',
                opacity: 0.4,
            }} />
            {/* Bottom-right */}
            <span className="absolute bottom-0 right-0 pointer-events-none" style={{
                width: 'var(--cb-size)', height: 'var(--cb-size)',
                borderBottom: '2px solid var(--cb-color)',
                borderRight: '2px solid var(--cb-color)',
                opacity: 0.4,
            }} />
            {children}
        </div>
    );
}
