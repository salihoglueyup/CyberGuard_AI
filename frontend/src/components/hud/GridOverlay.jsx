export default function GridOverlay({ className = '' }) {
    return (
        <div className={`fixed inset-0 pointer-events-none z-0 ${className}`}
             style={{
                 background: `
                     repeating-linear-gradient(0deg, transparent, transparent 59px, rgba(0,229,255,0.025) 59px, rgba(0,229,255,0.025) 60px),
                     repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(0,229,255,0.025) 59px, rgba(0,229,255,0.025) 60px)
                 `,
             }}
        />
    );
}
