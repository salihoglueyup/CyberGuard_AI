export default function Skeleton({
    width,
    height = '1rem',
    circle = false,
    className = '',
    count = 1,
}) {
    const items = Array.from({ length: count }, (_, i) => i);

    return (
        <>
            {items.map((i) => (
                <div
                    key={i}
                    className={`skeleton ${className}`}
                    style={{
                        width: circle ? height : width,
                        height,
                        borderRadius: circle ? '50%' : undefined,
                    }}
                />
            ))}
        </>
    );
}

Skeleton.Text = function SkeletonText({ lines = 3, className = '' }) {
    return (
        <div className={`space-y-2 ${className}`}>
            {Array.from({ length: lines }).map((_, i) => (
                <div
                    key={i}
                    className="skeleton h-4"
                    style={{ width: i === lines - 1 ? '60%' : '100%' }}
                />
            ))}
        </div>
    );
};

Skeleton.Card = function SkeletonCard({ className = '' }) {
    return (
        <div className={`card ${className}`}>
            <div className="flex items-center gap-4 mb-4">
                <div className="skeleton w-12 h-12 rounded-full" />
                <div className="flex-1 space-y-2">
                    <div className="skeleton h-4 w-3/4" />
                    <div className="skeleton h-3 w-1/2" />
                </div>
            </div>
            <Skeleton.Text lines={3} />
        </div>
    );
};

Skeleton.Table = function SkeletonTable({ rows = 5, cols = 4 }) {
    return (
        <div className="table-container">
            <table className="table">
                <thead>
                    <tr>
                        {Array.from({ length: cols }).map((_, i) => (
                            <th key={i}><div className="skeleton h-4 w-20" /></th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {Array.from({ length: rows }).map((_, rowIdx) => (
                        <tr key={rowIdx}>
                            {Array.from({ length: cols }).map((_, colIdx) => (
                                <td key={colIdx}><div className="skeleton h-4" /></td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};
