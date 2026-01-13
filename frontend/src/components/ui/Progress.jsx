export function ProgressBar({
    value = 0,
    max = 100,
    size = 'md',
    variant = 'primary',
    showLabel = false,
    className = '',
}) {
    const percentage = Math.min(100, Math.max(0, (value / max) * 100));

    const sizes = {
        sm: 'h-1.5',
        md: 'h-2.5',
        lg: 'h-4',
    };

    const variants = {
        primary: 'bg-gradient-to-r from-blue-600 to-blue-400',
        success: 'bg-gradient-to-r from-green-600 to-green-400',
        warning: 'bg-gradient-to-r from-yellow-600 to-yellow-400',
        danger: 'bg-gradient-to-r from-red-600 to-red-400',
    };

    return (
        <div className={className}>
            <div className={`progress-bar ${sizes[size]}`}>
                <div
                    className={`progress-fill ${variants[variant]}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
            {showLabel && (
                <div className="text-right text-xs text-slate-400 mt-1">
                    {Math.round(percentage)}%
                </div>
            )}
        </div>
    );
}

export function ProgressCircle({
    value = 0,
    max = 100,
    size = 60,
    strokeWidth = 6,
    variant = 'primary',
    showLabel = true,
    className = '',
}) {
    const percentage = Math.min(100, Math.max(0, (value / max) * 100));
    const radius = (size - strokeWidth) / 2;
    const circumference = radius * 2 * Math.PI;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    const variants = {
        primary: '#3b82f6',
        success: '#22c55e',
        warning: '#f59e0b',
        danger: '#ef4444',
    };

    return (
        <div className={`relative inline-flex items-center justify-center ${className}`}>
            <svg width={size} height={size} className="transform -rotate-90">
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="rgba(148, 163, 184, 0.2)"
                    strokeWidth={strokeWidth}
                />
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={variants[variant]}
                    strokeWidth={strokeWidth}
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={strokeDashoffset}
                    style={{ transition: 'stroke-dashoffset 0.35s ease' }}
                />
            </svg>
            {showLabel && (
                <span className="absolute text-sm font-semibold text-white">
                    {Math.round(percentage)}%
                </span>
            )}
        </div>
    );
}

export default { ProgressBar, ProgressCircle };
