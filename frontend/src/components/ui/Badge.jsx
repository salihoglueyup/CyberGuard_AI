const variants = {
    success: 'badge-success',
    warning: 'badge-warning',
    danger: 'badge-danger',
    info: 'badge-info',
    primary: 'badge-primary',
    default: 'bg-slate-700 text-slate-300 border border-slate-600',
};

const sizes = {
    sm: 'text-[10px] px-2 py-0.5',
    md: 'text-xs px-3 py-1',
    lg: 'text-sm px-4 py-1.5',
};

export default function Badge({
    children,
    variant = 'default',
    size = 'md',
    dot = false,
    icon: Icon,
    className = '',
}) {
    return (
        <span className={`badge ${variants[variant]} ${sizes[size]} ${className}`}>
            {dot && (
                <span className={`w-2 h-2 rounded-full ${variant === 'success' ? 'bg-green-400' :
                        variant === 'warning' ? 'bg-yellow-400' :
                            variant === 'danger' ? 'bg-red-400' :
                                variant === 'info' ? 'bg-cyan-400' :
                                    variant === 'primary' ? 'bg-blue-400' :
                                        'bg-slate-400'
                    }`} />
            )}
            {Icon && <Icon className="w-3 h-3" />}
            {children}
        </span>
    );
}
