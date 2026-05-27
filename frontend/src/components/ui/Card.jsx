const variants = {
    default: 'card',
    glass: 'card card-glass',
    gradient: 'card card-gradient text-[var(--hud-text)]',
    glow: 'card card-glow',
    flat: 'bg-[var(--hud-surface)] rounded-xl p-6 border border-[var(--hud-border)]',
};

export default function Card({
    children,
    variant = 'default',
    className = '',
    hover = true,
    onClick,
    ...props
}) {
    return (
        <div
            className={`${variants[variant]} ${!hover ? '!transform-none hover:!shadow-md' : ''} ${onClick ? 'cursor-pointer' : ''} ${className}`}
            onClick={onClick}
            {...props}
        >
            {children}
        </div>
    );
}

Card.Header = function CardHeader({ children, className = '' }) {
    return (
        <div className={`pb-4 mb-4 border-b border-[var(--hud-border)] ${className}`}>
            {children}
        </div>
    );
};

Card.Title = function CardTitle({ children, icon: Icon, className = '' }) {
    return (
        <h3 className={`text-lg font-semibold text-[var(--hud-text-bright)] flex items-center gap-2 ${className}`}>
            {Icon && <Icon className="w-5 h-5 text-[var(--hud-cyan)]" />}
            {children}
        </h3>
    );
};

Card.Body = function CardBody({ children, className = '' }) {
    return <div className={className}>{children}</div>;
};

Card.Footer = function CardFooter({ children, className = '' }) {
    return (
        <div className={`pt-4 mt-4 border-t border-[var(--hud-border)] ${className}`}>
            {children}
        </div>
    );
};
