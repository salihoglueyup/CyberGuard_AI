const variants = {
    default: 'card',
    glass: 'card card-glass',
    gradient: 'card card-gradient text-white',
    glow: 'card card-glow',
    flat: 'bg-slate-800/50 rounded-xl p-6 border border-slate-700/50',
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
        <div className={`pb-4 mb-4 border-b border-slate-700/50 ${className}`}>
            {children}
        </div>
    );
};

Card.Title = function CardTitle({ children, icon: Icon, className = '' }) {
    return (
        <h3 className={`text-lg font-semibold text-white flex items-center gap-2 ${className}`}>
            {Icon && <Icon className="w-5 h-5 text-blue-400" />}
            {children}
        </h3>
    );
};

Card.Body = function CardBody({ children, className = '' }) {
    return <div className={className}>{children}</div>;
};

Card.Footer = function CardFooter({ children, className = '' }) {
    return (
        <div className={`pt-4 mt-4 border-t border-slate-700/50 ${className}`}>
            {children}
        </div>
    );
};
