import { AlertCircle, CheckCircle, AlertTriangle, Info, X } from 'lucide-react';

const icons = {
    info: Info,
    success: CheckCircle,
    warning: AlertTriangle,
    error: AlertCircle,
};

export default function Alert({
    type = 'info',
    title,
    children,
    dismissible = false,
    onDismiss,
    className = '',
}) {
    const Icon = icons[type];

    return (
        <div className={`alert alert-${type} ${className}`}>
            <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
                {title && <p className="font-semibold mb-1">{title}</p>}
                <div className="text-sm opacity-90">{children}</div>
            </div>
            {dismissible && (
                <button
                    onClick={onDismiss}
                    className="p-1 rounded hover:bg-white/10 transition-colors"
                >
                    <X className="w-4 h-4" />
                </button>
            )}
        </div>
    );
}
