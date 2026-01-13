import { forwardRef, useState } from 'react';
import { Eye, EyeOff, Search, X } from 'lucide-react';

const Input = forwardRef(({
    type = 'text',
    label,
    error,
    icon: Icon,
    iconRight: IconRight,
    className = '',
    ...props
}, ref) => {
    const [showPassword, setShowPassword] = useState(false);

    const inputType = type === 'password' && showPassword ? 'text' : type;

    return (
        <div className={`space-y-1.5 ${className}`}>
            {label && (
                <label className="block text-sm font-medium text-slate-300">
                    {label}
                </label>
            )}
            <div className="relative">
                {Icon && (
                    <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400">
                        <Icon className="w-4 h-4" />
                    </div>
                )}
                <input
                    ref={ref}
                    type={inputType}
                    className={`
            input
            ${Icon ? 'pl-10' : ''}
            ${type === 'password' || IconRight ? 'pr-10' : ''}
            ${error ? 'border-red-500 focus:border-red-500 focus:ring-red-500/20' : ''}
          `}
                    {...props}
                />
                {type === 'password' && (
                    <button
                        type="button"
                        onClick={() => setShowPassword(!showPassword)}
                        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-300"
                    >
                        {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                )}
                {IconRight && type !== 'password' && (
                    <div className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400">
                        <IconRight className="w-4 h-4" />
                    </div>
                )}
            </div>
            {error && <p className="text-xs text-red-400">{error}</p>}
        </div>
    );
});

Input.displayName = 'Input';

export function SearchInput({
    value,
    onChange,
    onClear,
    placeholder = 'Ara...',
    className = '',
}) {
    return (
        <div className={`relative ${className}`}>
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <input
                type="text"
                value={value}
                onChange={(e) => onChange(e.target.value)}
                placeholder={placeholder}
                className="input pl-10 pr-8"
            />
            {value && (
                <button
                    onClick={() => {
                        onChange('');
                        onClear?.();
                    }}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-300"
                >
                    <X className="w-4 h-4" />
                </button>
            )}
        </div>
    );
}

export default Input;
