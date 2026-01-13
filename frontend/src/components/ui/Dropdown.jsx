import { useState, useRef, useEffect } from 'react';
import { ChevronDown, Check } from 'lucide-react';

export default function Dropdown({
    options,
    value,
    onChange,
    placeholder = 'Seçin...',
    icon: Icon,
    disabled = false,
    className = '',
}) {
    const [isOpen, setIsOpen] = useState(false);
    const ref = useRef(null);

    const selectedOption = options.find((opt) => opt.value === value);

    useEffect(() => {
        const handleClickOutside = (e) => {
            if (ref.current && !ref.current.contains(e.target)) {
                setIsOpen(false);
            }
        };
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    return (
        <div ref={ref} className={`dropdown relative ${className}`}>
            <button
                type="button"
                onClick={() => !disabled && setIsOpen(!isOpen)}
                disabled={disabled}
                className={`
          w-full flex items-center justify-between gap-2
          px-4 py-2.5 text-sm
          bg-slate-800 border border-slate-700 rounded-lg
          text-left transition-colors
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-blue-500 cursor-pointer'}
          ${isOpen ? 'border-blue-500 ring-2 ring-blue-500/20' : ''}
        `}
            >
                <div className="flex items-center gap-2 flex-1 min-w-0">
                    {Icon && <Icon className="w-4 h-4 text-slate-400 flex-shrink-0" />}
                    <span className={selectedOption ? 'text-white' : 'text-slate-400'}>
                        {selectedOption?.label || placeholder}
                    </span>
                </div>
                <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>

            {isOpen && (
                <div className="dropdown-menu">
                    {options.map((option) => (
                        <div
                            key={option.value}
                            onClick={() => {
                                onChange(option.value);
                                setIsOpen(false);
                            }}
                            className={`dropdown-item ${value === option.value ? 'bg-blue-500/20 text-blue-400' : ''}`}
                        >
                            {option.icon && <option.icon className="w-4 h-4" />}
                            <span className="flex-1">{option.label}</span>
                            {value === option.value && <Check className="w-4 h-4" />}
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
