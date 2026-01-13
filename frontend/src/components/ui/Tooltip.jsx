export default function Tooltip({ children, text, position = 'top' }) {
    const positions = {
        top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
        bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
        left: 'right-full top-1/2 -translate-y-1/2 mr-2',
        right: 'left-full top-1/2 -translate-y-1/2 ml-2',
    };

    const arrows = {
        top: 'top-full left-1/2 -translate-x-1/2 border-t-slate-800',
        bottom: 'bottom-full left-1/2 -translate-x-1/2 border-b-slate-800',
        left: 'left-full top-1/2 -translate-y-1/2 border-l-slate-800',
        right: 'right-full top-1/2 -translate-y-1/2 border-r-slate-800',
    };

    return (
        <div className="tooltip-wrapper group relative inline-flex">
            {children}
            <div className={`
        absolute ${positions[position]}
        px-2.5 py-1.5 text-xs text-white bg-slate-800
        rounded-lg whitespace-nowrap z-50
        opacity-0 invisible group-hover:opacity-100 group-hover:visible
        transition-all duration-150
        shadow-lg
      `}>
                {text}
                <div className={`
          absolute w-0 h-0
          border-4 border-transparent
          ${arrows[position]}
        `} />
            </div>
        </div>
    );
}
