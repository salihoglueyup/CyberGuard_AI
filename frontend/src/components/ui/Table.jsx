import { useState } from 'react';
import { ChevronUp, ChevronDown } from 'lucide-react';

export default function Table({
    columns,
    data,
    sortable = true,
    striped = false,
    hoverable = true,
    compact = false,
    onRowClick,
    emptyMessage = 'Veri bulunamadı',
}) {
    const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

    const handleSort = (key) => {
        if (!sortable) return;

        let direction = 'asc';
        if (sortConfig.key === key && sortConfig.direction === 'asc') {
            direction = 'desc';
        }
        setSortConfig({ key, direction });
    };

    const sortedData = [...data].sort((a, b) => {
        if (!sortConfig.key) return 0;

        const aVal = a[sortConfig.key];
        const bVal = b[sortConfig.key];

        if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
        if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
        return 0;
    });

    return (
        <div className="table-container">
            <table className="table">
                <thead>
                    <tr>
                        {columns.map((col) => (
                            <th
                                key={col.key}
                                onClick={() => col.sortable !== false && handleSort(col.key)}
                                className={col.sortable !== false && sortable ? 'cursor-pointer select-none' : ''}
                                style={{ width: col.width }}
                            >
                                <div className="flex items-center gap-1">
                                    {col.label}
                                    {sortable && col.sortable !== false && (
                                        <span className="text-slate-500">
                                            {sortConfig.key === col.key ? (
                                                sortConfig.direction === 'asc' ? (
                                                    <ChevronUp className="w-4 h-4" />
                                                ) : (
                                                    <ChevronDown className="w-4 h-4" />
                                                )
                                            ) : (
                                                <ChevronUp className="w-4 h-4 opacity-30" />
                                            )}
                                        </span>
                                    )}
                                </div>
                            </th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {sortedData.length === 0 ? (
                        <tr>
                            <td colSpan={columns.length} className="text-center py-8 text-slate-400">
                                {emptyMessage}
                            </td>
                        </tr>
                    ) : (
                        sortedData.map((row, rowIdx) => (
                            <tr
                                key={row.id || rowIdx}
                                onClick={() => onRowClick?.(row)}
                                className={`
                  ${onRowClick ? 'cursor-pointer' : ''}
                  ${striped && rowIdx % 2 === 1 ? 'bg-slate-800/30' : ''}
                `}
                            >
                                {columns.map((col) => (
                                    <td key={col.key} className={compact ? 'py-2' : ''}>
                                        {col.render ? col.render(row[col.key], row) : row[col.key]}
                                    </td>
                                ))}
                            </tr>
                        ))
                    )}
                </tbody>
            </table>
        </div>
    );
}
