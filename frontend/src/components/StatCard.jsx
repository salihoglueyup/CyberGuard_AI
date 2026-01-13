export default function StatCard({ title, value, icon: Icon, trend, color = 'blue' }) {
    const colors = {
        blue: 'from-blue-500 to-blue-600',
        green: 'from-emerald-500 to-emerald-600',
        red: 'from-red-500 to-red-600',
        yellow: 'from-amber-500 to-amber-600',
        purple: 'from-purple-500 to-purple-600'
    };

    return (
        <div className="card group cursor-pointer">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-slate-400 text-sm mb-1">{title}</p>
                    <p className="text-3xl font-bold text-white">{value}</p>
                    {trend && (
                        <p className={`text-sm mt-2 ${trend > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {trend > 0 ? '↑' : '↓'} {Math.abs(trend)}% from last hour
                        </p>
                    )}
                </div>
                <div className={`p-3 rounded-xl bg-gradient-to-br ${colors[color]} 
          opacity-80 group-hover:opacity-100 transition-opacity`}>
                    <Icon className="w-6 h-6 text-white" />
                </div>
            </div>
        </div>
    );
}
