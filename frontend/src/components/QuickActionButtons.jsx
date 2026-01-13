import {
    Sparkles, AlertTriangle, Shield, TrendingUp, Search,
    FileText, Target, Activity, Zap, Brain
} from 'lucide-react';

/**
 * QuickActionButtons - AI Hızlı Aksiyon Butonları
 * 
 * Props:
 * - onAction: Aksiyon tıklandığında çağrılır (actionType, message)
 * - disabled: Butonlar devre dışı mı?
 * - variant: "horizontal" | "grid" - düzen tipi
 * - size: "sm" | "md" | "lg" - buton boyutu
 */
export default function QuickActionButtons({
    onAction,
    disabled = false,
    variant = "horizontal",
    size = "md"
}) {
    const actions = [
        {
            id: 'summary',
            label: 'Tehdit Özeti',
            icon: Sparkles,
            color: 'blue',
            message: '📊 Son 24 saatteki tehdit durumunu özetle. Toplam saldırı sayısı, engellenen saldırılar, en çok görülen saldırı tipleri ve ciddiyet dağılımını analiz et.',
            description: 'Son 24 saat özeti'
        },
        {
            id: 'critical',
            label: 'Kritik Analiz',
            icon: AlertTriangle,
            color: 'red',
            message: '🔴 Kritik ve yüksek seviyedeki saldırıları detaylı analiz et. Her bir saldırı tipi için risk değerlendirmesi yap ve acil müdahale gerektiren durumları listele.',
            description: 'Kritik tehditleri analiz et'
        },
        {
            id: 'defense',
            label: 'Savunma Önerileri',
            icon: Shield,
            color: 'green',
            message: '🛡️ Mevcut tehdit durumuna göre savunma önerileri sun. Firewall kuralları, IDS/IPS konfigürasyonları ve güvenlik politikaları için somut adımlar öner.',
            description: 'Güvenlik tavsiyeleri al'
        },
        {
            id: 'trends',
            label: 'Trend Analizi',
            icon: TrendingUp,
            color: 'orange',
            message: '📈 Saldırı trendlerini analiz et. Artış gösteren saldırı tipleri, en aktif saat dilimleri ve gelecek tahminleri yap. Pattern ve anomalileri tespit et.',
            description: 'Saldırı trendlerini incele'
        },
        {
            id: 'investigate',
            label: 'IP Araştır',
            icon: Search,
            color: 'purple',
            message: '🔍 En çok saldırı yapan IP adreslerini araştır. Her IP için ülke bilgisi, saldırı geçmişi, risk skoru ve bloklanma önerisi sun.',
            description: 'Şüpheli IP\'leri incele'
        },
        {
            id: 'report',
            label: 'Rapor Oluştur',
            icon: FileText,
            color: 'cyan',
            message: '📝 Son 24 saatlik güvenlik raporunu oluştur. Yönetici özeti, detaylı istatistikler, başarılı engelleme oranı ve iyileştirme önerileri içersin.',
            description: 'Günlük rapor hazırla'
        },
        {
            id: 'predict',
            label: 'Tahmin Yap',
            icon: Brain,
            color: 'pink',
            message: '🔮 Mevcut verilere dayanarak gelecek 24 saat için saldırı tahmini yap. Olası hedefler, beklenen saldırı tipleri ve hazırlık önerileri sun.',
            description: 'Gelecek tehditleri öngör'
        },
        {
            id: 'anomaly',
            label: 'Anomali Tespiti',
            icon: Activity,
            color: 'yellow',
            message: '⚡ Anormal trafik ve davranış kalıplarını tespit et. Normal dışı aktiviteler, potansiyel sızma girişimleri ve şüpheli hareketleri listele.',
            description: 'Anormal aktiviteleri bul'
        }
    ];

    const colorClasses = {
        blue: 'bg-blue-600/20 text-blue-400 hover:bg-blue-600/30 border-blue-500/30',
        red: 'bg-red-600/20 text-red-400 hover:bg-red-600/30 border-red-500/30',
        green: 'bg-green-600/20 text-green-400 hover:bg-green-600/30 border-green-500/30',
        orange: 'bg-orange-600/20 text-orange-400 hover:bg-orange-600/30 border-orange-500/30',
        purple: 'bg-purple-600/20 text-purple-400 hover:bg-purple-600/30 border-purple-500/30',
        cyan: 'bg-cyan-600/20 text-cyan-400 hover:bg-cyan-600/30 border-cyan-500/30',
        pink: 'bg-pink-600/20 text-pink-400 hover:bg-pink-600/30 border-pink-500/30',
        yellow: 'bg-yellow-600/20 text-yellow-400 hover:bg-yellow-600/30 border-yellow-500/30'
    };

    const sizeClasses = {
        sm: 'px-2.5 py-1.5 text-xs gap-1',
        md: 'px-3 py-2 text-sm gap-1.5',
        lg: 'px-4 py-2.5 text-sm gap-2'
    };

    const iconSizes = {
        sm: 'w-3.5 h-3.5',
        md: 'w-4 h-4',
        lg: 'w-5 h-5'
    };

    const handleClick = (action) => {
        if (disabled) return;
        onAction?.(action.id, action.message);
    };

    if (variant === 'grid') {
        return (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {actions.map(action => {
                    const Icon = action.icon;
                    return (
                        <button
                            key={action.id}
                            onClick={() => handleClick(action)}
                            disabled={disabled}
                            className={`
                                flex flex-col items-center justify-center p-4 rounded-xl border
                                ${colorClasses[action.color]}
                                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                                transition-all duration-200 hover:scale-105
                            `}
                        >
                            <Icon className="w-6 h-6 mb-2" />
                            <span className="font-medium">{action.label}</span>
                            <span className="text-xs opacity-70 mt-1 text-center">{action.description}</span>
                        </button>
                    );
                })}
            </div>
        );
    }

    // Horizontal variant (default)
    return (
        <div className="flex flex-wrap gap-2">
            {actions.slice(0, 5).map(action => {
                const Icon = action.icon;
                return (
                    <button
                        key={action.id}
                        onClick={() => handleClick(action)}
                        disabled={disabled}
                        title={action.description}
                        className={`
                            flex items-center rounded-full font-medium border
                            ${colorClasses[action.color]}
                            ${sizeClasses[size]}
                            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
                            transition-all duration-200
                        `}
                    >
                        <Icon className={iconSizes[size]} />
                        <span>{action.label}</span>
                    </button>
                );
            })}
        </div>
    );
}

// Export individual action configs for external use
export const AI_ACTIONS = {
    summary: {
        id: 'summary',
        message: '📊 Son 24 saatteki tehdit durumunu özetle. Toplam saldırı sayısı, engellenen saldırılar, en çok görülen saldırı tipleri ve ciddiyet dağılımını analiz et.'
    },
    critical: {
        id: 'critical',
        message: '🔴 Kritik ve yüksek seviyedeki saldırıları detaylı analiz et. Her bir saldırı tipi için risk değerlendirmesi yap ve acil müdahale gerektiren durumları listele.'
    },
    defense: {
        id: 'defense',
        message: '🛡️ Mevcut tehdit durumuna göre savunma önerileri sun. Firewall kuralları, IDS/IPS konfigürasyonları ve güvenlik politikaları için somut adımlar öner.'
    },
    trends: {
        id: 'trends',
        message: '📈 Saldırı trendlerini analiz et. Artış gösteren saldırı tipleri, en aktif saat dilimleri ve gelecek tahminleri yap. Pattern ve anomalileri tespit et.'
    },
    investigate: {
        id: 'investigate',
        message: '🔍 En çok saldırı yapan IP adreslerini araştır. Her IP için ülke bilgisi, saldırı geçmişi, risk skoru ve bloklanma önerisi sun.'
    }
};
