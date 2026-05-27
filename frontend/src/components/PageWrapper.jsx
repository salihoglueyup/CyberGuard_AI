import { useState, useEffect } from 'react';
import { Skeleton } from './ui';

/**
 * PageWrapper - Tüm sayfalar için ortak loading/error/empty state wrapper.
 * 
 * @param {Function} loadData - Async veri yükleme fonksiyonu
 * @param {string} title - Sayfa başlığı (opsiyonel)
 * @param {React.ReactNode} children - İçerik (render prop veya children)
 */
export default function PageWrapper({ loadData, title, children, deps = [] }) {
    const [loading, setLoading] = useState(!!loadData);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (!loadData) return;
        let cancelled = false;

        const load = async () => {
            try {
                setLoading(true);
                setError(null);
                await loadData();
            } catch (err) {
                if (!cancelled) {
                    setError(err?.response?.data?.message || err.message || 'Bir hata oluştu');
                    console.error(`[${title || 'Page'}] Yükleme hatası:`, err);
                }
            } finally {
                if (!cancelled) setLoading(false);
            }
        };

        load();
        return () => { cancelled = true; };
    }, deps);

    if (loading) {
        return (
            <div className="space-y-6">
                {title && <Skeleton className="h-8 w-64" />}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {[...Array(4)].map((_, i) => (
                        <Skeleton key={i} className="h-32 rounded-xl" />
                    ))}
                </div>
                <Skeleton className="h-96 rounded-xl" />
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex flex-col items-center justify-center h-64 bg-slate-800/50 rounded-xl border border-red-500/20">
                <div className="text-red-400 text-5xl mb-4">⚠️</div>
                <h3 className="text-[var(--hud-text)] text-lg font-bold mb-2">Yükleme Hatası</h3>
                <p className="text-slate-400 text-sm mb-4 max-w-md text-center">{error}</p>
                <button
                    onClick={() => { setError(null); setLoading(true); loadData?.().catch(setError).finally(() => setLoading(false)); }}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-[var(--hud-text)] rounded-lg text-sm transition-colors"
                >
                    Tekrar Dene
                </button>
            </div>
        );
    }

    return children;
}
