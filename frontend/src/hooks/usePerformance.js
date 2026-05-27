import { useEffect, useCallback } from 'react';
import { create } from 'zustand';

export const usePerformanceStore = create((set) => ({
    metrics: {
        fcp: null,
        lcp: null,
        cls: null,
        fid: null,
        ttfb: null,
        domNodes: 0,
        jsHeap: null,
        fps: 0,
    },
    setMetric: (key, value) => set((s) => ({ metrics: { ...s.metrics, [key]: value } })),
    setMetrics: (updates) => set((s) => ({ metrics: { ...s.metrics, ...updates } })),
}));

export function usePerformanceMonitor() {
    const setMetric = usePerformanceStore((s) => s.setMetric);
    const setMetrics = usePerformanceStore((s) => s.setMetrics);

    useEffect(() => {
        // Web Vitals via PerformanceObserver
        const observers = [];

        // LCP
        try {
            const lcpObs = new PerformanceObserver((list) => {
                const entries = list.getEntries();
                const last = entries[entries.length - 1];
                if (last) setMetric('lcp', Math.round(last.startTime));
            });
            lcpObs.observe({ type: 'largest-contentful-paint', buffered: true });
            observers.push(lcpObs);
        } catch (_e) { /* not supported */ }

        // FCP
        try {
            const fcpObs = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.name === 'first-contentful-paint') {
                        setMetric('fcp', Math.round(entry.startTime));
                    }
                }
            });
            fcpObs.observe({ type: 'paint', buffered: true });
            observers.push(fcpObs);
        } catch (_e) { /* not supported */ }

        // CLS
        try {
            let clsValue = 0;
            const clsObs = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (!entry.hadRecentInput) {
                        clsValue += entry.value;
                        setMetric('cls', Math.round(clsValue * 1000) / 1000);
                    }
                }
            });
            clsObs.observe({ type: 'layout-shift', buffered: true });
            observers.push(clsObs);
        } catch (_e) { /* not supported */ }

        // TTFB
        try {
            const navEntries = performance.getEntriesByType('navigation');
            if (navEntries.length > 0) {
                setMetric('ttfb', Math.round(navEntries[0].responseStart));
            }
        } catch (_e) { /* not supported */ }

        // FPS counter
        let frameCount = 0;
        let lastTime = performance.now();
        let animId;
        const measureFps = () => {
            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                setMetric('fps', frameCount);
                frameCount = 0;
                lastTime = now;
            }
            animId = requestAnimationFrame(measureFps);
        };
        animId = requestAnimationFrame(measureFps);

        // DOM nodes + JS heap (periodic)
        const intervalId = setInterval(() => {
            const updates = { domNodes: document.querySelectorAll('*').length };
            if (performance.memory) {
                updates.jsHeap = Math.round(performance.memory.usedJSHeapSize / 1048576);
            }
            setMetrics(updates);
        }, 5000);

        return () => {
            observers.forEach((o) => o.disconnect());
            cancelAnimationFrame(animId);
            clearInterval(intervalId);
        };
    }, [setMetric, setMetrics]);
}

export function useRenderCount(componentName) {
    const countRef = { current: 0 };
    countRef.current++;
    if (import.meta.env.DEV) {
        console.debug(`[Perf] ${componentName} render #${countRef.current}`);
    }
}

export function usePrefetch() {
    return useCallback((importFn) => {
        if (typeof importFn === 'function') {
            importFn().catch(() => {});
        }
    }, []);
}
