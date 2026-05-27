import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { usePerformanceStore } from '../../hooks/usePerformance';

describe('usePerformanceStore', () => {
    it('has default metric values', () => {
        const state = usePerformanceStore.getState();
        expect(state.metrics).toBeDefined();
        expect(state.metrics.fps).toBe(0);
        expect(state.metrics.fcp).toBeNull();
        expect(state.metrics.lcp).toBeNull();
    });

    it('setMetric updates a single metric', () => {
        const { setMetric } = usePerformanceStore.getState();
        setMetric('fps', 60);
        expect(usePerformanceStore.getState().metrics.fps).toBe(60);
    });

    it('setMetrics updates multiple metrics', () => {
        const { setMetrics } = usePerformanceStore.getState();
        setMetrics({ fcp: 1200, lcp: 2400 });
        const m = usePerformanceStore.getState().metrics;
        expect(m.fcp).toBe(1200);
        expect(m.lcp).toBe(2400);
    });
});
