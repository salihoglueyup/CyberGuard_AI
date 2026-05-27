import { describe, it, expect, beforeEach } from 'vitest';
import { useRealtimeMetrics } from '../../hooks/useRealtimeMetrics';
import { act } from '@testing-library/react';

// Since useRealtimeMetrics is a zustand store, we can test it directly
describe('useRealtimeMetrics store', () => {
    it('exports a hook function', () => {
        expect(typeof useRealtimeMetrics).toBe('function');
    });
});
