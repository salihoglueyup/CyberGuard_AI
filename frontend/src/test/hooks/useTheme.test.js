import { describe, it, expect, beforeEach } from 'vitest';
import { useThemeStore, THEMES } from '../../hooks/useTheme';

// Reset store between tests
beforeEach(() => {
    useThemeStore.setState({ theme: 'gotham' });
    document.documentElement.removeAttribute('data-theme');
});

describe('useThemeStore', () => {
    it('has gotham as default theme', () => {
        const state = useThemeStore.getState();
        expect(state.theme).toBe('gotham');
    });

    it('changes theme via setTheme', () => {
        const { setTheme } = useThemeStore.getState();
        setTheme('matrix');
        expect(useThemeStore.getState().theme).toBe('matrix');
    });

    it('THEMES object has 4 themes', () => {
        const keys = Object.keys(THEMES);
        expect(keys).toHaveLength(4);
        expect(keys).toEqual(['gotham', 'midnight', 'matrix', 'redteam']);
    });

    it('each theme has name and label', () => {
        Object.values(THEMES).forEach((t) => {
            expect(t).toHaveProperty('name');
            expect(t).toHaveProperty('label');
        });
    });
});
