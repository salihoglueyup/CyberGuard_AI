import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export const THEMES = {
    gotham: {
        name: 'Gotham',
        label: 'GOTHAM DARK',
        icon: '🌃',
        description: 'Varsayılan karanlık tema',
    },
    midnight: {
        name: 'Midnight',
        label: 'MIDNIGHT BLUE',
        icon: '🌌',
        description: 'Derin gece mavisi',
    },
    matrix: {
        name: 'Matrix',
        label: 'MATRIX GREEN',
        icon: '💚',
        description: 'Yeşil terminal teması',
    },
    redteam: {
        name: 'Red Team',
        label: 'RED TEAM',
        icon: '🔴',
        description: 'Kırmızı saldırı teması',
    },
};

export const useThemeStore = create(
    persist(
        (set) => ({
            theme: 'gotham',
            setTheme: (theme) => {
                document.documentElement.setAttribute('data-theme', theme);
                set({ theme });
            },
        }),
        {
            name: 'cyberguard-theme',
            onRehydrate: () => (state) => {
                if (state?.theme) {
                    document.documentElement.setAttribute('data-theme', state.theme);
                }
            },
        }
    )
);
