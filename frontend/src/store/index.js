import { create } from 'zustand';

// Theme Store
export const useThemeStore = create((set) => ({
    isDark: true,
    toggleTheme: () => set((state) => {
        const newTheme = !state.isDark;
        document.documentElement.setAttribute('data-theme', newTheme ? 'dark' : 'light');
        return { isDark: newTheme };
    })
}));

// Dashboard Store
export const useDashboardStore = create((set) => ({
    stats: null,
    recentAttacks: [],
    hourlyTrend: [],
    loading: false,
    error: null,

    setStats: (stats) => set({ stats }),
    setRecentAttacks: (attacks) => set({ recentAttacks: attacks }),
    setHourlyTrend: (trend) => set({ hourlyTrend: trend }),
    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error })
}));

// Chat Store
export const useChatStore = create((set) => ({
    messages: [],
    isLoading: false,
    selectedModel: null,

    addMessage: (message) => set((state) => ({
        messages: [...state.messages, message]
    })),
    setLoading: (loading) => set({ isLoading: loading }),
    setSelectedModel: (model) => set({ selectedModel: model }),
    clearMessages: () => set({ messages: [] })
}));

// Models Store
export const useModelsStore = create((set) => ({
    models: [],
    selectedModel: null,
    loading: false,

    setModels: (models) => set({ models }),
    setSelectedModel: (model) => set({ selectedModel: model }),
    setLoading: (loading) => set({ loading })
}));
