import { create } from 'zustand';

// Theme Store - re-export from hooks/useTheme for backward compat
export { useThemeStore } from '../hooks/useTheme';

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

// Sidebar Store
export const useSidebarStore = create((set) => ({
    collapsed: false,
    toggleCollapsed: () => set((state) => ({ collapsed: !state.collapsed }))
}));
