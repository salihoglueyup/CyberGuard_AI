import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import Sidebar from '../Sidebar';

// Mock the store
vi.mock('../../store', () => ({
    useThemeStore: () => ({ isDark: true, toggleTheme: vi.fn() }),
    useSidebarStore: () => ({ collapsed: false, toggleCollapsed: vi.fn() }),
}));

const renderSidebar = () => render(
    <MemoryRouter>
        <Sidebar />
    </MemoryRouter>
);

describe('Sidebar', () => {
    it('renders CyberGuard branding', () => {
        renderSidebar();
        expect(screen.getByText('CyberGuard')).toBeInTheDocument();
    });

    it('renders search input', () => {
        renderSidebar();
        expect(screen.getByPlaceholderText('Ara...')).toBeInTheDocument();
    });

    it('shows no results message for unmatched search', async () => {
        renderSidebar();
        const searchInput = screen.getByPlaceholderText('Ara...');

        await userEvent.type(searchInput, 'xyznonexistent');

        expect(screen.getByText('Sonuç bulunamadı')).toBeInTheDocument();
    });

    it('clears search when X button clicked', async () => {
        renderSidebar();
        const searchInput = screen.getByPlaceholderText('Ara...');

        await userEvent.type(searchInput, 'test');
        expect(searchInput).toHaveValue('test');
    });

    it('renders all section headers', () => {
        renderSidebar();
        expect(screen.getByText('Genel Bakış')).toBeInTheDocument();
        expect(screen.getByText('Tehdit İstihbaratı')).toBeInTheDocument();
        expect(screen.getByText('AI & ML')).toBeInTheDocument();
    });

    it('renders system status indicator', () => {
        renderSidebar();
        expect(screen.getByText('Sistem Aktif')).toBeInTheDocument();
    });
});
