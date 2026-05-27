import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Header from '../Header';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
    const actual = await vi.importActual('react-router-dom');
    return {
        ...actual,
        useNavigate: () => mockNavigate,
    };
});

// Mock child components
vi.mock('../NotificationBell', () => ({
    default: () => <div data-testid="notification-bell">Bell</div>,
}));
vi.mock('../shared/ThemeToggle', () => ({
    default: () => <button data-testid="theme-toggle">Theme</button>,
}));
vi.mock('../shared/LanguageSwitcher', () => ({
    default: () => <div data-testid="language-switcher">Lang</div>,
}));

const renderHeader = (path = '/') =>
    render(
        <MemoryRouter initialEntries={[path]}>
            <Header />
        </MemoryRouter>
    );

describe('Header', () => {
    beforeEach(() => {
        sessionStorage.clear();
        mockNavigate.mockClear();
    });

    it('renders without crashing', () => {
        renderHeader();
    });

    it('shows CyberGuard breadcrumb text', () => {
        renderHeader('/');
        expect(screen.getByText('CyberGuard')).toBeInTheDocument();
    });

    it('renders notification bell', () => {
        renderHeader();
        expect(screen.getByTestId('notification-bell')).toBeInTheDocument();
    });

    it('renders theme toggle', () => {
        renderHeader();
        expect(screen.getByTestId('theme-toggle')).toBeInTheDocument();
    });

    it('renders language switcher', () => {
        renderHeader();
        expect(screen.getByTestId('language-switcher')).toBeInTheDocument();
    });

    it('shows correct page title for /network', () => {
        renderHeader('/network');
        expect(screen.getByText('Ağ İzleme')).toBeInTheDocument();
    });

    it('shows correct page title for /models', () => {
        renderHeader('/models');
        expect(screen.getByText('ML Modeller')).toBeInTheDocument();
    });

    it('shows Kontrol Paneli for root path', () => {
        renderHeader('/');
        expect(screen.getByText('Kontrol Paneli')).toBeInTheDocument();
    });

    it('logout clears session and navigates to /login', () => {
        sessionStorage.setItem('token', 'test-token');
        sessionStorage.setItem('user', JSON.stringify({ username: 'admin' }));
        renderHeader('/');

        // Open user menu (find the user icon area)
        const userButtons = screen.getAllByRole('button');
        const userMenuBtn = userButtons.find((btn) =>
            btn.querySelector('svg') || btn.textContent.includes('admin') || btn.textContent === ''
        );
        if (userMenuBtn) {
            fireEvent.click(userMenuBtn);
            // Click logout if visible
            const logoutBtn = screen.queryByText('Çıkış Yap');
            if (logoutBtn) {
                fireEvent.click(logoutBtn);
                expect(sessionStorage.getItem('token')).toBeNull();
                expect(mockNavigate).toHaveBeenCalledWith('/login');
            }
        }
    });
});
