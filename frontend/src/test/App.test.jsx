import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import App from '../App';

// Mock all lazy-loaded pages
vi.mock('../pages/auth/Login', () => ({
    default: () => <div data-testid="login-page">Login</div>,
}));
vi.mock('../pages/auth/Register', () => ({
    default: () => <div data-testid="register-page">Register</div>,
}));
vi.mock('../pages/core/Dashboard', () => ({
    default: () => <div data-testid="dashboard-page">Dashboard</div>,
}));
vi.mock('../components/Layout', () => ({
    default: ({ children }) => <div data-testid="layout">{children}</div>,
}));
vi.mock('../components/ProtectedRoute', () => ({
    default: ({ children }) => <div>{children}</div>,
}));
vi.mock('../components/ErrorBoundary', () => ({
    default: ({ children }) => <div>{children}</div>,
}));
vi.mock('../components/shared/PerformanceOverlay', () => ({
    default: () => null,
}));

describe('App', () => {
    it('renders without crashing', () => {
        render(<App />);
    });

    it('renders login page at /login', () => {
        window.history.pushState({}, '', '/login');
        render(<App />);
        expect(screen.getByTestId('login-page')).toBeInTheDocument();
    });

    it('renders register page at /register', () => {
        window.history.pushState({}, '', '/register');
        render(<App />);
        expect(screen.getByTestId('register-page')).toBeInTheDocument();
    });
});
