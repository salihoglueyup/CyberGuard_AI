import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import ProtectedRoute from '../../components/ProtectedRoute';

const Protected = () => <div data-testid="protected-content">Protected</div>;
const Login = () => <div data-testid="login-page">Login</div>;

const renderWithRouter = (initialPath = '/') =>
    render(
        <MemoryRouter initialEntries={[initialPath]}>
            <Routes>
                <Route path="/login" element={<Login />} />
                <Route
                    path="/"
                    element={
                        <ProtectedRoute>
                            <Protected />
                        </ProtectedRoute>
                    }
                />
            </Routes>
        </MemoryRouter>
    );

describe('ProtectedRoute', () => {
    beforeEach(() => {
        sessionStorage.clear();
    });

    afterEach(() => {
        sessionStorage.clear();
    });

    it('redirects to /login when no token', () => {
        renderWithRouter('/');
        expect(screen.getByTestId('login-page')).toBeInTheDocument();
        expect(screen.queryByTestId('protected-content')).not.toBeInTheDocument();
    });

    it('renders children when token is present', () => {
        sessionStorage.setItem('token', 'valid-token-123');
        renderWithRouter('/');
        expect(screen.getByTestId('protected-content')).toBeInTheDocument();
        expect(screen.queryByTestId('login-page')).not.toBeInTheDocument();
    });

    it('redirects after token is removed', () => {
        sessionStorage.setItem('token', 'valid-token-123');
        const { unmount } = renderWithRouter('/');
        expect(screen.getByTestId('protected-content')).toBeInTheDocument();

        unmount();
        sessionStorage.removeItem('token');

        renderWithRouter('/');
        expect(screen.getByTestId('login-page')).toBeInTheDocument();
    });
});
