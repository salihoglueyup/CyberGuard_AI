import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ErrorBoundary from '../ErrorBoundary';

// Component that throws on demand
const ThrowOnRender = ({ shouldThrow }) => {
    if (shouldThrow) throw new Error('Test render error');
    return <div data-testid="child">Child Content</div>;
};

// Suppress console.error for expected error boundary logs
const consoleError = console.error;
beforeAll(() => {
    console.error = vi.fn();
});
afterAll(() => {
    console.error = consoleError;
});

describe('ErrorBoundary', () => {
    it('renders children when no error', () => {
        render(
            <ErrorBoundary>
                <div data-testid="content">OK</div>
            </ErrorBoundary>
        );
        expect(screen.getByTestId('content')).toBeInTheDocument();
    });

    it('renders error UI when child throws', () => {
        render(
            <ErrorBoundary>
                <ThrowOnRender shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Bir Hata Oluştu')).toBeInTheDocument();
        expect(screen.getByText('Test render error')).toBeInTheDocument();
    });

    it('shows Tekrar Dene button on error', () => {
        render(
            <ErrorBoundary>
                <ThrowOnRender shouldThrow={true} />
            </ErrorBoundary>
        );
        expect(screen.getByText('Tekrar Dene')).toBeInTheDocument();
    });

    it('Tekrar Dene button is clickable and does not crash', () => {
        render(
            <ErrorBoundary>
                <ThrowOnRender shouldThrow={true} />
            </ErrorBoundary>
        );

        // Button should exist and be clickable without throwing
        const btn = screen.getByText('Tekrar Dene');
        expect(btn).toBeInTheDocument();
        // Click should not throw
        expect(() => fireEvent.click(btn)).not.toThrow();
    });

    it('does not render error UI when child does not throw', () => {
        render(
            <ErrorBoundary>
                <ThrowOnRender shouldThrow={false} />
            </ErrorBoundary>
        );
        expect(screen.queryByText('Bir Hata Oluştu')).not.toBeInTheDocument();
        expect(screen.getByTestId('child')).toBeInTheDocument();
    });
});
