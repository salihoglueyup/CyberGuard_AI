import { describe, it, expect, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PageWrapper from '../PageWrapper';

// Mock the ui/Skeleton component
vi.mock('../ui', () => ({
    Skeleton: ({ className }) => <div data-testid="skeleton" className={className} />,
}));

describe('PageWrapper', () => {
    it('renders children when no loadData is provided', () => {
        render(
            <PageWrapper>
                <div>Test Content</div>
            </PageWrapper>
        );
        expect(screen.getByText('Test Content')).toBeInTheDocument();
    });

    it('shows loading skeleton while data is loading', () => {
        const loadData = () => new Promise(() => {}); // never resolves
        render(
            <PageWrapper loadData={loadData} title="Test">
                <div>Content</div>
            </PageWrapper>
        );
        expect(screen.getAllByTestId('skeleton').length).toBeGreaterThan(0);
        expect(screen.queryByText('Content')).not.toBeInTheDocument();
    });

    it('renders children after data loads successfully', async () => {
        const loadData = vi.fn().mockResolvedValueOnce(undefined);
        render(
            <PageWrapper loadData={loadData}>
                <div>Loaded Content</div>
            </PageWrapper>
        );
        await waitFor(() => {
            expect(screen.getByText('Loaded Content')).toBeInTheDocument();
        });
        expect(loadData).toHaveBeenCalledTimes(1);
    });

    it('shows error state when loadData fails', async () => {
        const loadData = vi.fn().mockRejectedValueOnce(new Error('API error'));
        render(
            <PageWrapper loadData={loadData}>
                <div>Content</div>
            </PageWrapper>
        );
        await waitFor(() => {
            expect(screen.getByText('Yükleme Hatası')).toBeInTheDocument();
        });
        expect(screen.getByText('API error')).toBeInTheDocument();
    });

    it('retries loading on retry button click', async () => {
        const loadData = vi.fn()
            .mockRejectedValueOnce(new Error('fail'))
            .mockResolvedValueOnce(undefined);

        render(
            <PageWrapper loadData={loadData}>
                <div>Success</div>
            </PageWrapper>
        );

        await waitFor(() => {
            expect(screen.getByText('Tekrar Dene')).toBeInTheDocument();
        });

        await userEvent.click(screen.getByText('Tekrar Dene'));

        await waitFor(() => {
            expect(screen.getByText('Success')).toBeInTheDocument();
        });
        expect(loadData).toHaveBeenCalledTimes(2);
    });
});
