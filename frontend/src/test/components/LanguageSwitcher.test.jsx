import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import LanguageSwitcher from '../../components/shared/LanguageSwitcher';

// Mock react-i18next
vi.mock('react-i18next', () => ({
    useTranslation: () => ({
        t: (key) => key,
        i18n: {
            language: 'tr',
            changeLanguage: vi.fn(),
        },
    }),
}));

describe('LanguageSwitcher', () => {
    it('renders compact mode with TR label', () => {
        render(<LanguageSwitcher compact />);
        expect(screen.getByText('TR')).toBeInTheDocument();
    });

    it('renders full mode with language options', () => {
        render(<LanguageSwitcher />);
        expect(screen.getByText('Türkçe')).toBeInTheDocument();
        expect(screen.getByText('English')).toBeInTheDocument();
    });

    it('calls changeLanguage on click', () => {
        const { useTranslation } = require('react-i18next');
        render(<LanguageSwitcher />);
        fireEvent.click(screen.getByText('English'));
    });
});
