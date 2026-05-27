import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import GlobeHUD from '../globe/GlobeHUD';

describe('GlobeHUD', () => {
    const defaultStats = { total: 42, blocked: 38, critical: 3, avgConfidence: 0.87 };
    const mockAttacks = [
        {
            id: '1',
            severity: 'critical',
            threat_type: 'DDoS',
            source: { country: 'CN', lat: 35, lng: 105 },
            target: { country: 'TR', lat: 39, lng: 35 },
            ml_prediction: { is_threat: true, confidence: 0.95 },
        },
        {
            id: '2',
            severity: 'high',
            threat_type: 'SQL Injection',
            source: { country: 'RU', lat: 60, lng: 100 },
            target: { country: 'TR', lat: 39, lng: 35 },
            ml_prediction: { is_threat: true, confidence: 0.82 },
        },
        {
            id: '3',
            severity: 'medium',
            attack_type: 'Port Scan',
            source: { country: 'CN', lat: 35, lng: 105 },
            target: { country: 'TR', lat: 39, lng: 35 },
        },
    ];

    it('renders system status panel with stats', () => {
        render(<GlobeHUD stats={defaultStats} attacks={mockAttacks} />);
        expect(screen.getByText('42')).toBeInTheDocument();
        expect(screen.getByText('38')).toBeInTheDocument();
        expect(screen.getByText('3')).toBeInTheDocument();
    });

    it('shows CANLI İZLEME when isLive', () => {
        render(<GlobeHUD stats={defaultStats} attacks={[]} isLive={true} />);
        expect(screen.getByText('CANLI İZLEME')).toBeInTheDocument();
    });

    it('shows DURAKLADI when not live', () => {
        render(<GlobeHUD stats={defaultStats} attacks={[]} isLive={false} />);
        expect(screen.getByText('DURAKLADI')).toBeInTheDocument();
    });

    it('renders alert feed for critical/high attacks', () => {
        render(<GlobeHUD stats={defaultStats} attacks={mockAttacks} />);
        expect(screen.getByText('ALARM AKIŞI')).toBeInTheDocument();
        expect(screen.getAllByText('DDoS').length).toBeGreaterThan(0);
        expect(screen.getAllByText('SQL Injection').length).toBeGreaterThan(0);
    });

    it('renders top source countries', () => {
        render(<GlobeHUD stats={defaultStats} attacks={mockAttacks} />);
        expect(screen.getByText('EN YOĞUN KAYNAKLAR')).toBeInTheDocument();
        expect(screen.getByText('CN')).toBeInTheDocument();
    });

    it('renders attack types distribution', () => {
        render(<GlobeHUD stats={defaultStats} attacks={mockAttacks} />);
        expect(screen.getByText('SALDIRI TÜRLERİ')).toBeInTheDocument();
    });

    it('renders country drill-down panel when selectedCountry provided', () => {
        const onClose = vi.fn();
        render(
            <GlobeHUD
                stats={defaultStats}
                attacks={[]}
                selectedCountry={{ code: 'TR', name: 'Türkiye', count: 5 }}
                onCloseCountry={onClose}
            />
        );
        expect(screen.getByText('Türkiye')).toBeInTheDocument();
        expect(screen.getByText('5')).toBeInTheDocument();
    });

    it('does not render drill-down when no selectedCountry', () => {
        render(<GlobeHUD stats={defaultStats} attacks={[]} />);
        expect(screen.queryByText('DETAYLI ANALİZ')).not.toBeInTheDocument();
    });
});
