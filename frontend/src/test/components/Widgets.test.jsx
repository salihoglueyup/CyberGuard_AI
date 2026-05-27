import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';

// Mock heavy dependencies
vi.mock('react-globe.gl', () => ({ default: () => <div data-testid="globe">Globe Mock</div> }));
vi.mock('framer-motion', () => ({
    motion: { div: (props) => <div {...props} />, span: (props) => <span {...props} /> },
    AnimatePresence: ({ children }) => <>{children}</>,
}));

describe('Widget components', () => {
    it('ThreatRadarWidget renders', async () => {
        vi.mock('../../components/charts', () => ({
            HudRadarChart: () => <div data-testid="radar">Radar</div>,
        }));
        const { default: ThreatRadarWidget } = await import('../../components/widgets/ThreatRadarWidget');
        render(<ThreatRadarWidget />);
        expect(screen.getByText(/TEHDIT RADAR/i)).toBeInTheDocument();
    });

    it('MiniStatWidget renders with props', async () => {
        const { default: MiniStatWidget } = await import('../../components/widgets/MiniStatWidget');
        const { Shield } = await import('lucide-react');
        render(
            <MiniStatWidget
                icon={Shield}
                label="Test"
                value="42"
                unit="req/s"
                trend={5.2}
                color="var(--hud-cyan)"
            />
        );
        expect(screen.getByText('Test')).toBeInTheDocument();
        expect(screen.getByText('42')).toBeInTheDocument();
    });
});
