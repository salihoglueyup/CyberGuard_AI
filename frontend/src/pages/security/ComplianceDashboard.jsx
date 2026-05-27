import { useState, useMemo } from 'react';
import { ClipboardCheck, CheckCircle2, XCircle, AlertTriangle, Clock, Shield, FileText, Filter, ChevronDown, ChevronRight } from 'lucide-react';

const FRAMEWORKS = [
    { id: 'nist', name: 'NIST CSF 2.0', version: '2.0', total: 23, passed: 18, failed: 3, partial: 2 },
    { id: 'iso27001', name: 'ISO 27001:2022', version: '2022', total: 93, passed: 71, failed: 12, partial: 10 },
    { id: 'pci', name: 'PCI DSS 4.0', version: '4.0', total: 64, passed: 52, failed: 6, partial: 6 },
    { id: 'gdpr', name: 'GDPR', version: '2018', total: 42, passed: 35, failed: 4, partial: 3 },
    { id: 'hipaa', name: 'HIPAA', version: '2023', total: 18, passed: 14, failed: 2, partial: 2 },
    { id: 'soc2', name: 'SOC 2 Type II', version: '2023', total: 61, passed: 48, failed: 7, partial: 6 },
];

const CONTROLS_DATA = {
    nist: [
        { id: 'GV.OC-01', category: 'GOVERN', title: 'Organizational Context', status: 'pass', evidence: 'Policy document v3.2 approved', lastAudit: '2025-12-15' },
        { id: 'GV.RM-01', category: 'GOVERN', title: 'Risk Management Strategy', status: 'pass', evidence: 'Risk register updated quarterly', lastAudit: '2025-12-10' },
        { id: 'ID.AM-01', category: 'IDENTIFY', title: 'Asset Management', status: 'fail', evidence: 'Shadow IT assets not inventoried', lastAudit: '2025-12-01' },
        { id: 'PR.AC-01', category: 'PROTECT', title: 'Access Control', status: 'pass', evidence: 'MFA enforced organization-wide', lastAudit: '2025-12-20' },
        { id: 'PR.DS-01', category: 'PROTECT', title: 'Data Security', status: 'partial', evidence: 'Encryption at rest: 85% coverage', lastAudit: '2025-11-30' },
        { id: 'DE.CM-01', category: 'DETECT', title: 'Continuous Monitoring', status: 'pass', evidence: 'SIEM operational 24/7', lastAudit: '2025-12-22' },
        { id: 'DE.AE-01', category: 'DETECT', title: 'Adverse Event Analysis', status: 'pass', evidence: 'SOC Level 1-3 staffed', lastAudit: '2025-12-18' },
        { id: 'RS.AN-01', category: 'RESPOND', title: 'Incident Analysis', status: 'pass', evidence: 'IR playbooks tested', lastAudit: '2025-12-12' },
        { id: 'RS.MI-01', category: 'RESPOND', title: 'Incident Mitigation', status: 'fail', evidence: 'Auto-containment gap in OT', lastAudit: '2025-11-25' },
        { id: 'RC.RP-01', category: 'RECOVER', title: 'Recovery Planning', status: 'pass', evidence: 'DR tested biannually', lastAudit: '2025-12-05' },
    ],
    iso27001: [
        { id: 'A.5.1', category: 'ORGANIZATIONAL', title: 'Information Security Policies', status: 'pass', evidence: '12 policies reviewed', lastAudit: '2025-12-20' },
        { id: 'A.6.1', category: 'PEOPLE', title: 'Screening', status: 'pass', evidence: 'Background checks 100%', lastAudit: '2025-12-15' },
        { id: 'A.7.1', category: 'PHYSICAL', title: 'Physical Security Perimeters', status: 'partial', evidence: 'Branch offices pending', lastAudit: '2025-12-01' },
        { id: 'A.8.1', category: 'TECHNOLOGICAL', title: 'Endpoint Security', status: 'pass', evidence: 'EDR deployed 98%', lastAudit: '2025-12-22' },
        { id: 'A.8.9', category: 'TECHNOLOGICAL', title: 'Configuration Management', status: 'fail', evidence: 'CIS benchmarks 72% compliant', lastAudit: '2025-11-28' },
    ],
};

const STATUS_CONFIG = {
    pass: { icon: CheckCircle2, color: '#10b981', label: 'BASARILI' },
    fail: { icon: XCircle, color: '#ef4444', label: 'BASARISIZ' },
    partial: { icon: AlertTriangle, color: '#ffab00', label: 'KISMI' },
};

export default function ComplianceDashboard() {
    const [activeFramework, setActiveFramework] = useState('nist');
    const [statusFilter, setStatusFilter] = useState('all');
    const [expandedCategories, setExpandedCategories] = useState(new Set());

    const framework = FRAMEWORKS.find(f => f.id === activeFramework);
    const controls = CONTROLS_DATA[activeFramework] || [];
    const filteredControls = statusFilter === 'all' ? controls : controls.filter(c => c.status === statusFilter);

    const categories = useMemo(() => {
        const map = {};
        filteredControls.forEach(c => {
            if (!map[c.category]) map[c.category] = [];
            map[c.category].push(c);
        });
        return map;
    }, [filteredControls]);

    const toggleCategory = (cat) => {
        setExpandedCategories(prev => {
            const next = new Set(prev);
            next.has(cat) ? next.delete(cat) : next.add(cat);
            return next;
        });
    };

    const overallScore = framework ? Math.round((framework.passed / framework.total) * 100) : 0;

    return (
        <div className="min-h-screen bg-[var(--hud-bg)] relative">
            <div className="border-b border-[var(--hud-border)] px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <ClipboardCheck className="w-5 h-5 text-[var(--hud-cyan)]" />
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">Compliance Dashboard</h1>
                </div>
                <div className="flex items-center gap-2 text-[10px]">
                    <span className="text-[var(--hud-text-dim)]">SON DENETIM:</span>
                    <span className="text-[var(--hud-cyan)]">2025-12-22 14:30 UTC</span>
                </div>
            </div>

            <div className="p-6 space-y-6">
                {/* Framework tabs */}
                <div className="flex gap-2 overflow-x-auto pb-2">
                    {FRAMEWORKS.map(fw => (
                        <button key={fw.id} onClick={() => { setActiveFramework(fw.id); setStatusFilter('all'); }}
                            className={`flex-shrink-0 px-4 py-2 rounded border text-[10px] tracking-wider transition-all ${
                                activeFramework === fw.id
                                    ? 'border-[var(--hud-cyan)]/40 text-[var(--hud-cyan)] bg-cyan-500/10'
                                    : 'border-[var(--hud-border)] text-[var(--hud-text-dim)] hover:text-[var(--hud-text-muted)]'
                            }`}>
                            {fw.name}
                        </button>
                    ))}
                </div>

                {/* Score overview */}
                <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
                    <div className="md:col-span-2 hud-panel flex items-center gap-6">
                        <div className="relative w-24 h-24">
                            <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
                                <circle cx="50" cy="50" r="42" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="8" />
                                <circle cx="50" cy="50" r="42" fill="none" stroke={overallScore >= 80 ? '#10b981' : overallScore >= 60 ? '#ffab00' : '#ef4444'} strokeWidth="8"
                                    strokeDasharray={`${overallScore * 2.64} ${264 - overallScore * 2.64}`} strokeLinecap="round" />
                            </svg>
                            <div className="absolute inset-0 flex items-center justify-center">
                                <span className="text-2xl font-bold" style={{ color: overallScore >= 80 ? '#10b981' : overallScore >= 60 ? '#ffab00' : '#ef4444' }}>{overallScore}%</span>
                            </div>
                        </div>
                        <div>
                            <div className="text-sm text-[var(--hud-text)] font-bold">{framework?.name}</div>
                            <div className="text-[9px] text-[var(--hud-text-dim)] mt-1">Versiyon {framework?.version}</div>
                            <div className="text-[9px] text-[var(--hud-text-dim)]">Toplam {framework?.total} kontrol</div>
                        </div>
                    </div>
                    {[
                        { label: 'BASARILI', value: framework?.passed, color: '#10b981' },
                        { label: 'BASARISIZ', value: framework?.failed, color: '#ef4444' },
                        { label: 'KISMI', value: framework?.partial, color: '#ffab00' },
                    ].map(s => (
                        <div key={s.label} className="hud-panel flex flex-col items-center justify-center">
                            <span className="text-2xl font-bold tabular-nums" style={{ color: s.color }}>{s.value}</span>
                            <span className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mt-1">{s.label}</span>
                        </div>
                    ))}
                </div>

                {/* Filters */}
                <div className="flex items-center gap-2">
                    <Filter className="w-3 h-3 text-[var(--hud-text-dim)]" />
                    {['all', 'pass', 'fail', 'partial'].map(s => (
                        <button key={s} onClick={() => setStatusFilter(s)}
                            className={`px-2 py-1 rounded border text-[10px] uppercase transition-all ${statusFilter === s ? 'border-[var(--hud-cyan)]/40 text-[var(--hud-cyan)] bg-cyan-500/10' : 'border-[var(--hud-border)] text-[var(--hud-text-dim)]'}`}>
                            {s === 'all' ? 'TUMU' : STATUS_CONFIG[s]?.label || s}
                        </button>
                    ))}
                </div>

                {/* Controls by category */}
                <div className="space-y-2">
                    {Object.entries(categories).map(([cat, items]) => (
                        <div key={cat} className="hud-panel p-0 overflow-hidden">
                            <button onClick={() => toggleCategory(cat)} className="w-full flex items-center justify-between px-4 py-2.5 hover:bg-[rgba(56,189,248,0.03)] transition-colors">
                                <div className="flex items-center gap-2">
                                    {expandedCategories.has(cat) ? <ChevronDown className="w-3 h-3 text-[var(--hud-cyan)]" /> : <ChevronRight className="w-3 h-3 text-[var(--hud-text-dim)]" />}
                                    <span className="text-[10px] text-[var(--hud-cyan)] tracking-wider font-bold">{cat}</span>
                                    <span className="text-[9px] text-[var(--hud-text-dim)]">({items.length})</span>
                                </div>
                                <div className="flex gap-2">
                                    {['pass', 'fail', 'partial'].map(s => {
                                        const c = items.filter(i => i.status === s).length;
                                        if (!c) return null;
                                        return <span key={s} className="text-[9px] px-1.5 py-0.5 rounded" style={{ color: STATUS_CONFIG[s].color, backgroundColor: `${STATUS_CONFIG[s].color}15` }}>{c}</span>;
                                    })}
                                </div>
                            </button>
                            {expandedCategories.has(cat) && (
                                <div className="border-t border-[var(--hud-border)]">
                                    {items.map(ctrl => {
                                        const SC = STATUS_CONFIG[ctrl.status];
                                        return (
                                            <div key={ctrl.id} className="flex items-center gap-4 px-4 py-2 border-b border-[rgba(255,255,255,0.02)] hover:bg-[rgba(56,189,248,0.02)] transition-colors">
                                                <SC.icon className="w-4 h-4 flex-shrink-0" style={{ color: SC.color }} />
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2">
                                                        <span className="text-[10px] text-[var(--hud-cyan)] font-bold">{ctrl.id}</span>
                                                        <span className="text-[10px] text-[var(--hud-text-muted)]">{ctrl.title}</span>
                                                    </div>
                                                    <div className="text-[9px] text-[var(--hud-text-dim)] mt-0.5">{ctrl.evidence}</div>
                                                </div>
                                                <div className="flex items-center gap-1 text-[9px] text-[var(--hud-text-dim)]">
                                                    <Clock className="w-3 h-3" />{ctrl.lastAudit}
                                                </div>
                                            </div>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
