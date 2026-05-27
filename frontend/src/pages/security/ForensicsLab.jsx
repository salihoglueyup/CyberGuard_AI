import { useState, useMemo } from 'react';
import { Search, FileSearch, Clock, Tag, HardDrive, Cpu, Hash, ChevronRight, ChevronDown, Download, Filter, Eye } from 'lucide-react';

const EVIDENCE_ITEMS = [
    { id: 'EV-001', type: 'disk', label: 'Disk Image - Server Alpha', hash: 'a3f2b8c1d4e5...7890', size: '256 GB', acquired: '2025-12-20 08:30', analyst: 'SOC-L3', status: 'analyzing', chain: ['Acquisition: Agent K', 'Transfer: Secure vault', 'Analysis: SOC-L3'] },
    { id: 'EV-002', type: 'memory', label: 'RAM Dump - WS-042', hash: 'b7e9f0a2c3d4...5678', size: '32 GB', acquired: '2025-12-20 09:15', analyst: 'SOC-L2', status: 'complete', chain: ['Acquisition: Agent M', 'Transfer: Evidence locker', 'Analysis: SOC-L2', 'Report: Filed'] },
    { id: 'EV-003', type: 'network', label: 'PCAP - DMZ Segment', hash: 'c5d8e1f2a3b4...9012', size: '4.2 GB', acquired: '2025-12-20 10:00', analyst: 'SOC-L3', status: 'analyzing', chain: ['Capture: Tap-DMZ-01', 'Transfer: Analysis lab'] },
    { id: 'EV-004', type: 'log', label: 'Auth Logs - AD Controller', hash: 'd1a2b3c4e5f6...3456', size: '890 MB', acquired: '2025-12-19 22:00', analyst: 'SOC-L2', status: 'complete', chain: ['Export: DC-01', 'Verified: Hash match', 'Analysis: Complete'] },
    { id: 'EV-005', type: 'malware', label: 'Suspicious Binary - ep-003', hash: 'e4f5a6b7c8d9...7890', size: '2.4 MB', acquired: '2025-12-20 11:30', analyst: 'Malware Lab', status: 'quarantined', chain: ['Isolation: Auto-sandbox', 'Hash verified', 'Detonation pending'] },
];

const TIMELINE_EVENTS = [
    { time: '2025-12-20 07:42', event: 'Anomalous login detected', severity: 'high', source: 'SIEM', details: 'User admin_backup logged in from unusual IP 185.xxx.xxx.42' },
    { time: '2025-12-20 07:45', event: 'Lateral movement detected', severity: 'critical', source: 'EDR', details: 'PsExec execution from WS-042 to SRV-ALPHA' },
    { time: '2025-12-20 07:48', event: 'Data exfiltration attempt', severity: 'critical', source: 'DLP', details: '2.4GB transfer to external IP via encrypted channel' },
    { time: '2025-12-20 07:50', event: 'Containment initiated', severity: 'medium', source: 'SOAR', details: 'Auto-isolation of WS-042 and SRV-ALPHA' },
    { time: '2025-12-20 07:52', event: 'Malware sample collected', severity: 'high', source: 'EDR', details: 'Suspicious binary quarantined from WS-042' },
    { time: '2025-12-20 08:00', event: 'Incident escalated to L3', severity: 'medium', source: 'ITSM', details: 'INC-2025-1220-001 created' },
    { time: '2025-12-20 08:30', event: 'Disk image acquired', severity: 'low', source: 'Forensics', details: 'SRV-ALPHA full disk imaged' },
    { time: '2025-12-20 09:15', event: 'Memory dump acquired', severity: 'low', source: 'Forensics', details: 'WS-042 volatile memory captured' },
    { time: '2025-12-20 10:00', event: 'PCAP captured', severity: 'low', source: 'Network', details: 'DMZ segment traffic for timeframe' },
    { time: '2025-12-20 14:00', event: 'IOC extraction complete', severity: 'medium', source: 'Malware Lab', details: '12 IOCs identified, shared with TI team' },
];

const SEVERITY_COLORS = { critical: '#ef4444', high: '#ff6d00', medium: '#ffab00', low: 'var(--hud-cyan)' };
const STATUS_COLORS = { analyzing: '#ffab00', complete: '#10b981', quarantined: '#ef4444' };

const IOCS = [
    { type: 'IP', value: '185.xxx.xxx.42', confidence: 95, source: 'Network Analysis' },
    { type: 'Hash', value: 'e4f5a6b7c8d9...', confidence: 100, source: 'Malware Lab' },
    { type: 'Domain', value: 'c2.malicious-domain.xyz', confidence: 88, source: 'DNS Logs' },
    { type: 'URL', value: 'https://c2.malicious-domain.xyz/beacon', confidence: 88, source: 'Proxy Logs' },
    { type: 'Registry', value: 'HKLM\\...\\Run\\svchost_update', confidence: 92, source: 'EDR' },
    { type: 'File', value: 'svchost_update.exe', confidence: 100, source: 'Disk Analysis' },
];

export default function ForensicsLab() {
    const [tab, setTab] = useState('timeline');
    const [selectedEvidence, setSelectedEvidence] = useState(null);
    const [severityFilter, setSeverityFilter] = useState('all');

    const filteredTimeline = severityFilter === 'all' ? TIMELINE_EVENTS : TIMELINE_EVENTS.filter(e => e.severity === severityFilter);

    return (
        <div className="min-h-screen bg-[var(--hud-bg)] relative">
            <div className="border-b border-[var(--hud-border)] px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <FileSearch className="w-5 h-5 text-[var(--hud-cyan)]" />
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">Forensics Lab</h1>
                    <span className="text-[9px] text-red-400 bg-red-500/10 border border-red-500/20 px-2 py-0.5 rounded animate-pulse">AKTIF SORUSTURMA</span>
                </div>
                <span className="text-[10px] text-[var(--hud-text-dim)]">INC-2025-1220-001</span>
            </div>

            {/* Tabs */}
            <div className="flex gap-1 px-6 py-2 border-b border-[var(--hud-border)]">
                {['timeline', 'evidence', 'iocs'].map(t => (
                    <button key={t} onClick={() => setTab(t)}
                        className={`px-3 py-1.5 rounded text-[10px] uppercase tracking-wide transition-all ${tab === t ? 'bg-cyan-500/10 text-[var(--hud-cyan)] border border-[var(--hud-cyan)]/30' : 'text-[var(--hud-text-dim)] hover:text-[var(--hud-text-muted)]'}`}>
                        {t === 'timeline' ? 'ZAMAN CIZGISI' : t === 'evidence' ? 'KANITLAR' : 'IOC\'LAR'}
                    </button>
                ))}
            </div>

            <div className="p-6">
                {tab === 'timeline' && (
                    <div className="space-y-4">
                        <div className="flex items-center gap-2">
                            <Filter className="w-3 h-3 text-[var(--hud-text-dim)]" />
                            {['all', 'critical', 'high', 'medium', 'low'].map(s => (
                                <button key={s} onClick={() => setSeverityFilter(s)}
                                    className={`px-2 py-0.5 rounded text-[9px] uppercase transition-all ${severityFilter === s ? 'bg-cyan-500/10 text-[var(--hud-cyan)] border border-[var(--hud-cyan)]/30' : 'text-[var(--hud-text-dim)] border border-transparent'}`}>
                                    {s === 'all' ? 'TUMU' : s}
                                </button>
                            ))}
                        </div>
                        <div className="relative pl-6 space-y-0">
                            <div className="absolute left-2.5 top-0 bottom-0 w-px bg-[var(--hud-border)]" />
                            {filteredTimeline.map((evt, i) => (
                                <div key={i} className="relative pb-4 group">
                                    <div className="absolute left-[-16px] top-1 w-3 h-3 rounded-full border-2" style={{ borderColor: SEVERITY_COLORS[evt.severity], backgroundColor: `${SEVERITY_COLORS[evt.severity]}30` }} />
                                    <div className="hud-panel p-3 ml-2 hover:border-[rgba(56,189,248,0.15)] transition-colors">
                                        <div className="flex items-center justify-between">
                                            <div className="flex items-center gap-2">
                                                <span className="text-[9px] text-[var(--hud-text-dim)]">{evt.time}</span>
                                                <span className="text-[9px] px-1.5 py-0.5 rounded uppercase" style={{ color: SEVERITY_COLORS[evt.severity], backgroundColor: `${SEVERITY_COLORS[evt.severity]}15` }}>{evt.severity}</span>
                                                <span className="text-[9px] text-[var(--hud-cyan)]/60">[{evt.source}]</span>
                                            </div>
                                        </div>
                                        <div className="text-[11px] text-[var(--hud-text)] mt-1 font-bold">{evt.event}</div>
                                        <div className="text-[10px] text-[var(--hud-text-dim)] mt-0.5">{evt.details}</div>
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {tab === 'evidence' && (
                    <div className="space-y-3">
                        {EVIDENCE_ITEMS.map(ev => (
                            <div key={ev.id} className="hud-panel p-0 overflow-hidden">
                                <button onClick={() => setSelectedEvidence(selectedEvidence === ev.id ? null : ev.id)} className="w-full flex items-center justify-between px-4 py-3 hover:bg-[rgba(56,189,248,0.02)] transition-colors">
                                    <div className="flex items-center gap-3">
                                        {selectedEvidence === ev.id ? <ChevronDown className="w-3 h-3 text-[var(--hud-cyan)]" /> : <ChevronRight className="w-3 h-3 text-[var(--hud-text-dim)]" />}
                                        <span className="text-[10px] text-[var(--hud-cyan)] font-bold">{ev.id}</span>
                                        <span className="text-[10px] text-[var(--hud-text)]">{ev.label}</span>
                                    </div>
                                    <div className="flex items-center gap-3">
                                        <span className="text-[9px] text-[var(--hud-text-dim)]">{ev.size}</span>
                                        <span className="text-[9px] px-1.5 py-0.5 rounded" style={{ color: STATUS_COLORS[ev.status], backgroundColor: `${STATUS_COLORS[ev.status]}15` }}>{ev.status.toUpperCase()}</span>
                                    </div>
                                </button>
                                {selectedEvidence === ev.id && (
                                    <div className="border-t border-[var(--hud-border)] px-4 py-3 space-y-2 bg-[rgba(0,0,0,0.2)]">
                                        <div className="grid grid-cols-2 gap-2 text-[10px]">
                                            <div><span className="text-[var(--hud-text-dim)]">Hash: </span><span className="text-[var(--hud-cyan)]">{ev.hash}</span></div>
                                            <div><span className="text-[var(--hud-text-dim)]">Analist: </span><span className="text-[var(--hud-text)]">{ev.analyst}</span></div>
                                            <div><span className="text-[var(--hud-text-dim)]">Elde Edilme: </span><span className="text-[var(--hud-text)]">{ev.acquired}</span></div>
                                        </div>
                                        <div>
                                            <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mb-1">KANIT ZINCIRI</div>
                                            {ev.chain.map((step, i) => (
                                                <div key={i} className="flex items-center gap-2 text-[9px] text-[var(--hud-text-muted)] py-0.5">
                                                    <span className="text-[var(--hud-cyan)]/40">{i + 1}.</span>{step}
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ))}
                    </div>
                )}

                {tab === 'iocs' && (
                    <div className="space-y-4">
                        <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider">TESPIT EDILEN IOC'LAR — {IOCS.length} GOSTERGE</div>
                        <div className="hud-panel p-0 overflow-hidden">
                            <table className="w-full text-[10px]">
                                <thead>
                                    <tr className="border-b border-[var(--hud-border)]">
                                        <th className="text-left px-4 py-2 text-[var(--hud-text-dim)] tracking-wider font-medium">TIP</th>
                                        <th className="text-left px-4 py-2 text-[var(--hud-text-dim)] tracking-wider font-medium">DEGER</th>
                                        <th className="text-left px-4 py-2 text-[var(--hud-text-dim)] tracking-wider font-medium">GUVEN</th>
                                        <th className="text-left px-4 py-2 text-[var(--hud-text-dim)] tracking-wider font-medium">KAYNAK</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {IOCS.map((ioc, i) => (
                                        <tr key={i} className="border-b border-[rgba(255,255,255,0.02)] hover:bg-[rgba(56,189,248,0.02)]">
                                            <td className="px-4 py-2 text-[var(--hud-cyan)]">{ioc.type}</td>
                                            <td className="px-4 py-2 text-[var(--hud-text)]">{ioc.value}</td>
                                            <td className="px-4 py-2">
                                                <span style={{ color: ioc.confidence >= 90 ? '#10b981' : '#ffab00' }}>{ioc.confidence}%</span>
                                            </td>
                                            <td className="px-4 py-2 text-[var(--hud-text-dim)]">{ioc.source}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
