import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { Network, Server, Shield, Wifi, Database, Globe, Cpu, Lock, ZoomIn, ZoomOut, Maximize2, RotateCcw } from 'lucide-react';

const NODE_TYPES = {
    firewall: { icon: '🛡️', color: '#ef4444', label: 'Firewall' },
    switch: { icon: '🔀', color: 'var(--hud-cyan)', label: 'Switch' },
    server: { icon: '🖥️', color: '#b388ff', label: 'Server' },
    router: { icon: '🌐', color: '#10b981', label: 'Router' },
    endpoint: { icon: '💻', color: '#448aff', label: 'Endpoint' },
    database: { icon: '🗄️', color: '#ffab00', label: 'Database' },
    cloud: { icon: '☁️', color: '#00b0ff', label: 'Cloud' },
    iot: { icon: '📡', color: '#ff6d00', label: 'IoT' },
};

function generateTopology() {
    const nodes = [];
    const links = [];
    // Core layer
    nodes.push({ id: 'fw-1', type: 'firewall', label: 'FW-CORE-01', x: 400, y: 80, status: 'active', traffic: 94 });
    nodes.push({ id: 'fw-2', type: 'firewall', label: 'FW-CORE-02', x: 600, y: 80, status: 'active', traffic: 87 });
    // Distribution
    nodes.push({ id: 'sw-1', type: 'switch', label: 'SW-DIST-01', x: 200, y: 200, status: 'active', traffic: 72 });
    nodes.push({ id: 'sw-2', type: 'switch', label: 'SW-DIST-02', x: 500, y: 200, status: 'active', traffic: 65 });
    nodes.push({ id: 'sw-3', type: 'switch', label: 'SW-DIST-03', x: 800, y: 200, status: 'warning', traffic: 91 });
    nodes.push({ id: 'rt-1', type: 'router', label: 'RT-EDGE-01', x: 500, y: 50, status: 'active', traffic: 78 });
    // Servers
    for (let i = 0; i < 6; i++) {
        nodes.push({ id: `srv-${i}`, type: 'server', label: `SRV-${String(i + 1).padStart(2, '0')}`, x: 100 + i * 150, y: 340, status: i === 3 ? 'critical' : 'active', traffic: 30 + Math.random() * 60 });
    }
    // Databases
    nodes.push({ id: 'db-1', type: 'database', label: 'DB-PRIMARY', x: 300, y: 460, status: 'active', traffic: 55 });
    nodes.push({ id: 'db-2', type: 'database', label: 'DB-REPLICA', x: 600, y: 460, status: 'active', traffic: 42 });
    // Cloud
    nodes.push({ id: 'cloud-1', type: 'cloud', label: 'AWS-VPC', x: 150, y: 560, status: 'active', traffic: 38 });
    nodes.push({ id: 'cloud-2', type: 'cloud', label: 'AZURE-VNET', x: 450, y: 560, status: 'active', traffic: 45 });
    // IoT
    for (let i = 0; i < 4; i++) {
        nodes.push({ id: `iot-${i}`, type: 'iot', label: `IOT-SENS-${i + 1}`, x: 650 + (i % 2) * 120, y: 460 + Math.floor(i / 2) * 100, status: i === 2 ? 'warning' : 'active', traffic: 10 + Math.random() * 30 });
    }
    // Endpoints
    for (let i = 0; i < 8; i++) {
        nodes.push({ id: `ep-${i}`, type: 'endpoint', label: `WS-${String(i + 1).padStart(3, '0')}`, x: 50 + i * 115, y: 660, status: 'active', traffic: 5 + Math.random() * 40 });
    }

    // Links
    links.push({ from: 'rt-1', to: 'fw-1' }, { from: 'rt-1', to: 'fw-2' });
    links.push({ from: 'fw-1', to: 'sw-1' }, { from: 'fw-1', to: 'sw-2' });
    links.push({ from: 'fw-2', to: 'sw-2' }, { from: 'fw-2', to: 'sw-3' });
    for (let i = 0; i < 6; i++) {
        links.push({ from: i < 2 ? 'sw-1' : i < 4 ? 'sw-2' : 'sw-3', to: `srv-${i}` });
    }
    links.push({ from: 'srv-1', to: 'db-1' }, { from: 'srv-3', to: 'db-2' });
    links.push({ from: 'sw-1', to: 'cloud-1' }, { from: 'sw-2', to: 'cloud-2' });
    links.push({ from: 'sw-3', to: 'iot-0' }, { from: 'sw-3', to: 'iot-1' });
    links.push({ from: 'iot-0', to: 'iot-2' }, { from: 'iot-1', to: 'iot-3' });
    for (let i = 0; i < 8; i++) {
        links.push({ from: i < 3 ? 'sw-1' : i < 6 ? 'sw-2' : 'sw-3', to: `ep-${i}` });
    }
    return { nodes, links };
}

const STATUS_COLORS = { active: '#10b981', warning: '#ffab00', critical: '#ef4444', inactive: '#555' };

export default function TopologyMap() {
    const canvasRef = useRef(null);
    const [topology] = useState(() => generateTopology());
    const [selected, setSelected] = useState(null);
    const [zoom, setZoom] = useState(1);
    const [pan, setPan] = useState({ x: 0, y: 0 });
    const [dragging, setDragging] = useState(false);
    const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

    const nodeMap = useMemo(() => {
        const map = {};
        topology.nodes.forEach(n => { map[n.id] = n; });
        return map;
    }, [topology]);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width = canvas.clientWidth;
        const h = canvas.height = canvas.clientHeight;

        function draw() {
            ctx.clearRect(0, 0, w, h);
            ctx.save();
            ctx.translate(pan.x, pan.y);
            ctx.scale(zoom, zoom);

            // Draw links
            topology.links.forEach(link => {
                const from = nodeMap[link.from];
                const to = nodeMap[link.to];
                if (!from || !to) return;
                ctx.beginPath();
                ctx.moveTo(from.x, from.y);
                ctx.lineTo(to.x, to.y);
                ctx.strokeStyle = 'rgba(56,189,248,0.12)';
                ctx.lineWidth = 1;
                ctx.stroke();
                // Animate pulse
                const t = (Date.now() % 3000) / 3000;
                const px = from.x + (to.x - from.x) * t;
                const py = from.y + (to.y - from.y) * t;
                ctx.beginPath();
                ctx.arc(px, py, 2, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(56,189,248,0.3)';
                ctx.fill();
            });

            // Draw nodes
            topology.nodes.forEach(node => {
                const type = NODE_TYPES[node.type];
                const isSelected = selected?.id === node.id;
                // Outer glow
                if (isSelected || node.status === 'critical') {
                    ctx.beginPath();
                    ctx.arc(node.x, node.y, 24, 0, Math.PI * 2);
                    ctx.fillStyle = isSelected ? 'rgba(56,189,248,0.1)' : 'rgba(239,68,68,0.15)';
                    ctx.fill();
                }
                // Node circle
                ctx.beginPath();
                ctx.arc(node.x, node.y, 16, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(6,10,20,0.9)';
                ctx.fill();
                ctx.strokeStyle = STATUS_COLORS[node.status] || type.color;
                ctx.lineWidth = isSelected ? 2 : 1;
                ctx.stroke();
                // Icon
                ctx.font = '12px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(type.icon, node.x, node.y);
                // Label
                ctx.font = '8px "JetBrains Mono", monospace';
                ctx.fillStyle = 'rgba(255,255,255,0.5)';
                ctx.fillText(node.label, node.x, node.y + 26);
            });

            ctx.restore();
        }

        draw();
        const anim = setInterval(draw, 50);
        return () => clearInterval(anim);
    }, [topology, nodeMap, selected, zoom, pan]);

    const handleCanvasClick = useCallback((e) => {
        const rect = canvasRef.current.getBoundingClientRect();
        const mx = (e.clientX - rect.left - pan.x) / zoom;
        const my = (e.clientY - rect.top - pan.y) / zoom;
        const clicked = topology.nodes.find(n => Math.hypot(n.x - mx, n.y - my) < 18);
        setSelected(clicked || null);
    }, [topology, zoom, pan]);

    return (
        <div className="min-h-screen bg-[var(--hud-bg)] relative">
            <div className="border-b border-[var(--hud-border)] px-6 py-3 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <Network className="w-5 h-5 text-[var(--hud-cyan)]" />
                    <h1 className="text-xl font-semibold text-[var(--hud-text)]">Network Topology</h1>
                    <span className="text-[9px] text-[var(--hud-text-dim)] bg-[rgba(56,189,248,0.08)] border border-[var(--hud-border)] px-2 py-0.5 rounded">
                        {topology.nodes.length} NODES • {topology.links.length} LINKS
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <button aria-label="Yakınlaştır" onClick={() => setZoom(z => Math.min(2, z + 0.2))} className="p-1 rounded border border-[var(--hud-border)] text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)] transition-colors"><ZoomIn className="w-4 h-4" /></button>
                    <button aria-label="Uzaklaştır" onClick={() => setZoom(z => Math.max(0.3, z - 0.2))} className="p-1 rounded border border-[var(--hud-border)] text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)] transition-colors"><ZoomOut className="w-4 h-4" /></button>
                    <button aria-label="Sıfırla" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} className="p-1 rounded border border-[var(--hud-border)] text-[var(--hud-text-dim)] hover:text-[var(--hud-cyan)] transition-colors"><RotateCcw className="w-4 h-4" /></button>
                </div>
            </div>

            <div className="flex" style={{ height: 'calc(100vh - 56px)' }}>
                {/* Canvas */}
                <canvas
                    ref={canvasRef}
                    className="flex-1 cursor-crosshair"
                    onClick={handleCanvasClick}
                    onMouseDown={e => { setDragging(true); setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y }); }}
                    onMouseMove={e => { if (dragging) setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y }); }}
                    onMouseUp={() => setDragging(false)}
                    onMouseLeave={() => setDragging(false)}
                />

                {/* Side panel */}
                <div className="w-72 border-l border-[var(--hud-border)] p-4 overflow-y-auto">
                    {selected ? (
                        <div className="space-y-4">
                            <div>
                                <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mb-1">SECILI DUGUM</div>
                                <div className="text-sm text-[var(--hud-cyan)] font-bold">{selected.label}</div>
                                <div className="text-[10px] text-[var(--hud-text-dim)]">{NODE_TYPES[selected.type]?.label}</div>
                            </div>
                            <div className="space-y-2">
                                <div className="flex justify-between text-[10px]">
                                    <span className="text-[var(--hud-text-dim)]">Durum</span>
                                    <span style={{ color: STATUS_COLORS[selected.status] }}>{selected.status.toUpperCase()}</span>
                                </div>
                                <div className="flex justify-between text-[10px]">
                                    <span className="text-[var(--hud-text-dim)]">Trafik</span>
                                    <span className="text-[var(--hud-cyan)]">{selected.traffic?.toFixed(0)}%</span>
                                </div>
                                <div className="h-1.5 bg-[rgba(255,255,255,0.04)] rounded-sm border border-[var(--hud-border)]">
                                    <div className="h-full rounded-sm" style={{ width: `${selected.traffic}%`, backgroundColor: selected.traffic > 80 ? '#ef4444' : 'var(--hud-cyan)' }} />
                                </div>
                            </div>
                            <div>
                                <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mb-2">BAGLANTI</div>
                                {topology.links.filter(l => l.from === selected.id || l.to === selected.id).map((l, i) => {
                                    const peerId = l.from === selected.id ? l.to : l.from;
                                    const peer = nodeMap[peerId];
                                    return (
                                        <div key={i} className="flex items-center gap-2 text-[10px] py-1 border-b border-[rgba(255,255,255,0.03)]" onClick={() => setSelected(peer)} style={{ cursor: 'pointer' }}>
                                            <span>{NODE_TYPES[peer?.type]?.icon}</span>
                                            <span className="text-[var(--hud-text-muted)] hover:text-[var(--hud-cyan)]">{peer?.label}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider">TOPOLOJI OZETI</div>
                            {Object.entries(NODE_TYPES).map(([key, val]) => {
                                const count = topology.nodes.filter(n => n.type === key).length;
                                if (!count) return null;
                                return (
                                    <div key={key} className="flex items-center justify-between text-[10px]">
                                        <span className="text-[var(--hud-text-muted)]">{val.icon} {val.label}</span>
                                        <span className="text-[var(--hud-cyan)] font-bold">{count}</span>
                                    </div>
                                );
                            })}
                            <div className="border-t border-[var(--hud-border)] pt-2 mt-2">
                                <div className="text-[9px] text-[var(--hud-text-dim)] tracking-wider mb-2">DURUM</div>
                                {Object.entries(STATUS_COLORS).map(([status, color]) => {
                                    const count = topology.nodes.filter(n => n.status === status).length;
                                    if (!count) return null;
                                    return (
                                        <div key={status} className="flex items-center justify-between text-[10px]">
                                            <span className="text-[var(--hud-text-muted)]">{status.toUpperCase()}</span>
                                            <span style={{ color }} className="font-bold">{count}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
