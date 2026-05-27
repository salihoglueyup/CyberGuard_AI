import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { AmbientLight, BoxGeometry, BufferGeometry, Color, DoubleSide, Fog, GridHelper, Group, Line, LineBasicMaterial, Mesh, MeshBasicMaterial, MeshPhongMaterial, OctahedronGeometry, PerspectiveCamera, PointLight, Raycaster, RingGeometry, Scene, SphereGeometry, Vector2, Vector3, WebGLRenderer } from 'three';
import api from '../../services/api';

import {
    Network, Shield, Wifi, AlertTriangle, Activity, Eye,
    ZoomIn, ZoomOut, RotateCw, Maximize2
} from 'lucide-react';

// --- Force-Directed Layout ---
function forceLayout(nodes, edges, iterations = 80) {
    const pos = nodes.map(() => ({
        x: (Math.random() - 0.5) * 60,
        y: (Math.random() - 0.5) * 40,
        z: (Math.random() - 0.5) * 30,
        vx: 0, vy: 0, vz: 0,
    }));
    const k = 12; // ideal distance
    const gravity = 0.01;

    for (let iter = 0; iter < iterations; iter++) {
        const cooling = 1 - iter / iterations;
        // Repulsion
        for (let i = 0; i < pos.length; i++) {
            for (let j = i + 1; j < pos.length; j++) {
                let dx = pos[i].x - pos[j].x;
                let dy = pos[i].y - pos[j].y;
                let dz = pos[i].z - pos[j].z;
                let dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
                let force = (k * k) / dist * cooling * 0.5;
                let fx = (dx / dist) * force;
                let fy = (dy / dist) * force;
                let fz = (dz / dist) * force;
                pos[i].vx += fx; pos[i].vy += fy; pos[i].vz += fz;
                pos[j].vx -= fx; pos[j].vy -= fy; pos[j].vz -= fz;
            }
        }
        // Attraction (edges)
        edges.forEach(([s, t]) => {
            let dx = pos[t].x - pos[s].x;
            let dy = pos[t].y - pos[s].y;
            let dz = pos[t].z - pos[s].z;
            let dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.1;
            let force = dist / k * cooling * 0.3;
            let fx = (dx / dist) * force;
            let fy = (dy / dist) * force;
            let fz = (dz / dist) * force;
            pos[s].vx += fx; pos[s].vy += fy; pos[s].vz += fz;
            pos[t].vx -= fx; pos[t].vy -= fy; pos[t].vz -= fz;
        });
        // Gravity toward center
        for (let i = 0; i < pos.length; i++) {
            pos[i].vx -= pos[i].x * gravity;
            pos[i].vy -= pos[i].y * gravity;
            pos[i].vz -= pos[i].z * gravity;
        }
        // Apply
        for (let i = 0; i < pos.length; i++) {
            pos[i].x += pos[i].vx * 0.4;
            pos[i].y += pos[i].vy * 0.4;
            pos[i].z += pos[i].vz * 0.4;
            pos[i].vx *= 0.7;
            pos[i].vy *= 0.7;
            pos[i].vz *= 0.7;
        }
    }
    return pos;
}

// --- Generate sample network ---
function generateNetwork() {
    const types = ['server', 'workstation', 'firewall', 'router', 'database', 'iot', 'unknown'];
    const statusOptions = ['healthy', 'healthy', 'healthy', 'warning', 'compromised'];
    const nodes = Array.from({ length: 60 }, (_, i) => ({
        id: i,
        name: `node-${String(i).padStart(3, '0')}`,
        type: types[Math.floor(Math.random() * types.length)],
        status: statusOptions[Math.floor(Math.random() * statusOptions.length)],
        ip: `10.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
        connections: 0,
    }));
    const edges = [];
    for (let i = 0; i < nodes.length; i++) {
        const conns = 1 + Math.floor(Math.random() * 3);
        for (let j = 0; j < conns; j++) {
            const t = Math.floor(Math.random() * nodes.length);
            if (t !== i) {
                edges.push([i, t]);
                nodes[i].connections++;
                nodes[t].connections++;
            }
        }
    }
    return { nodes, edges };
}

const TYPE_COLORS = {
    server: 0x00e5ff,
    workstation: 0x448aff,
    firewall: 0x00e676,
    router: 0xffab00,
    database: 0xb388ff,
    iot: 0x80deea,
    unknown: 0x546e7a,
};

const STATUS_EMISSIVE = {
    healthy: 0x00e676,
    warning: 0xffab00,
    compromised: 0xff003c,
};

const Network3D = () => {
    const containerRef = useRef(null);
    const sceneRef = useRef(null);
    const [attacks, setAttacks] = useState([]);
    const [hovered, setHovered] = useState(null);
    const [autoRotate, setAutoRotate] = useState(true);
    const [networkData] = useState(() => generateNetwork());

    useEffect(() => {
        const loadAttacks = async () => {
            try {
                const res = await api.get('/attack-map/live?limit=20');
                if (res.data.data?.attacks) setAttacks(res.data.data.attacks);
            } catch { /* silent */ }
        };
        loadAttacks();
        const iv = setInterval(loadAttacks, 5000);
        return () => clearInterval(iv);
    }, []);

    useEffect(() => {
        if (!containerRef.current) return;
        const container = containerRef.current;
        const w = container.clientWidth;
        const h = container.clientHeight;

        // Scene
        const scene = new Scene();
        scene.background = new Color(0x060a14);
        scene.fog = new Fog(0x060a14, 60, 200);
        sceneRef.current = scene;

        // Camera
        const camera = new PerspectiveCamera(60, w / h, 0.1, 500);
        camera.position.set(0, 15, 55);
        camera.lookAt(0, 0, 0);

        // Renderer
        const renderer = new WebGLRenderer({ antialias: true, alpha: true });
        renderer.setSize(w, h);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        container.appendChild(renderer.domElement);

        // Lights
        scene.add(new AmbientLight(0x1a2a3a, 1.5));
        const mainLight = new PointLight(0x00e5ff, 0.6, 200);
        mainLight.position.set(30, 40, 30);
        scene.add(mainLight);
        const accentLight = new PointLight(0xff003c, 0.3, 150);
        accentLight.position.set(-30, -20, -20);
        scene.add(accentLight);

        // Grid floor
        const gridHelper = new GridHelper(120, 40, 0x0a1628, 0x0a1628);
        gridHelper.position.y = -25;
        gridHelper.material.opacity = 0.3;
        gridHelper.material.transparent = true;
        scene.add(gridHelper);

        // Force layout
        const { nodes, edges } = networkData;
        const positions = forceLayout(nodes, edges);

        // Node meshes
        const nodeGroup = new Group();
        const nodeMeshes = nodes.map((n, i) => {
            const size = n.type === 'server' ? 0.8 : n.type === 'firewall' ? 0.7 : 0.5;
            const geo = n.type === 'firewall'
                ? new OctahedronGeometry(size)
                : n.type === 'router'
                    ? new BoxGeometry(size, size, size)
                    : new SphereGeometry(size, 16, 16);
            const mat = new MeshPhongMaterial({
                color: TYPE_COLORS[n.type] || 0x546e7a,
                emissive: STATUS_EMISSIVE[n.status] || 0x00e676,
                emissiveIntensity: n.status === 'compromised' ? 0.6 : 0.15,
                transparent: true,
                opacity: 0.9,
                shininess: 80,
            });
            const mesh = new Mesh(geo, mat);
            mesh.position.set(positions[i].x, positions[i].y, positions[i].z);
            mesh.userData = { nodeData: n, index: i };
            nodeGroup.add(mesh);

            // Glow ring for compromised
            if (n.status === 'compromised') {
                const ringGeo = new RingGeometry(size * 1.3, size * 1.6, 24);
                const ringMat = new MeshBasicMaterial({
                    color: 0xff003c,
                    transparent: true,
                    opacity: 0.2,
                    side: DoubleSide,
                    depthWrite: false,
                });
                const ring = new Mesh(ringGeo, ringMat);
                ring.position.copy(mesh.position);
                ring.__pulseRing = true;
                nodeGroup.add(ring);
            }
            return mesh;
        });
        scene.add(nodeGroup);

        // Edge lines
        const edgeGroup = new Group();
        edges.forEach(([s, t]) => {
            const src = positions[s];
            const tgt = positions[t];
            const isAlert = nodes[s].status === 'compromised' || nodes[t].status === 'compromised';
            const pts = [
                new Vector3(src.x, src.y, src.z),
                new Vector3(tgt.x, tgt.y, tgt.z),
            ];
            const geo = new BufferGeometry().setFromPoints(pts);
            const mat = new LineBasicMaterial({
                color: isAlert ? 0xff003c : 0x00e5ff,
                transparent: true,
                opacity: isAlert ? 0.25 : 0.08,
                depthWrite: false,
            });
            edgeGroup.add(new Line(geo, mat));
        });
        scene.add(edgeGroup);

        // Attack particles
        const particles = [];
        for (let i = 0; i < 15; i++) {
            const geo = new SphereGeometry(0.18, 6, 6);
            const mat = new MeshBasicMaterial({ color: 0xff003c, transparent: true, opacity: 0.9 });
            const p = new Mesh(geo, mat);
            const si = Math.floor(Math.random() * nodes.length);
            const ti = Math.floor(Math.random() * nodes.length);
            p.userData = {
                srcPos: new Vector3(positions[si].x, positions[si].y, positions[si].z),
                tgtPos: new Vector3(positions[ti].x, positions[ti].y, positions[ti].z),
                progress: Math.random(),
            };
            p.position.copy(p.userData.srcPos);
            scene.add(p);
            particles.push(p);
        }

        // Raycaster for hover
        const raycaster = new Raycaster();
        const mouse = new Vector2();
        let hoveredMesh = null;

        const onPointerMove = (e) => {
            const rect = container.getBoundingClientRect();
            mouse.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
        };
        container.addEventListener('pointermove', onPointerMove);

        // Camera orbit state
        let angle = 0;
        let targetAngle = 0;
        let isDragging = false;
        let prevX = 0;

        const onDown = (e) => { isDragging = true; prevX = e.clientX; };
        const onUp = () => { isDragging = false; };
        const onDrag = (e) => {
            if (!isDragging) return;
            targetAngle += (e.clientX - prevX) * 0.005;
            prevX = e.clientX;
        };
        container.addEventListener('pointerdown', onDown);
        window.addEventListener('pointerup', onUp);
        window.addEventListener('pointermove', onDrag);

        // Animation
        let raf;
        const animate = () => {
            raf = requestAnimationFrame(animate);

            // Auto-rotate
            if (autoRotate && !isDragging) {
                targetAngle += 0.001;
            }
            angle += (targetAngle - angle) * 0.05;
            camera.position.x = Math.sin(angle) * 55;
            camera.position.z = Math.cos(angle) * 55;
            camera.lookAt(0, 0, 0);

            // Pulse compromised rings
            const time = Date.now() * 0.003;
            nodeGroup.children.forEach(child => {
                if (child.__pulseRing) {
                    child.material.opacity = 0.1 + Math.sin(time) * 0.12;
                    child.rotation.z += 0.01;
                }
            });

            // Node breathing
            nodeMeshes.forEach((mesh, i) => {
                const s = 1 + Math.sin(time + i * 0.5) * 0.06;
                mesh.scale.set(s, s, s);
            });

            // Move attack particles
            particles.forEach(p => {
                p.userData.progress += 0.012;
                if (p.userData.progress >= 1) {
                    p.userData.progress = 0;
                    const si = Math.floor(Math.random() * nodes.length);
                    const ti = Math.floor(Math.random() * nodes.length);
                    p.userData.srcPos.set(positions[si].x, positions[si].y, positions[si].z);
                    p.userData.tgtPos.set(positions[ti].x, positions[ti].y, positions[ti].z);
                }
                p.position.lerpVectors(p.userData.srcPos, p.userData.tgtPos, p.userData.progress);
            });

            // Raycasting
            raycaster.setFromCamera(mouse, camera);
            const hits = raycaster.intersectObjects(nodeMeshes);
            if (hits.length > 0 && hits[0].object.userData.nodeData) {
                if (hoveredMesh !== hits[0].object) {
                    if (hoveredMesh) hoveredMesh.material.emissiveIntensity = hoveredMesh.userData.nodeData?.status === 'compromised' ? 0.6 : 0.15;
                    hoveredMesh = hits[0].object;
                    hoveredMesh.material.emissiveIntensity = 0.8;
                    setHovered(hoveredMesh.userData.nodeData);
                }
            } else {
                if (hoveredMesh) {
                    hoveredMesh.material.emissiveIntensity = hoveredMesh.userData.nodeData?.status === 'compromised' ? 0.6 : 0.15;
                    hoveredMesh = null;
                    setHovered(null);
                }
            }

            renderer.render(scene, camera);
        };
        animate();

        // Resize
        const onResize = () => {
            const nw = container.clientWidth;
            const nh = container.clientHeight;
            camera.aspect = nw / nh;
            camera.updateProjectionMatrix();
            renderer.setSize(nw, nh);
        };
        window.addEventListener('resize', onResize);

        return () => {
            cancelAnimationFrame(raf);
            window.removeEventListener('resize', onResize);
            window.removeEventListener('pointerup', onUp);
            window.removeEventListener('pointermove', onDrag);
            container.removeEventListener('pointermove', onPointerMove);
            container.removeEventListener('pointerdown', onDown);
            scene.traverse(obj => {
                if (obj.geometry) obj.geometry.dispose();
                if (obj.material) {
                    if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
                    else obj.material.dispose();
                }
            });
            renderer.dispose();
            if (renderer.domElement.parentNode === container) {
                container.removeChild(renderer.domElement);
            }
        };
    }, [networkData, autoRotate]);

    const summary = useMemo(() => {
        const { nodes, edges } = networkData;
        return {
            total: nodes.length,
            healthy: nodes.filter(n => n.status === 'healthy').length,
            warning: nodes.filter(n => n.status === 'warning').length,
            compromised: nodes.filter(n => n.status === 'compromised').length,
            connections: edges.length,
        };
    }, [networkData]);

    return (
        <div className="relative w-full h-full min-h-screen" style={{ background: 'var(--hud-bg)' }}>

            {/* Header */}
            <div className="absolute top-4 left-4 right-4 z-20 flex items-center justify-between">
                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-2 px-3 py-1.5">
                        <Network size={16} style={{ color: 'var(--hud-cyan)' }} />
                        <span className="font-mono text-sm font-bold tracking-wider" style={{ color: 'var(--hud-text)' }}>
                            3D Network Topology
                        </span>
                    </div>
                    <div className="flex items-center gap-1.5">
                        <div className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: 'var(--hud-emerald)' }} />
                        <span className="font-mono text-[9px] tracking-wider" style={{ color: 'var(--hud-text-muted)' }}>LIVE</span>
                    </div>
                </div>

                {/* Controls */}
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => setAutoRotate(!autoRotate)}
                        className="w-8 h-8 flex items-center justify-center rounded-md transition-all"
                        style={{
                            background: autoRotate ? 'rgba(56,189,248,0.12)' : 'var(--hud-surface)',
                            border: `1px solid ${autoRotate ? 'rgba(56,189,248,0.3)' : 'var(--hud-border)'}`,
                            color: autoRotate ? 'var(--hud-cyan)' : 'var(--hud-text-muted)',
                        }}
                    >
                        <RotateCw size={14} />
                    </button>
                </div>
            </div>

            {/* Stats bar */}
            <div className="absolute top-16 left-4 z-20 flex gap-2">
                <StatChip label="NODES" value={summary.total} color="var(--hud-cyan)" icon={<Wifi size={10} />} />
                <StatChip label="HEALTHY" value={summary.healthy} color="var(--hud-emerald)" icon={<Shield size={10} />} />
                <StatChip label="WARNING" value={summary.warning} color="var(--hud-amber)" icon={<Activity size={10} />} />
                <StatChip label="COMPROMISED" value={summary.compromised} color="var(--hud-red)" icon={<AlertTriangle size={10} />} />
                <StatChip label="EDGES" value={summary.connections} color="var(--hud-blue)" icon={<Network size={10} />} />
            </div>

            {/* Hover tooltip */}
            {hovered && (
                <div className="absolute bottom-20 left-1/2 -translate-x-1/2 z-20 px-4 py-3 rounded-lg" style={{
                    background: 'var(--hud-surface-elevated)',
                    border: '1px solid var(--hud-border-strong)',
                    boxShadow: 'var(--hud-shadow-lg)',
                    minWidth: 240,
                }}>
                    <div className="flex items-center gap-2 mb-2">
                        <div className="w-2 h-2 rounded-full" style={{
                            background: hovered.status === 'compromised' ? 'var(--hud-red)' : hovered.status === 'warning' ? 'var(--hud-amber)' : 'var(--hud-emerald)',
                        }} />
                        <span className="font-mono text-xs font-bold" style={{ color: 'var(--hud-text)' }}>{hovered.name}</span>
                        <span className="font-mono text-[8px] px-1.5 py-0.5 rounded ml-auto" style={{
                            background: 'rgba(56,189,248,0.08)',
                            color: 'var(--hud-cyan)',
                            border: '1px solid rgba(56,189,248,0.15)',
                        }}>
                            {hovered.type.toUpperCase()}
                        </span>
                    </div>
                    <div className="grid grid-cols-3 gap-3 font-mono text-[9px]">
                        <div>
                            <div style={{ color: 'var(--hud-text-dim)' }}>IP</div>
                            <div style={{ color: 'var(--hud-text)' }}>{hovered.ip}</div>
                        </div>
                        <div>
                            <div style={{ color: 'var(--hud-text-dim)' }}>STATUS</div>
                            <div style={{
                                color: hovered.status === 'compromised' ? 'var(--hud-red)' : hovered.status === 'warning' ? 'var(--hud-amber)' : 'var(--hud-emerald)'
                            }}>
                                {hovered.status.toUpperCase()}
                            </div>
                        </div>
                        <div>
                            <div style={{ color: 'var(--hud-text-dim)' }}>CONNS</div>
                            <div style={{ color: 'var(--hud-text)' }}>{hovered.connections}</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Legend */}
            <div className="absolute bottom-4 left-4 z-20 px-3 py-2 rounded-lg" style={{
                background: 'var(--hud-surface)',
                border: '1px solid var(--hud-border)',
            }}>
                <div className="font-mono text-[8px] tracking-widest mb-1.5" style={{ color: 'var(--hud-text-dim)' }}>NODE TYPES</div>
                <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    {Object.entries(TYPE_COLORS).map(([type, color]) => (
                        <div key={type} className="flex items-center gap-1.5">
                            <div className="w-2 h-2 rounded-full" style={{ background: `#${color.toString(16).padStart(6, '0')}` }} />
                            <span className="font-mono text-[8px]" style={{ color: 'var(--hud-text-muted)' }}>
                                {type}
                            </span>
                        </div>
                    ))}
                </div>
            </div>

            {/* 3D Container */}
            <div ref={containerRef} className="w-full h-screen" />
        </div>
    );
};

function StatChip({ label, value, color, icon }) {
    return (
        <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-md" style={{
            background: 'var(--hud-surface)',
            border: '1px solid var(--hud-border)',
        }}>
            <span style={{ color }}>{icon}</span>
            <div>
                <div className="font-mono text-[7px] tracking-widest" style={{ color: 'var(--hud-text-dim)' }}>{label}</div>
                <div className="font-mono text-sm font-bold" style={{ color }}>{value}</div>
            </div>
        </div>
    );
}

export default Network3D;
