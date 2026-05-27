import { useRef, useEffect, useState, useMemo, useCallback } from 'react';
import Globe from 'react-globe.gl';
import { DirectionalLight, DoubleSide, HemisphereLight, Mesh, MeshBasicMaterial, PointLight, RingGeometry, SphereGeometry, Vector3 } from 'three';

// --- Country Data ---
const COUNTRIES_DATA = {
    TR: { name: 'Turkiye', lat: 39.0, lng: 35.0 },
    US: { name: 'ABD', lat: 38.0, lng: -97.0 },
    CN: { name: 'Cin', lat: 35.0, lng: 105.0 },
    RU: { name: 'Rusya', lat: 60.0, lng: 100.0 },
    DE: { name: 'Almanya', lat: 51.0, lng: 9.0 },
    GB: { name: 'Ingiltere', lat: 54.0, lng: -2.0 },
    FR: { name: 'Fransa', lat: 46.0, lng: 2.0 },
    JP: { name: 'Japonya', lat: 36.0, lng: 138.0 },
    KR: { name: 'Guney Kore', lat: 36.0, lng: 128.0 },
    IN: { name: 'Hindistan', lat: 21.0, lng: 78.0 },
    BR: { name: 'Brezilya', lat: -10.0, lng: -55.0 },
    AU: { name: 'Avustralya', lat: -25.0, lng: 135.0 },
    CA: { name: 'Kanada', lat: 56.0, lng: -106.0 },
    IT: { name: 'Italya', lat: 42.8, lng: 12.8 },
    ES: { name: 'Ispanya', lat: 40.0, lng: -4.0 },
    NL: { name: 'Hollanda', lat: 52.5, lng: 5.7 },
    UA: { name: 'Ukrayna', lat: 49.0, lng: 32.0 },
    PL: { name: 'Polonya', lat: 52.0, lng: 19.0 },
    IR: { name: 'Iran', lat: 32.0, lng: 53.0 },
    SA: { name: 'Suudi Arabistan', lat: 24.0, lng: 45.0 },
    EG: { name: 'Misir', lat: 27.0, lng: 30.0 },
    ZA: { name: 'Guney Afrika', lat: -29.0, lng: 24.0 },
    NG: { name: 'Nijerya', lat: 10.0, lng: 8.0 },
    ID: { name: 'Endonezya', lat: -2.0, lng: 118.0 },
    PK: { name: 'Pakistan', lat: 30.0, lng: 70.0 },
    MX: { name: 'Meksika', lat: 23.0, lng: -102.0 },
    VN: { name: 'Vietnam', lat: 16.0, lng: 108.0 },
    TH: { name: 'Tayland', lat: 15.0, lng: 101.0 },
    NK: { name: 'Kuzey Kore', lat: 40.0, lng: 127.0 },
    AR: { name: 'Arjantin', lat: -34.0, lng: -64.0 },
};

// --- Severity Colors (tactical neon) ---
const SEVERITY_COLORS = {
    critical: { main: '#ef4444', glow: 'rgba(239, 68, 68, 0.6)', arc: '#ff2d5f' },
    high:     { main: '#ff6a00', glow: 'rgba(255, 106, 0, 0.5)', arc: '#ff8533' },
    medium:   { main: '#ffc400', glow: 'rgba(255, 196, 0, 0.4)', arc: '#ffd433' },
    low:      { main: '#10b981', glow: 'rgba(16, 185, 129, 0.3)', arc: '#33ff99' },
    info:     { main: '#38bdf8', glow: 'rgba(56, 189, 248, 0.3)', arc: '#33eeff' },
};

const getSeverityColor = (severity) => SEVERITY_COLORS[severity] || SEVERITY_COLORS.info;

// --- Custom globe material (Gotham dark tactical) ---
function applyGlobeMaterial(globeRef) {
    if (!globeRef.current) return;
    const scene = globeRef.current.scene();
    scene.traverse((obj) => {
        if (obj.type === 'Mesh' && obj.geometry?.type === 'SphereGeometry') {
            if (obj.__customApplied) return;
            obj.__customApplied = true;
        }
    });
}

// --- Orbital ring geometry ---
function addOrbitalRings(globeRef) {
    if (!globeRef.current) return;
    const scene = globeRef.current.scene();
    const GLOBE_RADIUS = 100;

    // Remove old rings
    scene.children.filter(c => c.__orbitalRing).forEach(c => scene.remove(c));

    const ringConfigs = [
        { radius: GLOBE_RADIUS * 1.15, color: 0x00e5ff, opacity: 0.06, tilt: [0.3, 0.1, 0] },
        { radius: GLOBE_RADIUS * 1.25, color: 0x00e5ff, opacity: 0.04, tilt: [-0.1, 0.4, 0.2] },
        { radius: GLOBE_RADIUS * 1.35, color: 0x00e676, opacity: 0.03, tilt: [0.5, -0.2, 0.1] },
    ];

    ringConfigs.forEach(({ radius, color, opacity, tilt }) => {
        const geometry = new RingGeometry(radius - 0.3, radius + 0.3, 128);
        const material = new MeshBasicMaterial({
            color,
            transparent: true,
            opacity,
            side: DoubleSide,
            depthWrite: false,
        });
        const ring = new Mesh(geometry, material);
        ring.rotation.set(...tilt);
        ring.__orbitalRing = true;
        scene.add(ring);
    });

    // Satellite orbit dots (moving along orbit paths)
    const SAT_CONFIGS = [
        { radius: GLOBE_RADIUS * 1.18, tilt: [0.4, 0.2, 0], count: 3, color: 0x00e5ff, speed: 0.005 },
        { radius: GLOBE_RADIUS * 1.28, tilt: [-0.15, 0.5, 0.2], count: 2, color: 0x00e676, speed: 0.003 },
        { radius: GLOBE_RADIUS * 1.38, tilt: [0.6, -0.3, 0.1], count: 2, color: 0xb388ff, speed: 0.004 },
    ];

    SAT_CONFIGS.forEach(({ radius, tilt, count, color, speed }) => {
        for (let s = 0; s < count; s++) {
            const geo = new SphereGeometry(1.2, 8, 8);
            const mat = new MeshBasicMaterial({
                color, transparent: true, opacity: 0.8, depthWrite: false,
            });
            const sat = new Mesh(geo, mat);
            sat.__satellite = true;
            sat.__orbitRadius = radius;
            sat.__orbitTilt = tilt;
            sat.__orbitSpeed = speed;
            sat.__orbitPhase = (s / count) * Math.PI * 2;
            scene.add(sat);
        }
    });
}

// --- Impact shockwave markers ---
function createShockwaveMarker(severity) {
    const colors = getSeverityColor(severity);
    return `
        <div style="position:relative;width:40px;height:40px;">
            <div style="
                position:absolute;inset:0;
                border:2px solid ${colors.main};
                border-radius:50%;
                animation: sonar-ring 2s ease-out infinite;
                opacity:0.6;
            "></div>
            <div style="
                position:absolute;inset:6px;
                border:1px solid ${colors.main};
                border-radius:50%;
                animation: sonar-ring 2s ease-out infinite 0.4s;
                opacity:0.4;
            "></div>
            <div style="
                position:absolute;top:50%;left:50%;
                width:8px;height:8px;
                transform:translate(-50%,-50%);
                background:${colors.main};
                border-radius:50%;
                box-shadow:0 0 12px ${colors.glow};
            "></div>
        </div>
    `;
}

// --- Main Component ---
export default function AdvancedGlobe3D({
    attacks = [],
    isLive = true,
    countriesGeoJSON = { features: [] },
    setHoveredAttack,
    setHoveredCountry,
    containerDimensions = { width: 800, height: 600 },
    onCountryClick,
}) {
    const globeRef = useRef();
    const [geoData, setGeoData] = useState(countriesGeoJSON);
    const animationPhaseRef = useRef(0);
    const lastCriticalIdRef = useRef(null);
    const orbitalApplied = useRef(false);

    // --- Load GeoJSON ---
    useEffect(() => {
        if (countriesGeoJSON.features.length > 0) {
            setGeoData(countriesGeoJSON);
            return;
        }
        fetch('https://raw.githubusercontent.com/vasturiano/react-globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson')
            .then(res => res.json())
            .then(setGeoData)
            .catch(() => console.warn('[Globe] GeoJSON load failed'));
    }, [countriesGeoJSON]);

    // --- Animation loop (ref-based, no re-renders) ---
    useEffect(() => {
        let raf;
        const tick = () => {
            animationPhaseRef.current = (animationPhaseRef.current + 0.02) % (Math.PI * 2);

            // Animate satellites
            if (globeRef.current) {
                const scene = globeRef.current.scene();
                scene.children.filter(c => c.__satellite).forEach(sat => {
                    sat.__orbitPhase += sat.__orbitSpeed;
                    const r = sat.__orbitRadius;
                    const t = sat.__orbitTilt;
                    const p = sat.__orbitPhase;
                    // Base position on XZ plane
                    let x = Math.cos(p) * r;
                    let y = 0;
                    let z = Math.sin(p) * r;
                    // Apply tilt rotation (simplified Euler)
                    const cx = Math.cos(t[0]), sx = Math.sin(t[0]);
                    const cy = Math.cos(t[1]), sy = Math.sin(t[1]);
                    const y1 = y * cx - z * sx;
                    const z1 = y * sx + z * cx;
                    const x2 = x * cy + z1 * sy;
                    const z2 = -x * sy + z1 * cy;
                    sat.position.set(x2, y1, z2);
                });
            }

            raf = requestAnimationFrame(tick);
        };
        tick();
        return () => cancelAnimationFrame(raf);
    }, []);

    // --- Camera + orbital rings + scene setup ---
    useEffect(() => {
        if (!globeRef.current) return;
        globeRef.current.pointOfView({ lat: 25, lng: 15, altitude: 2.2 }, 2000);
        const controls = globeRef.current.controls();
        controls.autoRotate = isLive;
        controls.autoRotateSpeed = 0.3;
        controls.enableDamping = true;
        controls.dampingFactor = 0.06;
        controls.minDistance = 120;
        controls.maxDistance = 600;

        // Apply custom material & orbital rings
        setTimeout(() => {
            applyGlobeMaterial(globeRef);
            if (!orbitalApplied.current) {
                addOrbitalRings(globeRef);
                orbitalApplied.current = true;
            }

            // Add ambient glow light + night-side hemisphere
            const scene = globeRef.current?.scene();
            if (scene && !scene.__glowLight) {
                const light = new PointLight(0x00e5ff, 0.4, 500);
                light.position.set(0, 0, 250);
                scene.add(light);

                // Directional light for day/night effect
                const sunLight = new DirectionalLight(0xfff4e6, 0.3);
                sunLight.position.set(200, 100, 300);
                scene.add(sunLight);

                // Subtle hemisphere light (top=sky, bottom=dark)
                const hemiLight = new HemisphereLight(0x001122, 0x000000, 0.15);
                scene.add(hemiLight);

                // Atmosphere outer glow ring (larger, fainter)
                const atmoGeo = new RingGeometry(100 * 1.005, 100 * 1.06, 128);
                const atmoMat = new MeshBasicMaterial({
                    color: 0x00e5ff,
                    transparent: true,
                    opacity: 0.015,
                    side: DoubleSide,
                    depthWrite: false,
                });
                const atmoRing = new Mesh(atmoGeo, atmoMat);
                atmoRing.lookAt(new Vector3(0, 0, 1));
                scene.add(atmoRing);

                scene.__glowLight = true;
            }
        }, 1000);
    }, [isLive]);

    // --- Camera: fly to critical attacks ---
    useEffect(() => {
        if (!attacks.length || !isLive || !globeRef.current) return;
        const critical = attacks.find(a => a.severity === 'critical');
        if (critical && critical.id !== lastCriticalIdRef.current && critical.source?.lat) {
            lastCriticalIdRef.current = critical.id;
            globeRef.current.pointOfView(
                { lat: critical.source.lat, lng: critical.source.lng, altitude: 1.8 },
                800
            );
        }
    }, [attacks, isLive]);

    // --- Attack counts per country ---
    const attackCounts = useMemo(() => {
        const counts = {};
        attacks.forEach(a => {
            const code = a.source?.country;
            if (code) counts[code] = (counts[code] || 0) + 1;
        });
        return counts;
    }, [attacks]);

    // --- ARC DATA: Laser-beam attack trails ---
    const arcsData = useMemo(() => {
        return attacks.slice(0, 80).map((attack) => {
            const colors = getSeverityColor(attack.severity);
            const isThreat = attack.ml_prediction?.is_threat;
            const conf = attack.ml_prediction?.confidence || 0;

            return {
                ...attack,
                startLat: attack.source?.lat || 0,
                startLng: attack.source?.lng || 0,
                endLat: attack.target?.lat || 39.0,
                endLng: attack.target?.lng || 35.0,
                color: [colors.arc, isThreat ? '#ef4444' : '#00ffaa'],
                stroke: attack.severity === 'critical' ? 1.5 : conf > 0.8 ? 1.0 : 0.5,
                dashLen: attack.severity === 'critical' ? 0.6 : 0.3,
                dashGap: attack.severity === 'critical' ? 0.08 : 0.15,
                animTime: attack.severity === 'critical' ? 600 : 1200,
            };
        }).filter(a => a.startLat !== 0 && a.endLat !== 0);
    }, [attacks]);

    // --- RINGS DATA: Pulsing target indicators ---
    const ringsData = useMemo(() => {
        const rings = [{
            lat: 39.0, lng: 35.0,
            maxR: 6, propagationSpeed: 2, repeatPeriod: 800,
            color: 'rgba(0, 229, 255, 0.5)',
        }];

        attacks.slice(0, 15).forEach(a => {
            if (!a.source?.lat) return;
            const c = getSeverityColor(a.severity);
            rings.push({
                lat: a.source.lat, lng: a.source.lng,
                maxR: a.severity === 'critical' ? 5 : 3,
                propagationSpeed: a.severity === 'critical' ? 4 : 2,
                repeatPeriod: a.severity === 'critical' ? 500 : 1000,
                color: c.glow,
            });
        });

        return rings;
    }, [attacks]);

    // --- HEXBIN DATA: 3D heatmap columns ---
    const hexbinData = useMemo(() => {
        return attacks.map(a => ({
            lat: a.source?.lat || 0,
            lng: a.source?.lng || 0,
            weight: a.severity === 'critical' ? 5 : a.severity === 'high' ? 3 : 1,
        })).filter(h => h.lat !== 0);
    }, [attacks]);

    // --- POINTS DATA: ML prediction glow points ---
    const pointsData = useMemo(() => {
        return attacks
            .filter(a => a.ml_prediction?.is_threat && a.source?.lat)
            .slice(0, 40)
            .map(a => ({
                lat: a.source.lat,
                lng: a.source.lng,
                size: 0.3 + (a.ml_prediction.confidence || 0) * 0.5,
                color: getSeverityColor(a.severity).main,
                label: `AI: ${((a.ml_prediction.confidence || 0) * 100).toFixed(0)}%`,
            }));
    }, [attacks]);

    // --- HTML MARKERS: Critical attack shockwaves ---
    const htmlMarkersData = useMemo(() => {
        return attacks
            .filter(a => a.severity === 'critical' && a.source?.lat)
            .slice(0, 8)
            .map(a => ({
                lat: a.source.lat,
                lng: a.source.lng,
                severity: a.severity,
            }));
    }, [attacks]);

    // --- LABEL DATA: Country attack labels ---
    const labelsData = useMemo(() => {
        return Object.entries(attackCounts)
            .filter(([, count]) => count >= 2)
            .map(([code, count]) => {
                const c = COUNTRIES_DATA[code];
                if (!c) return null;
                return {
                    lat: c.lat, lng: c.lng,
                    text: `${c.name} [${count}]`,
                    color: count > 15 ? '#ef4444' : count > 8 ? '#ff6a00' : count > 3 ? '#ffc400' : '#38bdf8',
                    size: Math.min(0.6 + count * 0.04, 1.4),
                };
            }).filter(Boolean);
    }, [attackCounts]);

    // --- Polygon colors (heatmap) ---
    const getPolygonColor = useCallback((d) => {
        const code = d.properties?.ISO_A2;
        if (code === 'TR') return 'rgba(0, 229, 255, 0.25)';
        const count = attackCounts[code] || 0;
        if (count > 15) return 'rgba(255, 0, 60, 0.45)';
        if (count > 8) return 'rgba(255, 106, 0, 0.35)';
        if (count > 3) return 'rgba(255, 196, 0, 0.25)';
        if (count > 0) return 'rgba(0, 229, 255, 0.1)';
        return 'rgba(8, 12, 24, 0.7)';
    }, [attackCounts]);

    const getPolygonStroke = useCallback((d) => {
        const code = d.properties?.ISO_A2;
        if (code === 'TR') return 'rgba(0, 229, 255, 0.6)';
        const count = attackCounts[code] || 0;
        if (count > 10) return 'rgba(255, 0, 60, 0.4)';
        return 'rgba(0, 229, 255, 0.08)';
    }, [attackCounts]);

    // --- Polygon altitude (country extrusion based on attacks) ---
    const getPolygonAltitude = useCallback((d) => {
        const code = d.properties?.ISO_A2;
        if (code === 'TR') return 0.025;
        const count = attackCounts[code] || 0;
        if (count > 15) return 0.04;
        if (count > 8) return 0.025;
        if (count > 3) return 0.015;
        return 0.005;
    }, [attackCounts]);

    // --- Polygon HTML tooltip ---
    const getPolygonLabel = useCallback((d) => {
        const code = d.properties?.ISO_A2;
        const name = d.properties?.NAME || 'Bilinmiyor';
        const count = attackCounts[code] || 0;
        const level = count > 15 ? 'KRITIK' : count > 8 ? 'YUKSEK' : count > 3 ? 'ORTA' : count > 0 ? 'DUSUK' : 'TEMIZ';
        const levelColor = count > 15 ? '#ef4444' : count > 8 ? '#ff6a00' : count > 3 ? '#ffc400' : count > 0 ? '#38bdf8' : '#10b981';

        return `
            <div style="
                background: linear-gradient(135deg, rgba(6,10,20,0.97), rgba(12,18,35,0.95));
                padding: 12px 16px;
                border-radius: 8px;
                border: 1px solid rgba(56,189,248,0.2);
                box-shadow: 0 8px 32px rgba(0,0,0,0.7), 0 0 20px rgba(56,189,248,0.08);
                font-family: 'JetBrains Mono', monospace;
                min-width: 200px;
                backdrop-filter: blur(20px);
            ">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="color:#c8d6e5; font-size:12px; font-weight:700; letter-spacing:1px; text-transform:uppercase;">${name}</span>
                    <span style="color:${levelColor}; font-size:8px; font-weight:700; padding:2px 6px; border:1px solid ${levelColor}33; border-radius:3px; letter-spacing:2px; background:${levelColor}15;">${level}</span>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                    <div style="background:rgba(56,189,248,0.04); padding:6px 8px; border-radius:4px; border:1px solid rgba(56,189,248,0.06);">
                        <div style="color:#4a5568; font-size:8px; text-transform:uppercase; letter-spacing:2px;">SALDIRI</div>
                        <div style="color:${levelColor}; font-size:20px; font-weight:800; font-family:'JetBrains Mono',monospace;">${count}</div>
                    </div>
                    <div style="background:rgba(56,189,248,0.04); padding:6px 8px; border-radius:4px; border:1px solid rgba(56,189,248,0.06);">
                        <div style="color:#4a5568; font-size:8px; text-transform:uppercase; letter-spacing:2px;">KOD</div>
                        <div style="color:#7a8ba0; font-size:20px; font-weight:800; font-family:'JetBrains Mono',monospace;">${code || '--'}</div>
                    </div>
                </div>
            </div>
        `;
    }, [attackCounts]);

    // --- Handle country click ---
    const handlePolygonClick = useCallback((d) => {
        const code = d.properties?.ISO_A2;
        const country = COUNTRIES_DATA[code];
        if (country && globeRef.current) {
            globeRef.current.pointOfView(
                { lat: country.lat, lng: country.lng, altitude: 1.5 },
                1000
            );
        }
        onCountryClick?.({ code, name: d.properties?.NAME, count: attackCounts[code] || 0 });
    }, [attackCounts, onCountryClick]);

    // --- RENDER ---
    return (
        <Globe
            ref={globeRef}
            width={containerDimensions.width}
            height={containerDimensions.height}

            globeImageUrl="//unpkg.com/three-globe/example/img/earth-night.jpg"
            bumpImageUrl="//unpkg.com/three-globe/example/img/earth-topology.png"
            backgroundImageUrl="//unpkg.com/three-globe/example/img/night-sky.png"

            atmosphereColor="#38bdf8"
            atmosphereAltitude={0.22}

            polygonsData={geoData.features}
            polygonAltitude={getPolygonAltitude}
            polygonCapColor={getPolygonColor}
            polygonSideColor={() => 'rgba(0, 229, 255, 0.06)'}
            polygonStrokeColor={getPolygonStroke}
            polygonLabel={getPolygonLabel}
            onPolygonClick={handlePolygonClick}
            onPolygonHover={setHoveredCountry}

            arcsData={arcsData}
            arcColor="color"
            arcDashLength="dashLen"
            arcDashGap="dashGap"
            arcDashAnimateTime="animTime"
            arcStroke="stroke"
            arcAltitudeAutoScale={0.45}
            onArcHover={setHoveredAttack}

            ringsData={ringsData}
            ringColor="color"
            ringMaxRadius="maxR"
            ringPropagationSpeed="propagationSpeed"
            ringRepeatPeriod="repeatPeriod"

            hexBinPointsData={hexbinData}
            hexBinPointWeight="weight"
            hexBinResolution={3}
            hexAltitude={d => {
                const base = d.sumWeight * 0.014;
                const breath = Math.sin(animationPhaseRef.current + d.center[0] * 0.1) * 0.005;
                return Math.max(0.008, base + breath);
            }}
            hexTopColor={d => {
                const intensity = Math.min(1, d.sumWeight / 15);
                return `rgba(255, ${Math.floor(40 + (1 - intensity) * 160)}, ${Math.floor(40 * (1 - intensity))}, ${0.8 + intensity * 0.2})`;
            }}
            hexSideColor={d => {
                const intensity = Math.min(1, d.sumWeight / 15);
                return `rgba(${Math.floor(200 + intensity * 55)}, ${Math.floor(30 + (1 - intensity) * 70)}, ${Math.floor(30 * (1 - intensity))}, ${0.5 + intensity * 0.3})`;
            }}
            hexBinMerge={true}

            pointsData={pointsData}
            pointAltitude={0.02}
            pointRadius="size"
            pointColor="color"
            pointLabel="label"

            htmlElementsData={htmlMarkersData}
            htmlLat="lat"
            htmlLng="lng"
            htmlAltitude={0.05}
            htmlElement={d => {
                const el = document.createElement('div');
                el.innerHTML = createShockwaveMarker(d.severity);
                el.style.pointerEvents = 'none';
                return el;
            }}

            labelsData={labelsData}
            labelLat="lat"
            labelLng="lng"
            labelText="text"
            labelColor="color"
            labelSize="size"
            labelDotRadius={0.3}
            labelAltitude={0.02}
            labelResolution={2}
        />
    );
}
