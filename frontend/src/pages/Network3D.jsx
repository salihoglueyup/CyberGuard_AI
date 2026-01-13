import React, { useRef, useEffect, useState } from 'react';
import * as THREE from 'three';
import api from '../services/api';

const Network3D = () => {
    const containerRef = useRef(null);
    const [nodes, setNodes] = useState([]);
    const [attacks, setAttacks] = useState([]);
    const [stats, setStats] = useState(null);

    useEffect(() => {
        loadData();
        const interval = setInterval(loadData, 5000);
        return () => clearInterval(interval);
    }, []);

    const loadData = async () => {
        try {
            const [nodesRes, attacksRes] = await Promise.all([
                api.get('/network/topology'),
                api.get('/attack-map/live?limit=20')
            ]);
            if (nodesRes.data.data?.nodes) setNodes(nodesRes.data.data.nodes);
            if (attacksRes.data.data?.attacks) setAttacks(attacksRes.data.data.attacks);
        } catch (error) {
            console.error('Error loading 3D data:', error);
        }
    };

    useEffect(() => {
        if (!containerRef.current) return;

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x0f172a);

        // Camera
        const camera = new THREE.PerspectiveCamera(
            75,
            containerRef.current.clientWidth / containerRef.current.clientHeight,
            0.1,
            1000
        );
        camera.position.z = 50;

        // Renderer
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        containerRef.current.appendChild(renderer.domElement);

        // Lights
        const ambientLight = new THREE.AmbientLight(0x404040, 2);
        scene.add(ambientLight);

        const pointLight = new THREE.PointLight(0x3b82f6, 1, 100);
        pointLight.position.set(10, 10, 10);
        scene.add(pointLight);

        // Create nodes
        const nodeObjects = [];
        const nodePositions = [];

        // Generate random node positions
        for (let i = 0; i < 50; i++) {
            const geometry = new THREE.SphereGeometry(0.5, 32, 32);
            const material = new THREE.MeshPhongMaterial({
                color: Math.random() > 0.8 ? 0xef4444 : 0x3b82f6,
                emissive: Math.random() > 0.8 ? 0xef4444 : 0x3b82f6,
                emissiveIntensity: 0.3
            });
            const sphere = new THREE.Mesh(geometry, material);

            sphere.position.x = (Math.random() - 0.5) * 60;
            sphere.position.y = (Math.random() - 0.5) * 40;
            sphere.position.z = (Math.random() - 0.5) * 30;

            nodeObjects.push(sphere);
            nodePositions.push(sphere.position.clone());
            scene.add(sphere);
        }

        // Create connections
        const lineMaterial = new THREE.LineBasicMaterial({
            color: 0x3b82f6,
            transparent: true,
            opacity: 0.3
        });

        for (let i = 0; i < nodeObjects.length; i++) {
            // Connect each node to 2-3 random others
            const connections = Math.floor(Math.random() * 2) + 1;
            for (let j = 0; j < connections; j++) {
                const targetIdx = Math.floor(Math.random() * nodeObjects.length);
                if (targetIdx !== i) {
                    const points = [
                        nodePositions[i],
                        nodePositions[targetIdx]
                    ];
                    const geometry = new THREE.BufferGeometry().setFromPoints(points);
                    const line = new THREE.Line(geometry, lineMaterial);
                    scene.add(line);
                }
            }
        }

        // Attack particles
        const attackParticles = [];
        const createAttackParticle = () => {
            const geometry = new THREE.SphereGeometry(0.2, 8, 8);
            const material = new THREE.MeshBasicMaterial({ color: 0xef4444 });
            const particle = new THREE.Mesh(geometry, material);

            const sourceIdx = Math.floor(Math.random() * nodeObjects.length);
            const targetIdx = Math.floor(Math.random() * nodeObjects.length);

            particle.userData = {
                source: nodePositions[sourceIdx].clone(),
                target: nodePositions[targetIdx].clone(),
                progress: 0
            };

            particle.position.copy(particle.userData.source);
            scene.add(particle);
            attackParticles.push(particle);
        };

        // Add initial attack particles
        for (let i = 0; i < 10; i++) {
            createAttackParticle();
        }

        // Mouse controls
        let mouseX = 0;
        let mouseY = 0;

        const onMouseMove = (event) => {
            mouseX = (event.clientX / window.innerWidth) * 2 - 1;
            mouseY = -(event.clientY / window.innerHeight) * 2 + 1;
        };
        window.addEventListener('mousemove', onMouseMove);

        // Animation
        const animate = () => {
            requestAnimationFrame(animate);

            // Rotate camera around center
            camera.position.x += (mouseX * 20 - camera.position.x) * 0.05;
            camera.position.y += (mouseY * 10 - camera.position.y) * 0.05;
            camera.lookAt(scene.position);

            // Animate attack particles
            attackParticles.forEach((particle, idx) => {
                particle.userData.progress += 0.02;

                if (particle.userData.progress >= 1) {
                    // Reset particle
                    particle.userData.progress = 0;
                    const sourceIdx = Math.floor(Math.random() * nodeObjects.length);
                    const targetIdx = Math.floor(Math.random() * nodeObjects.length);
                    particle.userData.source = nodePositions[sourceIdx].clone();
                    particle.userData.target = nodePositions[targetIdx].clone();
                }

                particle.position.lerpVectors(
                    particle.userData.source,
                    particle.userData.target,
                    particle.userData.progress
                );
            });

            // Pulse nodes
            nodeObjects.forEach((node, i) => {
                const scale = 1 + Math.sin(Date.now() * 0.003 + i) * 0.1;
                node.scale.set(scale, scale, scale);
            });

            renderer.render(scene, camera);
        };
        animate();

        // Cleanup
        return () => {
            window.removeEventListener('mousemove', onMouseMove);
            if (containerRef.current) {
                containerRef.current.removeChild(renderer.domElement);
            }
            renderer.dispose();
        };
    }, []);

    return (
        <div className="min-h-screen bg-gray-900 text-white">
            <div className="p-6">
                <h1 className="text-3xl font-bold text-cyan-400 mb-2">🌐 3D Network Visualization</h1>
                <p className="text-gray-400 mb-4">Real-time network topology and attack visualization</p>

                <div className="grid grid-cols-4 gap-4 mb-4">
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Active Nodes</p>
                        <p className="text-2xl font-bold text-cyan-400">50</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Connections</p>
                        <p className="text-2xl font-bold text-blue-400">127</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Active Attacks</p>
                        <p className="text-2xl font-bold text-red-400">{attacks.length}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Threat Level</p>
                        <p className="text-2xl font-bold text-orange-400">MEDIUM</p>
                    </div>
                </div>
            </div>

            <div
                ref={containerRef}
                className="w-full"
                style={{ height: 'calc(100vh - 200px)' }}
            />

            <div className="absolute bottom-4 left-4 bg-gray-800/80 rounded-lg p-4 border border-gray-700">
                <h3 className="text-sm font-semibold text-cyan-400 mb-2">Legend</h3>
                <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-blue-500"></div>
                        <span>Normal Node</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-3 h-3 rounded-full bg-red-500"></div>
                        <span>Compromised Node</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-0.5 bg-blue-500/50"></div>
                        <span>Connection</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse"></div>
                        <span>Attack Packet</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default Network3D;
