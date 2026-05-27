import React, { useState, useEffect } from 'react';
import api from '../../services/api';
import { useToast } from '../../components/ui/Toast';

const BlockchainAudit = () => {
    const [chain, setChain] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState(null);
    const toast = useToast();

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            const [chainRes, statsRes] = await Promise.all([
                api.get('/blockchain/chain?limit=20'),
                api.get('/blockchain/stats')
            ]);
            setChain(chainRes.data.data?.chain || []);
            setStats(statsRes.data.data);
        } catch (error) {
            console.error('Error loading blockchain data:', error);
            setChain([]);
        } finally {
            setLoading(false);
        }
    };

    const handleSearch = async () => {
        if (!searchQuery) return;
        try {
            const response = await api.get(`/blockchain/search?event_type=${searchQuery}`);
            setSearchResults(response.data.data.results);
        } catch (error) {
            console.error('Error searching:', error);
        }
    };

    const verifyBlock = async (blockId) => {
        try {
            const response = await api.get(`/blockchain/verify/${blockId}`);
            const result = response.data.data;
            toast[result.is_valid ? 'success' : 'error'](result.is_valid ? 'Block doğrulandı!' : 'Block geçersiz!');
        } catch (error) {
            console.error('Error verifying:', error);
        }
    };

    const getEventIcon = (type) => {
        switch (type) {
            case 'login': return '🔐';
            case 'data_access': return '📂';
            case 'config_change': return '⚙️';
            case 'attack_detected': return '🚨';
            case 'report_generated': return '📋';
            default: return '📝';
        }
    };

    if (loading) {
        return (
            <div className="flex items-center justify-center h-screen bg-[var(--hud-bg)]">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-yellow-500 mx-auto"></div>
                    <p className="mt-4 text-[var(--hud-amber)]">Loading Blockchain...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] p-6">
            <div className="mb-6">
                <h1 className="text-xl font-semibold text-[var(--hud-text)]">Blockchain Audit Trail</h1>
                <p className="text-[var(--hud-text-muted)] text-xs tracking-wide mt-1">Değiştirilemez güvenlik log kaydı</p>
            </div>

            {/* Stats */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Toplam Block</p>
                        <p className="text-2xl font-bold text-[var(--hud-amber)]">{stats.total_blocks?.toLocaleString()}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">24s Block</p>
                        <p className="text-2xl font-bold text-green-400">{stats.blocks_24h}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Zincir Durumu</p>
                        <p className="text-xl font-bold text-green-400">✓ {stats.chain_integrity}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Son Block</p>
                        <p className="text-sm font-bold text-[var(--hud-cyan)]">{new Date(stats.last_block_time).toLocaleTimeString()}</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Depolama</p>
                        <p className="text-2xl font-bold text-purple-400">{stats.storage_size_mb} MB</p>
                    </div>
                    <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                        <p className="text-[var(--hud-text-muted)] text-sm">Doğrulama Oranı</p>
                        <p className="text-2xl font-bold text-green-400">{stats.verification_success_rate}%</p>
                    </div>
                </div>
            )}

            {/* Search */}
            <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)] mb-6">
                <div className="flex gap-4">
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Event tipi, kullanıcı ID veya kaynak ara..."
                        className="flex-1 px-4 py-2 bg-[var(--hud-panel)] border border-[var(--hud-border)] rounded-lg focus:border-yellow-500 focus:outline-none"
                    />
                    <button
                        onClick={handleSearch}
                        className="px-6 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg font-medium"
                    >
                        🔍 Ara
                    </button>
                </div>
            </div>

            {/* Search Results */}
            {searchResults && (
                <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-yellow-500/30 mb-6">
                    <h3 className="text-lg font-semibold text-[var(--hud-amber)] mb-4">Arama Sonuçları ({searchResults.length})</h3>
                    <div className="space-y-2">
                        {searchResults.map((r) => (
                            <div key={r.block_id} className="p-3 bg-[var(--hud-panel)]/50 rounded-lg flex justify-between">
                                <div>
                                    <span className="mr-2">{getEventIcon(r.event_type)}</span>
                                    <span>{r.event_type}</span>
                                    <span className="text-[var(--hud-text-muted)] ml-2">• {r.action}</span>
                                </div>
                                <span className="text-xs text-[var(--hud-text-dim)]">{new Date(r.timestamp).toLocaleString()}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Blockchain */}
            <div className="bg-[var(--hud-surface)] rounded-lg p-4 border border-[var(--hud-border)]">
                <h3 className="text-lg font-semibold text-[var(--hud-amber)] mb-4">📦 Son Bloklar</h3>

                <div className="space-y-3">
                    {chain?.length > 0 ? chain.map((block, index) => (
                        <div key={block.block_id} className="relative">
                            {/* Connection line */}
                            {index < chain.length - 1 && (
                                <div className="absolute left-6 top-16 w-0.5 h-8 bg-yellow-600/50"></div>
                            )}

                            <div className="p-4 bg-[var(--hud-panel)]/50 rounded-lg border border-yellow-600/30">
                                <div className="flex justify-between items-start">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 bg-yellow-600/20 rounded-lg flex items-center justify-center text-2xl">
                                            {getEventIcon(block.event?.type)}
                                        </div>
                                        <div>
                                            <p className="font-medium">Block #{block.block_id}</p>
                                            <p className="text-sm text-[var(--hud-text-muted)]">
                                                {block.event?.type} • {block.event?.action} • {block.event?.user_id}
                                            </p>
                                        </div>
                                    </div>
                                    <button
                                        onClick={() => verifyBlock(block.block_id)}
                                        className="px-3 py-1 bg-yellow-600/30 hover:bg-yellow-600/50 rounded text-sm"
                                    >
                                        ✓ Doğrula
                                    </button>
                                </div>

                                <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                                    <div className="bg-[var(--hud-surface)]/50 p-2 rounded font-mono">
                                        <span className="text-[var(--hud-text-dim)]">Hash: </span>
                                        <span className="text-[var(--hud-amber)]">{block.block_hash?.slice(0, 24)}...</span>
                                    </div>
                                    <div className="bg-[var(--hud-surface)]/50 p-2 rounded font-mono">
                                        <span className="text-[var(--hud-text-dim)]">Prev: </span>
                                        <span className="text-[var(--hud-text-muted)]">{block.previous_hash?.slice(0, 24)}...</span>
                                    </div>
                                </div>

                                <p className="text-xs text-[var(--hud-text-dim)] mt-2">{new Date(block.timestamp).toLocaleString()}</p>
                            </div>
                        </div>
                    )) : (
                        <div className="text-center py-8 text-[var(--hud-text-dim)]">
                            <p>Blok bulunamadı</p>
                            <p className="text-sm">Blockchain henüz başlatılmamış</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default BlockchainAudit;
