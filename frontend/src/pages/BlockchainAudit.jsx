import React, { useState, useEffect } from 'react';
import api from '../services/api';

const BlockchainAudit = () => {
    const [chain, setChain] = useState([]);
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [searchQuery, setSearchQuery] = useState('');
    const [searchResults, setSearchResults] = useState(null);

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
            alert(result.is_valid ? '✅ Block doğrulandı!' : '❌ Block geçersiz!');
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
            <div className="flex items-center justify-center h-screen bg-gray-900">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-yellow-500 mx-auto"></div>
                    <p className="mt-4 text-yellow-400">Loading Blockchain...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-900 text-white p-6">
            <div className="mb-6">
                <h1 className="text-3xl font-bold text-yellow-400">⛓️ Blockchain Audit Trail</h1>
                <p className="text-gray-400">Değiştirilemez güvenlik log kaydı</p>
            </div>

            {/* Stats */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-6">
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Toplam Block</p>
                        <p className="text-2xl font-bold text-yellow-400">{stats.total_blocks?.toLocaleString()}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">24s Block</p>
                        <p className="text-2xl font-bold text-green-400">{stats.blocks_24h}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Zincir Durumu</p>
                        <p className="text-xl font-bold text-green-400">✓ {stats.chain_integrity}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Son Block</p>
                        <p className="text-sm font-bold text-cyan-400">{new Date(stats.last_block_time).toLocaleTimeString()}</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Depolama</p>
                        <p className="text-2xl font-bold text-purple-400">{stats.storage_size_mb} MB</p>
                    </div>
                    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <p className="text-gray-400 text-sm">Doğrulama Oranı</p>
                        <p className="text-2xl font-bold text-green-400">{stats.verification_success_rate}%</p>
                    </div>
                </div>
            )}

            {/* Search */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700 mb-6">
                <div className="flex gap-4">
                    <input
                        type="text"
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        placeholder="Event tipi, kullanıcı ID veya kaynak ara..."
                        className="flex-1 px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg focus:border-yellow-500 focus:outline-none"
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
                <div className="bg-gray-800 rounded-lg p-4 border border-yellow-500/30 mb-6">
                    <h3 className="text-lg font-semibold text-yellow-400 mb-4">Arama Sonuçları ({searchResults.length})</h3>
                    <div className="space-y-2">
                        {searchResults.map((r) => (
                            <div key={r.block_id} className="p-3 bg-gray-700/50 rounded-lg flex justify-between">
                                <div>
                                    <span className="mr-2">{getEventIcon(r.event_type)}</span>
                                    <span>{r.event_type}</span>
                                    <span className="text-gray-400 ml-2">• {r.action}</span>
                                </div>
                                <span className="text-xs text-gray-500">{new Date(r.timestamp).toLocaleString()}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Blockchain */}
            <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
                <h3 className="text-lg font-semibold text-yellow-400 mb-4">📦 Son Bloklar</h3>

                <div className="space-y-3">
                    {chain?.length > 0 ? chain.map((block, index) => (
                        <div key={block.block_id} className="relative">
                            {/* Connection line */}
                            {index < chain.length - 1 && (
                                <div className="absolute left-6 top-16 w-0.5 h-8 bg-yellow-600/50"></div>
                            )}

                            <div className="p-4 bg-gray-700/50 rounded-lg border border-yellow-600/30">
                                <div className="flex justify-between items-start">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 bg-yellow-600/20 rounded-lg flex items-center justify-center text-2xl">
                                            {getEventIcon(block.event?.type)}
                                        </div>
                                        <div>
                                            <p className="font-medium">Block #{block.block_id}</p>
                                            <p className="text-sm text-gray-400">
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
                                    <div className="bg-gray-800/50 p-2 rounded font-mono">
                                        <span className="text-gray-500">Hash: </span>
                                        <span className="text-yellow-400">{block.block_hash?.slice(0, 24)}...</span>
                                    </div>
                                    <div className="bg-gray-800/50 p-2 rounded font-mono">
                                        <span className="text-gray-500">Prev: </span>
                                        <span className="text-gray-400">{block.previous_hash?.slice(0, 24)}...</span>
                                    </div>
                                </div>

                                <p className="text-xs text-gray-500 mt-2">{new Date(block.timestamp).toLocaleString()}</p>
                            </div>
                        </div>
                    )) : (
                        <div className="text-center py-8 text-gray-500">
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
