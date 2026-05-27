/**
 * WebSocket Service - CyberGuard AI
 * 
 * DEPRECATED: Bu dosya geriye uyumluluk için tutulmaktadır.
 * Yeni kod hooks/useWebSocket.js kullanmalıdır.
 * 
 * Re-export from the unified WebSocket hook.
 */
export { useWebSocket } from '../hooks/useWebSocket';

// Event types (backward compatibility)
export const WS_EVENTS = {
    CONNECTION: 'connection',
    ERROR: 'error',
    NEW_ATTACK: 'new_attack',
    ATTACK_BLOCKED: 'attack_blocked',
    IDS_STATUS: 'ids_status',
    IDS_ALERT: 'ids_alert',
    TRAINING_PROGRESS: 'training_progress',
    TRAINING_COMPLETE: 'training_complete',
    STATS_UPDATE: 'stats_update',
    LOG_ENTRY: 'log_entry',
};
