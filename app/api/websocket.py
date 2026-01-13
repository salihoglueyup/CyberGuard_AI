"""
WebSocket Server for Real-time Attack Broadcasting
Connects Globe3D, Attack Map and ML Predictions
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0
        self.max_connections = 100

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        if len(self.active_connections) >= self.max_connections:
            await websocket.close(code=1013)  # Try again later
            return False

        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        print(f"[WS] New connection. Total: {len(self.active_connections)}")
        return True

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"[WS] Disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_json = json.dumps(message, default=str)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_json)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_attack(
        self, attack: Dict[str, Any], ml_prediction: Dict[str, Any] = None
    ):
        """Broadcast an attack event with ML prediction"""
        message = {
            "type": "attack",
            "data": {
                **attack,
                "ml_prediction": ml_prediction
                or {
                    "is_threat": True,
                    "confidence": 0.85,
                    "suggested_action": "monitor",
                },
            },
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(message)

    async def broadcast_stats(self, stats: Dict[str, Any]):
        """Broadcast attack statistics"""
        message = {
            "type": "stats",
            "data": stats,
            "timestamp": datetime.now().isoformat(),
        }
        await self.broadcast(message)

    def get_status(self) -> Dict[str, Any]:
        """Get WebSocket server status"""
        return {
            "active_connections": len(self.active_connections),
            "total_connections": self.connection_count,
            "max_connections": self.max_connections,
            "status": "running",
        }


# Global connection manager
manager = ConnectionManager()


async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint handler"""
    connected = await manager.connect(websocket)
    if not connected:
        return

    try:
        # Send welcome message
        await websocket.send_json(
            {
                "type": "connected",
                "message": "Connected to CyberGuard AI WebSocket",
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (ping/pong or commands)
                data = await asyncio.wait_for(
                    websocket.receive_text(), timeout=60.0  # 60 second timeout
                )

                # Handle client commands
                try:
                    message = json.loads(data)
                    if message.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif message.get("type") == "subscribe":
                        # Could implement channel subscriptions here
                        pass
                except json.JSONDecodeError:
                    pass

            except asyncio.TimeoutError:
                # Send heartbeat
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS] Error: {e}")
    finally:
        manager.disconnect(websocket)


def get_manager() -> ConnectionManager:
    """Get the global connection manager"""
    return manager
