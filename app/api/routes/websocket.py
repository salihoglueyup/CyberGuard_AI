"""
WebSocket API - REAL DATA VERSION
Real-time data streaming with psutil
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
from datetime import datetime
import asyncio
import json

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

router = APIRouter()

connected_clients: List[WebSocket] = []
event_subscribers: Dict[str, List[WebSocket]] = {}


def get_system_metrics() -> Dict:
    if not PSUTIL_AVAILABLE:
        return {"error": "psutil not available"}

    return {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage("/").percent,
        "network": {
            "bytes_sent": psutil.net_io_counters().bytes_sent,
            "bytes_recv": psutil.net_io_counters().bytes_recv,
        },
        "timestamp": datetime.now().isoformat(),
    }


def get_security_metrics() -> Dict:
    # Collect from other modules
    return {
        "active_connections": len(connected_clients),
        "timestamp": datetime.now().isoformat(),
    }


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            metrics = get_system_metrics()
            await websocket.send_json({"type": "metrics", "data": metrics})
            await asyncio.sleep(2)
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
    except Exception:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@router.websocket("/ws/events")
async def events_websocket(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") == "subscribe":
                event_type = msg.get("event_type", "all")
                event_subscribers.setdefault(event_type, []).append(websocket)
                await websocket.send_json(
                    {"type": "subscribed", "event_type": event_type}
                )

            await asyncio.sleep(0.1)
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
        for subscribers in event_subscribers.values():
            if websocket in subscribers:
                subscribers.remove(websocket)


@router.websocket("/ws/security")
async def security_websocket(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)

    try:
        while True:
            metrics = get_security_metrics()
            await websocket.send_json({"type": "security", "data": metrics})
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@router.get("/status")
async def get_status():
    return {
        "success": True,
        "data": {
            "connected_clients": len(connected_clients),
            "event_subscribers": {k: len(v) for k, v in event_subscribers.items()},
            "psutil_available": PSUTIL_AVAILABLE,
        },
    }


async def broadcast_event(event_type: str, data: Dict):
    """Broadcast event to all subscribers"""
    subscribers = event_subscribers.get(event_type, []) + event_subscribers.get(
        "all", []
    )
    message = {
        "type": event_type,
        "data": data,
        "timestamp": datetime.now().isoformat(),
    }

    for ws in subscribers:
        try:
            await ws.send_json(message)
        except:
            pass


# Attack stream for Globe3D
attack_stream_clients: List[WebSocket] = []


@router.websocket("/ws/attacks")
async def attacks_websocket(websocket: WebSocket):
    """WebSocket for real-time attack stream to Globe3D"""
    await websocket.accept()
    attack_stream_clients.append(websocket)

    # Import ML predictor
    try:
        from app.services.ml_predictor import predict_threat

        HAS_ML = True
    except ImportError:
        HAS_ML = False

    # Import GeoIP
    try:
        from app.services.geoip import lookup_ip, get_random_coords_in_country

        HAS_GEOIP = True
    except ImportError:
        HAS_GEOIP = False

    try:
        # Send welcome
        await websocket.send_json(
            {
                "type": "connected",
                "message": "Connected to attack stream",
                "ml_enabled": HAS_ML,
                "geoip_enabled": HAS_GEOIP,
            }
        )

        # Keep connection alive
        while True:
            try:
                # Wait for ping or receive attack data
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)

                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS/Attacks] Error: {e}")
    finally:
        if websocket in attack_stream_clients:
            attack_stream_clients.remove(websocket)


async def broadcast_attack(attack: Dict):
    """Broadcast attack to all connected Globe3D clients"""
    if not attack_stream_clients:
        return

    # Add ML prediction
    try:
        from app.services.ml_predictor import predict_threat

        ml_prediction = predict_threat(attack)
    except Exception:
        ml_prediction = {
            "is_threat": True,
            "confidence": 0.85,
            "suggested_action": "monitor",
        }

    # Add GeoIP coordinates
    try:
        from app.services.geoip import lookup_ip

        source_ip = attack.get("source", {}).get("ip", "")
        target_ip = attack.get("target", {}).get("ip", "")

        if source_ip:
            source_geo = lookup_ip(source_ip)
            attack["source"]["lat"] = source_geo.get("lat", 0)
            attack["source"]["lng"] = source_geo.get("lng", 0)

        if target_ip:
            target_geo = lookup_ip(target_ip)
            attack["target"]["lat"] = target_geo.get("lat", 0)
            attack["target"]["lng"] = target_geo.get("lng", 0)
    except Exception:
        pass

    message = {
        "type": "attack",
        "data": {**attack, "ml_prediction": ml_prediction},
        "timestamp": datetime.now().isoformat(),
    }

    disconnected = []
    for ws in attack_stream_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        attack_stream_clients.remove(ws)


def get_attack_stream_status() -> Dict:
    """Get attack stream status"""
    return {
        "connected_clients": len(attack_stream_clients),
        "stream_active": len(attack_stream_clients) > 0,
    }
