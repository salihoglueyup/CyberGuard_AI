"""
Attack Surface Management API - REAL DATA VERSION
External asset discovery and risk assessment
"""

import ipaddress
import json
import logging
import os
import socket
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.routes.auth import require_auth

logger = logging.getLogger(__name__)

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

router = APIRouter(dependencies=[Depends(require_auth)])

ALLOWED_SCAN_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
]


def is_safe_scan_target(host: str) -> bool:
    """Only allow scanning private/local IPs"""
    try:
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        return any(addr in net for net in ALLOWED_SCAN_NETWORKS)
    except (ValueError, socket.gaierror):
        return False

from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)
data_dir = os.path.join(project_root, "data")
assets_file = os.path.join(data_dir, "attack_surface.json")

ASSETS = {"domains": [], "ips": [], "services": [], "findings": []}


class AssetAdd(BaseModel):
    type: str  # domain, ip, service
    value: str
    tags: list[str] = []


def load_assets():
    global ASSETS
    if os.path.exists(assets_file):
        try:
            with open(assets_file) as f:
                ASSETS.update(json.load(f))
        except Exception as e:
            print(f"[attack_surface] Failed to load assets: {e}")


def save_assets():
    os.makedirs(os.path.dirname(assets_file), exist_ok=True)
    with open(assets_file, "w") as f:
        json.dump(ASSETS, f, indent=2, default=str)


def scan_port(host: str, port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(1)
        result = sock.connect_ex((host, port))
        return result == 0
    except Exception:
        return False
    finally:
        sock.close()


def discover_local_services() -> list[dict]:
    services = []
    if PSUTIL_AVAILABLE:
        for conn in psutil.net_connections(kind="inet"):
            if conn.status == "LISTEN":
                services.append(
                    {
                        "port": conn.laddr.port,
                        "address": conn.laddr.ip,
                        "pid": conn.pid,
                        "status": "listening",
                    }
                )
    return services


def check_domain_dns(domain: str) -> dict:
    try:
        ip = socket.gethostbyname(domain)
        return {"domain": domain, "ip": ip, "resolved": True}
    except Exception:
        return {"domain": domain, "resolved": False}


load_assets()


@router.get("/status")
async def get_status():
    services = discover_local_services()
    return {
        "success": True,
        "data": {
            "status": "active",
            "domains": len(ASSETS.get("domains", [])),
            "ips": len(ASSETS.get("ips", [])),
            "services": len(services),
            "findings": len(ASSETS.get("findings", [])),
        },
    }


@router.get("/discover")
async def discover_assets():
    """Discover local attack surface"""
    services = discover_local_services()
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "127.0.0.1"

    common_ports = [21, 22, 23, 25, 80, 443, 3306, 5432, 8080, 8443]
    open_ports = []
    for port in common_ports:
        if scan_port(local_ip, port):
            open_ports.append(port)

    return {
        "success": True,
        "data": {
            "hostname": hostname,
            "local_ip": local_ip,
            "listening_services": services,
            "open_ports": open_ports,
            "timestamp": datetime.now().isoformat(),
        },
    }


@router.post("/assets")
async def add_asset(asset: AssetAdd):
    new_asset = {
        "id": f"AST-{len(ASSETS.get(asset.type + 's', [])) + 1}",
        "type": asset.type,
        "value": asset.value,
        "tags": asset.tags,
        "added_at": datetime.now().isoformat(),
        "status": "active",
    }
    ASSETS.setdefault(asset.type + "s", []).append(new_asset)
    save_assets()
    return {"success": True, "data": new_asset}


@router.get("/assets")
async def get_assets(asset_type: str | None = None):
    if asset_type:
        return {"success": True, "data": ASSETS.get(asset_type + "s", [])}
    return {"success": True, "data": ASSETS}


@router.post("/scan/{asset_id}")
async def scan_asset(asset_id: str):
    for asset_type in ["domains", "ips", "services"]:
        for asset in ASSETS.get(asset_type, []):
            if asset.get("id") == asset_id:
                findings = []
                value = asset.get("value", "")

                if asset.get("type") == "domain":
                    dns = check_domain_dns(value)
                    if dns["resolved"] and is_safe_scan_target(dns["ip"]):
                        for port in [80, 443, 22, 21]:
                            if scan_port(dns["ip"], port):
                                findings.append({"port": port, "status": "open"})
                elif asset.get("type") == "ip":
                    if not is_safe_scan_target(value):
                        raise HTTPException(400, "Yalnızca yerel/özel ağlar taranabilir")
                    for port in [80, 443, 22, 21, 3306, 5432]:
                        if scan_port(value, port):
                            findings.append({"port": port, "status": "open"})

                asset["last_scan"] = datetime.now().isoformat()
                asset["findings"] = findings
                save_assets()
                return {"success": True, "data": {"asset": asset, "findings": findings}}

    raise HTTPException(status_code=404)


@router.get("/findings")
async def get_findings():
    all_findings = ASSETS.get("findings", [])
    for asset_type in ["domains", "ips", "services"]:
        for asset in ASSETS.get(asset_type, []):
            for f in asset.get("findings", []):
                all_findings.append({**f, "asset": asset.get("value")})
    return {"success": True, "data": {"findings": all_findings}}


@router.get("/risk-score")
async def get_risk_score():
    services = discover_local_services()
    high_risk_ports = [21, 23, 3389, 445]
    risk = 0
    for s in services:
        if s.get("port") in high_risk_ports:
            risk += 20
    return {
        "success": True,
        "data": {"risk_score": min(100, risk), "services_count": len(services)},
    }
