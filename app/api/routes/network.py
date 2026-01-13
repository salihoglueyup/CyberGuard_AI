"""
Network API - REAL DATA VERSION
Uses psutil for actual network metrics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import socket
import platform
from datetime import datetime

# Try to import psutil
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

router = APIRouter()


class NetworkScanRequest(BaseModel):
    target: str
    scan_type: str = "quick"  # quick, full, stealth


def get_real_network_interfaces() -> List[Dict]:
    """Get real network interface information (frontend-compatible format)"""
    interfaces = []

    if not PSUTIL_AVAILABLE:
        return interfaces

    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    io_counters = psutil.net_io_counters(pernic=True)

    for iface, addr_list in addrs.items():
        # Get IPv4 address
        ipv4_addr = None
        for addr in addr_list:
            if addr.family == socket.AF_INET:
                ipv4_addr = addr.address
                break

        iface_info = {
            "name": iface,
            "ip": ipv4_addr,  # Frontend expects 'ip' field
            "status": "down",
            "speed": "0 Mbps",
            "is_up": False,
            "mtu": 0,
            "bytes_sent": 0,
            "bytes_recv": 0,
            "packets_sent": 0,
            "packets_recv": 0,
            "addresses": [],
        }

        # Add addresses
        for addr in addr_list:
            if addr.family == socket.AF_INET:
                iface_info["addresses"].append(
                    {
                        "type": "IPv4",
                        "address": addr.address,
                        "netmask": addr.netmask,
                        "broadcast": addr.broadcast,
                    }
                )
            elif addr.family == socket.AF_INET6:
                iface_info["addresses"].append(
                    {
                        "type": "IPv6",
                        "address": addr.address,
                    }
                )

        # Add stats
        if iface in stats:
            s = stats[iface]
            iface_info["is_up"] = s.isup
            iface_info["status"] = (
                "up" if s.isup else "down"
            )  # Frontend expects 'up'/'down'
            iface_info["speed"] = f"{s.speed} Mbps" if s.speed else "Unknown"
            iface_info["mtu"] = s.mtu

        # Add IO counters
        if iface in io_counters:
            io = io_counters[iface]
            iface_info["bytes_sent"] = io.bytes_sent
            iface_info["bytes_recv"] = io.bytes_recv
            iface_info["packets_sent"] = io.packets_sent
            iface_info["packets_recv"] = io.packets_recv

        interfaces.append(iface_info)

    return interfaces


def get_active_connections() -> List[Dict]:
    """Get real active network connections"""
    connections = []

    if not PSUTIL_AVAILABLE:
        return connections

    for conn in psutil.net_connections(kind="inet"):
        try:
            conn_info = {
                "family": "IPv4" if conn.family == socket.AF_INET else "IPv6",
                "type": "TCP" if conn.type == socket.SOCK_STREAM else "UDP",
                "local_address": (
                    f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
                ),
                "remote_address": (
                    f"{conn.raddr.ip}:{conn.raddr.port}" if conn.raddr else ""
                ),
                "status": conn.status,
                "pid": conn.pid,
            }

            # Get process name if available
            if conn.pid:
                try:
                    proc = psutil.Process(conn.pid)
                    conn_info["process"] = proc.name()
                except:
                    conn_info["process"] = "Unknown"
            else:
                conn_info["process"] = None

            connections.append(conn_info)
        except:
            pass

    return connections


def get_network_io_total() -> Dict:
    """Get total network IO statistics"""
    if not PSUTIL_AVAILABLE:
        return {}

    io = psutil.net_io_counters()
    return {
        "bytes_sent": io.bytes_sent,
        "bytes_recv": io.bytes_recv,
        "packets_sent": io.packets_sent,
        "packets_recv": io.packets_recv,
        "errin": io.errin,
        "errout": io.errout,
        "dropin": io.dropin,
        "dropout": io.dropout,
    }


def check_port(host: str, port: int, timeout: float = 1.0) -> Dict:
    """Check if a port is open"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()

        return {
            "port": port,
            "status": "open" if result == 0 else "closed",
            "service": get_service_name(port),
        }
    except Exception as e:
        return {"port": port, "status": "error", "error": str(e)}


def get_service_name(port: int) -> str:
    """Get common service name for port"""
    services = {
        21: "FTP",
        22: "SSH",
        23: "Telnet",
        25: "SMTP",
        53: "DNS",
        80: "HTTP",
        110: "POP3",
        143: "IMAP",
        443: "HTTPS",
        445: "SMB",
        3306: "MySQL",
        5432: "PostgreSQL",
        3389: "RDP",
        5000: "Flask",
        5173: "Vite",
        8000: "HTTP-Alt",
        8080: "HTTP-Proxy",
        27017: "MongoDB",
        6379: "Redis",
    }
    return services.get(port, "Unknown")


@router.get("/status")
async def get_network_status():
    """Get overall network status with system resources"""
    interfaces = get_real_network_interfaces()
    io_total = get_network_io_total()

    active_interfaces = [i for i in interfaces if i.get("is_up")]

    # Get CPU and memory for frontend
    cpu_percent = 0
    memory_percent = 0
    if PSUTIL_AVAILABLE:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

    return {
        "success": True,
        "data": {
            "status": "online" if active_interfaces else "offline",
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "interfaces_total": len(interfaces),
            "interfaces_active": len(active_interfaces),
            "active_connections": len(get_active_connections()),
            # Frontend expects these fields:
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "bytes_sent": io_total.get("bytes_sent", 0),
            "bytes_recv": io_total.get("bytes_recv", 0),
            "packets_sent": io_total.get("packets_sent", 0),
            "packets_recv": io_total.get("packets_recv", 0),
            "io_stats": io_total,
            "psutil_available": PSUTIL_AVAILABLE,
            "timestamp": datetime.now().isoformat(),
        },
    }


@router.get("/interfaces")
async def get_interfaces():
    """Get all network interfaces"""
    interfaces = get_real_network_interfaces()

    return {
        "success": True,
        "data": {
            "interfaces": interfaces,
            "total": len(interfaces),
            "active": len([i for i in interfaces if i.get("is_up")]),
        },
    }


@router.get("/connections")
async def get_connections(limit: int = 100):
    """Get active network connections"""
    connections = get_active_connections()

    # Filter and limit
    filtered = connections[:limit]

    # Group by status
    by_status = {}
    for conn in connections:
        status = conn.get("status", "UNKNOWN")
        by_status[status] = by_status.get(status, 0) + 1

    return {
        "success": True,
        "data": {
            "connections": filtered,
            "total": len(connections),
            "by_status": by_status,
            "established": by_status.get("ESTABLISHED", 0),
            "listening": by_status.get("LISTEN", 0),
        },
    }


@router.get("/topology")
async def get_network_topology():
    """Get network topology (nodes and edges)"""
    nodes = []
    edges = []

    # Add this machine as central node
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    nodes.append(
        {
            "id": "local",
            "type": "workstation",
            "label": hostname,
            "ip": local_ip,
        }
    )

    # Add gateway
    gateway_ip = ".".join(local_ip.split(".")[:3]) + ".1"
    nodes.append(
        {
            "id": "gateway",
            "type": "router",
            "label": "Gateway",
            "ip": gateway_ip,
        }
    )
    edges.append({"from": "gateway", "to": "local", "status": "active"})

    # Add connected hosts from active connections
    if PSUTIL_AVAILABLE:
        seen_ips = set([local_ip, gateway_ip])
        conn_count = 0

        for conn in psutil.net_connections(kind="inet"):
            if conn.raddr and conn.status == "ESTABLISHED":
                remote_ip = conn.raddr.ip
                if remote_ip not in seen_ips and not remote_ip.startswith("127."):
                    seen_ips.add(remote_ip)
                    conn_count += 1

                    node_id = f"remote_{conn_count}"
                    nodes.append(
                        {
                            "id": node_id,
                            "type": "server",
                            "label": f"Remote:{conn.raddr.port}",
                            "ip": remote_ip,
                        }
                    )
                    edges.append({"from": "local", "to": node_id, "status": "active"})

                    if conn_count >= 20:  # Limit
                        break

    return {
        "success": True,
        "data": {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_devices": len(nodes),
                "active_connections": len(
                    [e for e in edges if e["status"] == "active"]
                ),
            },
        },
    }


@router.post("/scan")
async def scan_network(request: NetworkScanRequest):
    """Scan a target for open ports"""
    target = request.target

    # Validate target
    try:
        socket.gethostbyname(target)
    except socket.gaierror:
        raise HTTPException(status_code=400, detail="Invalid target hostname or IP")

    # Common ports to scan
    if request.scan_type == "quick":
        ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 3306, 3389, 5432, 8080]
    elif request.scan_type == "full":
        ports = list(range(1, 1025))
    else:  # stealth
        ports = [22, 80, 443, 3389]

    results = []
    open_ports = []

    for port in ports:
        result = check_port(target, port, timeout=0.5)
        results.append(result)
        if result["status"] == "open":
            open_ports.append(result)

    return {
        "success": True,
        "data": {
            "target": target,
            "scan_type": request.scan_type,
            "ports_scanned": len(ports),
            "open_ports": len(open_ports),
            "results": open_ports,  # Only return open ports
            "all_results": results if request.scan_type == "quick" else None,
            "timestamp": datetime.now().isoformat(),
        },
    }


# Bandwidth history storage (in-memory for now)
_bandwidth_history = []


@router.get("/bandwidth")
async def get_bandwidth(minutes: int = 30):
    """Get current bandwidth usage and history"""
    global _bandwidth_history

    if not PSUTIL_AVAILABLE:
        return {"success": False, "error": "psutil not available"}

    # Get current IO
    io = psutil.net_io_counters()

    # Calculate approximate speed (using history if available)
    download_speed = 0
    upload_speed = 0

    if _bandwidth_history:
        last = _bandwidth_history[-1]
        time_diff = 5  # Assume 5 second intervals
        download_speed = (
            (io.bytes_recv - last.get("raw_recv", io.bytes_recv)) / time_diff / 1024
        )  # KB/s
        upload_speed = (
            (io.bytes_sent - last.get("raw_sent", io.bytes_sent)) / time_diff / 1024
        )  # KB/s

    # Add current data point
    current_point = {
        "minute": datetime.now().strftime("%H:%M"),
        "download": max(0, round(download_speed, 1)),
        "upload": max(0, round(upload_speed, 1)),
        "raw_recv": io.bytes_recv,
        "raw_sent": io.bytes_sent,
        "timestamp": datetime.now().isoformat(),
    }

    _bandwidth_history.append(current_point)

    # Keep only last N points (based on minutes)
    max_points = minutes * 12  # 5 second intervals = 12 per minute
    if len(_bandwidth_history) > max_points:
        _bandwidth_history = _bandwidth_history[-max_points:]

    # Prepare history for frontend (without raw values)
    history = [
        {"minute": p["minute"], "download": p["download"], "upload": p["upload"]}
        for p in _bandwidth_history
    ]

    return {
        "success": True,
        "data": {
            "history": history,
            "current": {
                "download_speed_kbps": round(download_speed, 1),
                "upload_speed_kbps": round(upload_speed, 1),
            },
            "total_sent_gb": round(io.bytes_sent / 1024 / 1024 / 1024, 2),
            "total_recv_gb": round(io.bytes_recv / 1024 / 1024 / 1024, 2),
            "timestamp": datetime.now().isoformat(),
        },
    }


@router.get("/dns")
async def get_dns_info():
    """Get DNS configuration"""
    hostname = socket.gethostname()

    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = "127.0.0.1"

    try:
        fqdn = socket.getfqdn()
    except:
        fqdn = hostname

    # Try to resolve common domains to check DNS
    dns_check = []
    for domain in ["google.com", "cloudflare.com", "github.com"]:
        try:
            ip = socket.gethostbyname(domain)
            dns_check.append({"domain": domain, "resolved": ip, "status": "ok"})
        except:
            dns_check.append({"domain": domain, "resolved": None, "status": "failed"})

    return {
        "success": True,
        "data": {
            "hostname": hostname,
            "fqdn": fqdn,
            "local_ip": local_ip,
            "dns_check": dns_check,
            "dns_working": all(d["status"] == "ok" for d in dns_check),
        },
    }


@router.get("/stats")
async def get_network_stats():
    """Get comprehensive network statistics"""
    interfaces = get_real_network_interfaces()
    connections = get_active_connections()
    io_total = get_network_io_total()

    # Count by protocol
    tcp_count = len([c for c in connections if c.get("type") == "TCP"])
    udp_count = len([c for c in connections if c.get("type") == "UDP"])

    # Count by status
    established = len([c for c in connections if c.get("status") == "ESTABLISHED"])
    listening = len([c for c in connections if c.get("status") == "LISTEN"])

    return {
        "success": True,
        "data": {
            "interfaces": {
                "total": len(interfaces),
                "active": len([i for i in interfaces if i.get("is_up")]),
            },
            "connections": {
                "total": len(connections),
                "tcp": tcp_count,
                "udp": udp_count,
                "established": established,
                "listening": listening,
            },
            "io": io_total,
            "system": {
                "hostname": socket.gethostname(),
                "platform": platform.system(),
            },
            "timestamp": datetime.now().isoformat(),
            "psutil_available": PSUTIL_AVAILABLE,
        },
    }
