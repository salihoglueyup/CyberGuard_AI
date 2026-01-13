"""
Container Security API - REAL DATA VERSION
Uses Docker API when available, with fallback data
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
import json
import subprocess

# Try to import Docker
try:
    import docker

    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

router = APIRouter()

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
data_dir = os.path.join(project_root, "data")
scan_results_file = os.path.join(data_dir, "container_scans.json")

# In-memory scan results
SCAN_RESULTS = []


class ContainerScanRequest(BaseModel):
    image: str
    scan_type: str = "quick"  # quick, full


def load_scan_results():
    """Load scan results from file"""
    global SCAN_RESULTS

    if os.path.exists(scan_results_file):
        try:
            with open(scan_results_file, "r", encoding="utf-8") as f:
                SCAN_RESULTS = json.load(f)
        except:
            pass


def save_scan_results():
    """Save scan results to file"""
    os.makedirs(os.path.dirname(scan_results_file), exist_ok=True)

    with open(scan_results_file, "w", encoding="utf-8") as f:
        json.dump(SCAN_RESULTS[-100:], f, indent=2, default=str)


def get_docker_client():
    """Get Docker client if available"""
    if not DOCKER_AVAILABLE:
        return None

    try:
        client = docker.from_env()
        client.ping()
        return client
    except:
        return None


def get_container_list() -> List[Dict]:
    """Get list of running containers"""
    client = get_docker_client()

    if not client:
        return []

    containers = []
    try:
        for c in client.containers.list(all=True):
            containers.append(
                {
                    "id": c.short_id,
                    "name": c.name,
                    "image": c.image.tags[0] if c.image.tags else c.image.short_id,
                    "status": c.status,
                    "created": c.attrs.get("Created", ""),
                    "ports": c.ports,
                }
            )
    except Exception as e:
        print(f"Error getting containers: {e}")

    return containers


def get_image_list() -> List[Dict]:
    """Get list of Docker images"""
    client = get_docker_client()

    if not client:
        return []

    images = []
    try:
        for img in client.images.list():
            images.append(
                {
                    "id": img.short_id,
                    "tags": img.tags,
                    "created": img.attrs.get("Created", ""),
                    "size_mb": round(img.attrs.get("Size", 0) / 1024 / 1024, 2),
                }
            )
    except Exception as e:
        print(f"Error getting images: {e}")

    return images


def analyze_image(image_name: str) -> Dict:
    """Analyze a Docker image for vulnerabilities"""
    vulnerabilities = []

    client = get_docker_client()
    if client:
        try:
            # Get image details
            image = client.images.get(image_name)

            # Check for common issues
            config = image.attrs.get("Config", {})

            # Check if running as root
            user = config.get("User", "")
            if not user or user == "root" or user == "0":
                vulnerabilities.append(
                    {
                        "id": "SEC-ROOT-USER",
                        "title": "Container runs as root",
                        "severity": "high",
                        "description": "Container is configured to run as root user",
                        "remediation": "Add USER directive to Dockerfile",
                    }
                )

            # Check for exposed ports
            exposed = config.get("ExposedPorts", {})
            dangerous_ports = ["22/tcp", "23/tcp", "3389/tcp"]
            for port in dangerous_ports:
                if port in exposed:
                    vulnerabilities.append(
                        {
                            "id": f"SEC-PORT-{port.split('/')[0]}",
                            "title": f"Dangerous port {port} exposed",
                            "severity": "medium",
                            "description": f"Port {port} is exposed which may be a security risk",
                            "remediation": "Remove dangerous port exposure",
                        }
                    )

            # Check environment variables for secrets
            env_vars = config.get("Env", [])
            secret_patterns = ["PASSWORD", "SECRET", "API_KEY", "TOKEN", "PRIVATE"]
            for env in env_vars:
                for pattern in secret_patterns:
                    if pattern in env.upper() and "=" in env:
                        vulnerabilities.append(
                            {
                                "id": "SEC-ENV-SECRET",
                                "title": "Potential secret in environment variable",
                                "severity": "high",
                                "description": f"Environment variable may contain sensitive data",
                                "remediation": "Use Docker secrets or external secret management",
                            }
                        )
                        break

            return {
                "image": image_name,
                "image_id": image.short_id,
                "size_mb": round(image.attrs.get("Size", 0) / 1024 / 1024, 2),
                "created": image.attrs.get("Created", ""),
                "vulnerabilities": vulnerabilities,
                "vulnerability_count": len(vulnerabilities),
                "scan_status": "completed",
            }

        except docker.errors.ImageNotFound:
            return {"error": "Image not found", "scan_status": "failed"}
        except Exception as e:
            return {"error": str(e), "scan_status": "failed"}

    return {
        "image": image_name,
        "scan_status": "skipped",
        "note": "Docker not available",
        "vulnerabilities": [],
    }


# Initialize
load_scan_results()


@router.get("/status")
async def get_container_security_status():
    """Get container security status"""
    client = get_docker_client()
    containers = get_container_list() if client else []
    images = get_image_list() if client else []

    return {
        "success": True,
        "data": {
            "docker_available": client is not None,
            "docker_version": client.version()["Version"] if client else None,
            "containers_total": len(containers),
            "containers_running": len(
                [c for c in containers if c.get("status") == "running"]
            ),
            "images_total": len(images),
            "scans_performed": len(SCAN_RESULTS),
        },
    }


@router.get("/containers")
async def get_containers():
    """Get all containers"""
    containers = get_container_list()

    return {
        "success": True,
        "data": {
            "containers": containers,
            "total": len(containers),
            "docker_available": get_docker_client() is not None,
        },
    }


@router.get("/images")
async def get_images():
    """Get all Docker images"""
    images = get_image_list()

    return {"success": True, "data": {"images": images, "total": len(images)}}


@router.post("/scan")
async def scan_image(request: ContainerScanRequest):
    """Scan a Docker image for vulnerabilities"""
    result = analyze_image(request.image)

    # Save scan result
    scan_record = {
        "id": f"SCAN-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "image": request.image,
        "scan_type": request.scan_type,
        "timestamp": datetime.now().isoformat(),
        "result": result,
    }

    SCAN_RESULTS.append(scan_record)
    save_scan_results()

    return {"success": True, "data": scan_record}


@router.get("/scans")
async def get_scan_history(limit: int = 20):
    """Get scan history"""
    scans = sorted(SCAN_RESULTS, key=lambda x: x.get("timestamp", ""), reverse=True)

    return {"success": True, "data": {"scans": scans[:limit], "total": len(scans)}}


@router.get("/scan/{scan_id}")
async def get_scan_result(scan_id: str):
    """Get specific scan result"""
    for scan in SCAN_RESULTS:
        if scan.get("id") == scan_id:
            return {"success": True, "data": scan}

    raise HTTPException(status_code=404, detail="Scan not found")


@router.get("/vulnerabilities")
async def get_all_vulnerabilities():
    """Get all discovered vulnerabilities"""
    all_vulns = []

    for scan in SCAN_RESULTS:
        result = scan.get("result", {})
        vulns = result.get("vulnerabilities", [])
        for v in vulns:
            all_vulns.append(
                {
                    **v,
                    "image": scan.get("image"),
                    "scan_id": scan.get("id"),
                    "scan_date": scan.get("timestamp"),
                }
            )

    # Count by severity
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for v in all_vulns:
        sev = v.get("severity", "unknown")
        if sev in by_severity:
            by_severity[sev] += 1

    return {
        "success": True,
        "data": {
            "vulnerabilities": all_vulns,
            "total": len(all_vulns),
            "by_severity": by_severity,
        },
    }


@router.get("/stats")
async def get_container_stats():
    """Get container security statistics"""
    containers = get_container_list()
    images = get_image_list()

    # Count vulnerabilities
    total_vulns = 0
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}

    for scan in SCAN_RESULTS:
        result = scan.get("result", {})
        vulns = result.get("vulnerabilities", [])
        total_vulns += len(vulns)
        for v in vulns:
            sev = v.get("severity", "unknown")
            if sev in by_severity:
                by_severity[sev] += 1

    return {
        "success": True,
        "data": {
            "containers": {
                "total": len(containers),
                "running": len([c for c in containers if c.get("status") == "running"]),
                "stopped": len([c for c in containers if c.get("status") != "running"]),
            },
            "images": {
                "total": len(images),
                "total_size_mb": sum(i.get("size_mb", 0) for i in images),
            },
            "vulnerabilities": {"total": total_vulns, "by_severity": by_severity},
            "scans_performed": len(SCAN_RESULTS),
            "docker_available": get_docker_client() is not None,
        },
    }
