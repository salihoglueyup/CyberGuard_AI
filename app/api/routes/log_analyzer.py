"""
Log Analyzer API - REAL DATA VERSION
Analyzes actual log files from the logs directory
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
import os
import glob
import re

router = APIRouter()

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
logs_dir = os.path.join(project_root, "logs")


class LogQuery(BaseModel):
    pattern: str
    log_file: Optional[str] = None
    limit: int = 100


def get_log_files() -> List[Dict]:
    files = []
    if os.path.exists(logs_dir):
        for f in glob.glob(os.path.join(logs_dir, "*.log")):
            stat = os.stat(f)
            files.append(
                {
                    "name": os.path.basename(f),
                    "path": f,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )
    return files


def read_log_file(filepath: str, limit: int = 1000) -> List[str]:
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        return lines[-limit:]
    except:
        return []


def parse_log_line(line: str) -> Dict:
    # Try common log formats
    patterns = [
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(?P<level>\w+)\s+(?P<message>.*)$",
        r"^\[(?P<timestamp>[^\]]+)\]\s+(?P<level>\w+)\s+(?P<message>.*)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, line.strip())
        if match:
            return match.groupdict()

    return {"raw": line.strip()}


@router.get("/status")
async def get_status():
    files = get_log_files()
    total_size = sum(f["size_kb"] for f in files)
    return {
        "success": True,
        "data": {
            "log_files": len(files),
            "total_size_kb": total_size,
            "logs_dir": logs_dir,
        },
    }


@router.get("/files")
async def list_log_files():
    return {"success": True, "data": {"files": get_log_files()}}


@router.get("/file/{filename}")
async def get_log_content(filename: str, lines: int = 100):
    filepath = os.path.join(logs_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404)

    content = read_log_file(filepath, lines)
    parsed = [parse_log_line(l) for l in content]

    return {
        "success": True,
        "data": {"filename": filename, "lines": len(content), "content": parsed},
    }


@router.post("/search")
async def search_logs(query: LogQuery):
    results = []
    files_to_search = []

    if query.log_file:
        files_to_search = [os.path.join(logs_dir, query.log_file)]
    else:
        files_to_search = glob.glob(os.path.join(logs_dir, "*.log"))

    for filepath in files_to_search:
        if not os.path.exists(filepath):
            continue
        lines = read_log_file(filepath, 5000)
        for i, line in enumerate(lines):
            if query.pattern.lower() in line.lower():
                results.append(
                    {
                        "file": os.path.basename(filepath),
                        "line_number": i + 1,
                        "content": line.strip()[:500],
                    }
                )
                if len(results) >= query.limit:
                    break
        if len(results) >= query.limit:
            break

    return {
        "success": True,
        "data": {"pattern": query.pattern, "matches": len(results), "results": results},
    }


@router.get("/stats")
async def get_log_stats():
    files = get_log_files()
    level_counts = {"ERROR": 0, "WARNING": 0, "INFO": 0, "DEBUG": 0}

    for f in files[:5]:  # Analyze first 5 files
        lines = read_log_file(f["path"], 1000)
        for line in lines:
            upper = line.upper()
            if "ERROR" in upper:
                level_counts["ERROR"] += 1
            elif "WARNING" in upper or "WARN" in upper:
                level_counts["WARNING"] += 1
            elif "INFO" in upper:
                level_counts["INFO"] += 1
            elif "DEBUG" in upper:
                level_counts["DEBUG"] += 1

    return {
        "success": True,
        "data": {"files_count": len(files), "level_counts": level_counts},
    }


@router.get("/errors")
async def get_recent_errors(limit: int = 50):
    errors = []
    for f in get_log_files():
        lines = read_log_file(f["path"], 2000)
        for i, line in enumerate(lines):
            if "error" in line.lower() or "exception" in line.lower():
                errors.append(
                    {"file": f["name"], "line": i + 1, "content": line.strip()[:300]}
                )
                if len(errors) >= limit:
                    break
        if len(errors) >= limit:
            break
    return {"success": True, "data": {"errors": errors, "total": len(errors)}}
