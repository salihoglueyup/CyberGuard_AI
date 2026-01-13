"""
GeoIP Service - Free IP Geolocation
Uses ip-api.com (free, no API key required)
"""

import os
import json
import sqlite3
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import random

# Try to import requests
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
cache_db = os.path.join(project_root, "data", "geoip_cache.db")


# Country coordinates (fallback data)
COUNTRY_COORDS = {
    "US": {"lat": 37.0902, "lng": -95.7129, "name": "United States"},
    "CN": {"lat": 35.8617, "lng": 104.1954, "name": "China"},
    "RU": {"lat": 61.5240, "lng": 105.3188, "name": "Russia"},
    "DE": {"lat": 51.1657, "lng": 10.4515, "name": "Germany"},
    "GB": {"lat": 55.3781, "lng": -3.4360, "name": "United Kingdom"},
    "FR": {"lat": 46.2276, "lng": 2.2137, "name": "France"},
    "JP": {"lat": 36.2048, "lng": 138.2529, "name": "Japan"},
    "KR": {"lat": 35.9078, "lng": 127.7669, "name": "South Korea"},
    "BR": {"lat": -14.2350, "lng": -51.9253, "name": "Brazil"},
    "IN": {"lat": 20.5937, "lng": 78.9629, "name": "India"},
    "TR": {"lat": 38.9637, "lng": 35.2433, "name": "Turkey"},
    "NL": {"lat": 52.1326, "lng": 5.2913, "name": "Netherlands"},
    "UA": {"lat": 48.3794, "lng": 31.1656, "name": "Ukraine"},
    "AU": {"lat": -25.2744, "lng": 133.7751, "name": "Australia"},
    "CA": {"lat": 56.1304, "lng": -106.3468, "name": "Canada"},
    "IT": {"lat": 41.8719, "lng": 12.5674, "name": "Italy"},
    "ES": {"lat": 40.4637, "lng": -3.7492, "name": "Spain"},
    "PL": {"lat": 51.9194, "lng": 19.1451, "name": "Poland"},
    "IR": {"lat": 32.4279, "lng": 53.6880, "name": "Iran"},
    "VN": {"lat": 14.0583, "lng": 108.2772, "name": "Vietnam"},
    "MX": {"lat": 23.6345, "lng": -102.5528, "name": "Mexico"},
    "ID": {"lat": -0.7893, "lng": 113.9213, "name": "Indonesia"},
    "SA": {"lat": 23.8859, "lng": 45.0792, "name": "Saudi Arabia"},
    "AE": {"lat": 23.4241, "lng": 53.8478, "name": "UAE"},
    "SG": {"lat": 1.3521, "lng": 103.8198, "name": "Singapore"},
    "HK": {"lat": 22.3193, "lng": 114.1694, "name": "Hong Kong"},
    "TW": {"lat": 23.6978, "lng": 120.9605, "name": "Taiwan"},
    "TH": {"lat": 15.8700, "lng": 100.9925, "name": "Thailand"},
    "PH": {"lat": 12.8797, "lng": 121.7740, "name": "Philippines"},
    "NG": {"lat": 9.0820, "lng": 8.6753, "name": "Nigeria"},
}


def init_cache_db():
    """Initialize SQLite cache database"""
    os.makedirs(os.path.dirname(cache_db), exist_ok=True)

    conn = sqlite3.connect(cache_db)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS geoip_cache (
            ip TEXT PRIMARY KEY,
            country TEXT,
            country_code TEXT,
            city TEXT,
            lat REAL,
            lng REAL,
            isp TEXT,
            updated_at TEXT
        )
    """
    )
    conn.commit()
    conn.close()


def get_from_cache(ip: str) -> Optional[Dict]:
    """Get IP info from cache"""
    try:
        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT country, country_code, city, lat, lng, isp, updated_at FROM geoip_cache WHERE ip = ?",
            (ip,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            # Check if cache is still valid (24 hours)
            updated_at = datetime.fromisoformat(row[6])
            if datetime.now() - updated_at < timedelta(hours=24):
                return {
                    "ip": ip,
                    "country": row[0],
                    "country_code": row[1],
                    "city": row[2],
                    "lat": row[3],
                    "lng": row[4],
                    "isp": row[5],
                    "cached": True,
                }
    except Exception:
        pass
    return None


def save_to_cache(ip: str, data: Dict):
    """Save IP info to cache"""
    try:
        conn = sqlite3.connect(cache_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO geoip_cache 
            (ip, country, country_code, city, lat, lng, isp, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ip,
                data.get("country", "Unknown"),
                data.get("country_code", "XX"),
                data.get("city", "Unknown"),
                data.get("lat", 0),
                data.get("lng", 0),
                data.get("isp", "Unknown"),
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def lookup_ip(ip: str) -> Dict:
    """
    Look up IP address location using ip-api.com
    Free tier: 45 requests per minute
    """
    # Check cache first
    cached = get_from_cache(ip)
    if cached:
        return cached

    # Skip private/local IPs
    if ip.startswith(("10.", "172.", "192.168.", "127.", "0.")):
        return get_fallback_location("US", ip)

    # Try ip-api.com
    if REQUESTS_AVAILABLE:
        try:
            response = requests.get(
                f"http://ip-api.com/json/{ip}?fields=status,country,countryCode,city,lat,lon,isp",
                timeout=3,
            )

            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    result = {
                        "ip": ip,
                        "country": data.get("country", "Unknown"),
                        "country_code": data.get("countryCode", "XX"),
                        "city": data.get("city", "Unknown"),
                        "lat": data.get("lat", 0),
                        "lng": data.get("lon", 0),
                        "isp": data.get("isp", "Unknown"),
                        "cached": False,
                    }
                    save_to_cache(ip, result)
                    return result
        except Exception as e:
            print(f"[GeoIP] API error for {ip}: {e}")

    # Fallback: guess country from IP range
    return get_fallback_location(guess_country_from_ip(ip), ip)


def guess_country_from_ip(ip: str) -> str:
    """Guess country from IP first octet (simplified)"""
    try:
        first_octet = int(ip.split(".")[0])

        if first_octet in range(1, 56):
            return "US"
        elif first_octet in range(58, 61):
            return "CN"
        elif first_octet in range(77, 95):
            return random.choice(["RU", "DE", "GB", "FR"])
        elif first_octet in range(176, 195):
            return random.choice(["DE", "NL", "FR"])
        elif first_octet in range(200, 210):
            return "BR"
        elif first_octet in range(49, 60):
            return "IN"
        elif first_octet in range(126, 135):
            return "JP"
        elif first_octet in range(110, 125):
            return "KR"
        elif first_octet in range(78, 95):
            return "TR"
        elif first_octet in range(37, 46):
            return "UA"
        else:
            return random.choice(list(COUNTRY_COORDS.keys()))
    except Exception:
        return "US"


def get_fallback_location(country_code: str, ip: str) -> Dict:
    """Get fallback location from country code"""
    coords = COUNTRY_COORDS.get(country_code, COUNTRY_COORDS["US"])

    # Add some randomness to coordinates to avoid clustering
    lat_offset = random.uniform(-2, 2)
    lng_offset = random.uniform(-2, 2)

    return {
        "ip": ip,
        "country": coords["name"],
        "country_code": country_code,
        "city": "Unknown",
        "lat": coords["lat"] + lat_offset,
        "lng": coords["lng"] + lng_offset,
        "isp": "Unknown",
        "cached": False,
        "fallback": True,
    }


def get_country_coords(country_code: str) -> Tuple[float, float]:
    """Get coordinates for a country code"""
    coords = COUNTRY_COORDS.get(country_code, COUNTRY_COORDS["US"])
    return coords["lat"], coords["lng"]


def get_random_coords_in_country(country_code: str) -> Tuple[float, float]:
    """Get random coordinates within a country"""
    lat, lng = get_country_coords(country_code)
    return lat + random.uniform(-3, 3), lng + random.uniform(-3, 3)


# Initialize cache on module load
init_cache_db()
