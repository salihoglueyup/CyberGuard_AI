"""
Threat Analysis API Routes - CyberGuard AI
AI destekli tehdit analizi endpoint'leri

Dosya Yolu: app/api/routes/threat_analysis.py
"""

import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)

from app.api.routes.auth import require_auth

from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)

from src.utils.database import DatabaseManager

router = APIRouter(dependencies=[Depends(require_auth)])
db = DatabaseManager()


class ThreatAnalysisRequest(BaseModel):
    action: str  # summary, critical, defense, trends, investigate
    model_id: str | None = None
    hours: int | None = 24
    context: dict[str, Any] | None = None


@router.post("/analyze")
async def analyze_threats(request: ThreatAnalysisRequest):
    """AI ile tehdit analizi yap"""
    try:
        # Veritabanından verileri çek
        attacks = db.get_attacks(hours=request.hours)
        stats = db.get_attack_stats(hours=request.hours)

        # Analiz tipine göre veri hazırla
        analysis_data = {
            "action": request.action,
            "period_hours": request.hours,
            "total_attacks": len(attacks),
            "stats": stats,
            "timestamp": datetime.now().isoformat(),
        }

        if request.action == "summary":
            analysis_data["summary"] = _generate_summary(attacks, stats)
        elif request.action == "critical":
            critical_attacks = [
                a
                for a in attacks
                if a.get("severity") in ["critical", "high", "CRITICAL", "HIGH"]
            ]
            analysis_data["critical_attacks"] = critical_attacks[:20]
            analysis_data["critical_count"] = len(critical_attacks)
        elif request.action == "trends":
            analysis_data["trends"] = _analyze_trends(attacks)
        elif request.action == "investigate":
            analysis_data["top_attackers"] = db.get_top_attackers(
                limit=10, hours=request.hours or 24
            )
        elif request.action == "defense":
            analysis_data["recommendations"] = _generate_defense_recommendations(
                attacks, stats
            )

        # Gemini'ye gönderilecek prompt oluştur
        prompt = _build_analysis_prompt(request.action, analysis_data)

        return {
            "success": True,
            "data": {
                "analysis_data": analysis_data,
                "prompt": prompt,
                "model_id": request.model_id,
            },
        }

    except Exception as e:
        import traceback

        traceback.print_exc()
        return {"success": False, "error": str(e)}


@router.get("/summary")
async def get_threat_summary(hours: int = Query(24)):
    """Tehdit özeti getir"""
    try:
        attacks = db.get_attacks(hours=hours)
        stats = db.get_attack_stats(hours=hours)

        summary = _generate_summary(attacks, stats)

        return {"success": True, "data": summary}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/critical")
async def get_critical_threats(hours: int = Query(24), limit: int = Query(20)):
    """Kritik tehditleri getir"""
    try:
        attacks = db.get_attacks(hours=hours)
        critical = [
            a
            for a in attacks
            if a.get("severity") in ["critical", "high", "CRITICAL", "HIGH"]
        ]

        return {
            "success": True,
            "data": {
                "critical_attacks": critical[:limit],
                "total_critical": len(
                    [
                        a
                        for a in critical
                        if a.get("severity") in ["critical", "CRITICAL"]
                    ]
                ),
                "total_high": len(
                    [a for a in critical if a.get("severity") in ["high", "HIGH"]]
                ),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/trends")
async def get_threat_trends(hours: int = Query(24)):
    """Tehdit trendlerini getir"""
    try:
        attacks = db.get_attacks(hours=hours)
        trends = _analyze_trends(attacks)

        return {"success": True, "data": trends}
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/recommendations")
async def get_defense_recommendations(hours: int = Query(24)):
    """Savunma önerilerini getir"""
    try:
        attacks = db.get_attacks(hours=hours)
        stats = db.get_attack_stats(hours=hours)
        recommendations = _generate_defense_recommendations(attacks, stats)

        return {"success": True, "data": recommendations}
    except Exception as e:
        return {"success": False, "error": str(e)}


def _generate_summary(attacks: list[dict], stats: dict) -> dict:
    """Tehdit özeti oluştur"""
    total = len(attacks)
    blocked = len([a for a in attacks if a.get("blocked")])

    # Saldırı tipleri
    attack_types = {}
    for a in attacks:
        atype = a.get("attack_type", "Unknown")
        attack_types[atype] = attack_types.get(atype, 0) + 1

    # Ciddiyet dağılımı
    severity_dist = {
        "critical": len(
            [a for a in attacks if a.get("severity") in ["critical", "CRITICAL"]]
        ),
        "high": len([a for a in attacks if a.get("severity") in ["high", "HIGH"]]),
        "medium": len(
            [a for a in attacks if a.get("severity") in ["medium", "MEDIUM"]]
        ),
        "low": len([a for a in attacks if a.get("severity") in ["low", "LOW"]]),
    }

    # En çok saldırı yapan IP'ler
    ip_counts = {}
    for a in attacks:
        ip = a.get("source_ip", "unknown")
        ip_counts[ip] = ip_counts.get(ip, 0) + 1
    top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "total_attacks": total,
        "blocked_attacks": blocked,
        "block_rate": round((blocked / total * 100) if total > 0 else 0, 1),
        "attack_types": attack_types,
        "severity_distribution": severity_dist,
        "top_attacker_ips": [{"ip": ip, "count": count} for ip, count in top_ips],
        "risk_level": (
            "HIGH"
            if severity_dist["critical"] > 10 or severity_dist["high"] > 50
            else "MEDIUM" if severity_dist["high"] > 10 else "LOW"
        ),
    }


def _analyze_trends(attacks: list[dict]) -> dict:
    """Saldırı trendlerini analiz et"""
    if not attacks:
        return {"trends": [], "patterns": []}

    # Saate göre dağılım
    hourly = {}
    for a in attacks:
        ts = a.get("timestamp", "")
        if ts:
            try:
                hour = ts.split("T")[1][:2] if "T" in ts else "00"
                hourly[hour] = hourly.get(hour, 0) + 1
            except (IndexError, ValueError) as e:
                logger.debug(f"Timestamp parse error in threat analysis: {e}")

    # En yoğun saatler
    peak_hours = sorted(hourly.items(), key=lambda x: x[1], reverse=True)[:3]

    # Saldırı tipi trendleri
    type_counts = {}
    for a in attacks:
        atype = a.get("attack_type", "Unknown")
        type_counts[atype] = type_counts.get(atype, 0) + 1

    rising_threats = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        "hourly_distribution": hourly,
        "peak_hours": [{"hour": h, "count": c} for h, c in peak_hours],
        "rising_threats": [{"type": t, "count": c} for t, c in rising_threats],
        "total_analyzed": len(attacks),
    }


def _generate_defense_recommendations(attacks: list[dict], stats: dict) -> list[dict]:
    """Savunma önerileri oluştur"""
    recommendations = []

    # Saldırı tipine göre öneriler
    attack_types = {}
    for a in attacks:
        atype = a.get("attack_type", "Unknown")
        attack_types[atype] = attack_types.get(atype, 0) + 1

    if attack_types.get("DDoS", 0) > 10:
        recommendations.append(
            {
                "priority": "HIGH",
                "type": "DDoS Protection",
                "recommendation": "DDoS koruma çözümünü aktifleştirin veya kapasitesini artırın",
                "details": f"Son dönemde {attack_types.get('DDoS', 0)} DDoS saldırısı tespit edildi",
            }
        )

    if attack_types.get("SQL Injection", 0) > 5:
        recommendations.append(
            {
                "priority": "CRITICAL",
                "type": "SQL Injection",
                "recommendation": "WAF kurallarını güncelleyin ve parametre validasyonunu kontrol edin",
                "details": f"{attack_types.get('SQL Injection', 0)} SQL Injection denemesi tespit edildi",
            }
        )

    if attack_types.get("Brute Force", 0) > 20:
        recommendations.append(
            {
                "priority": "HIGH",
                "type": "Brute Force",
                "recommendation": "Rate limiting ve CAPTCHA kullanımını artırın, hesap kilitleme politikasını uygulayın",
                "details": f"{attack_types.get('Brute Force', 0)} Brute Force saldırısı tespit edildi",
            }
        )

    if attack_types.get("XSS", 0) > 5:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "type": "XSS",
                "recommendation": "Content Security Policy (CSP) header'larını yapılandırın",
                "details": f"{attack_types.get('XSS', 0)} XSS denemesi tespit edildi",
            }
        )

    # Genel öneriler
    total = len(attacks)
    blocked = len([a for a in attacks if a.get("blocked")])
    block_rate = (blocked / total * 100) if total > 0 else 0

    if block_rate < 60:
        recommendations.append(
            {
                "priority": "CRITICAL",
                "type": "Block Rate",
                "recommendation": "Engelleme oranı düşük! IDS/IPS kurallarını gözden geçirin",
                "details": f"Mevcut engelleme oranı: %{block_rate:.1f}",
            }
        )

    return recommendations


def _build_analysis_prompt(action: str, data: dict) -> str:
    """Gemini için analiz prompt'u oluştur"""
    prompts = {
        "summary": f"""
Aşağıdaki tehdit verilerini analiz et ve özet bir rapor hazırla:

📊 ÖZET VERİLER:
- Toplam Saldırı: {data.get('total_attacks', 0)}
- İstatistikler: {data.get('stats', {})}

Lütfen:
1. Genel güvenlik durumunu değerlendir
2. En kritik tehditleri vurgula
3. Acil aksiyon önerilerini listele
""",
        "critical": f"""
Kritik ve yüksek seviyeli saldırıları analiz et:

🔴 KRİTİK VERİLER:
- Kritik Saldırı Sayısı: {data.get('critical_count', 0)}
- Son Kritik Saldırılar: {data.get('critical_attacks', [])[:5]}

Lütfen:
1. Her kritik saldırıyı değerlendir
2. Acil müdahale gerektiren durumları belirt
3. Risk azaltma stratejileri öner
""",
        "trends": f"""
Saldırı trendlerini analiz et:

📈 TREND VERİLERİ:
{data.get('trends', {})}

Lütfen:
1. Artış gösteren tehditleri belirt
2. Pattern ve anomalileri tespit et
3. Gelecek dönem için tahminler yap
""",
        "investigate": f"""
En çok saldırı yapan IP'leri araştır:

🔍 ÜST SALDIRGANLAR:
{data.get('top_attackers', [])}

Lütfen:
1. Her IP için risk değerlendirmesi yap
2. Bloklanması gerekenleri belirt
3. Olası saldırgan profillerini analiz et
""",
        "defense": f"""
Savunma önerileri oluştur:

🛡️ ÖNERİLER:
{data.get('recommendations', [])}

Lütfen:
1. Önerileri önceliklendirerek detaylandır
2. Uygulama adımlarını açıkla
3. Beklenen iyileşme oranlarını tahmin et
""",
    }

    return prompts.get(action, "Tehdit analizi yap ve öneriler sun.")


# ============= MITRE ATT&CK Framework =============

MITRE_TACTICS = {
    "TA0001": {"name": "Initial Access", "name_tr": "İlk Erişim", "color": "#ef4444"},
    "TA0002": {"name": "Execution", "name_tr": "Yürütme", "color": "#f97316"},
    "TA0003": {"name": "Persistence", "name_tr": "Kalıcılık", "color": "#eab308"},
    "TA0004": {
        "name": "Privilege Escalation",
        "name_tr": "Yetki Yükseltme",
        "color": "#22c55e",
    },
    "TA0005": {
        "name": "Defense Evasion",
        "name_tr": "Savunma Atlatma",
        "color": "#14b8a6",
    },
    "TA0006": {
        "name": "Credential Access",
        "name_tr": "Kimlik Bilgisi Erişimi",
        "color": "#3b82f6",
    },
    "TA0007": {"name": "Discovery", "name_tr": "Keşif", "color": "#8b5cf6"},
    "TA0008": {
        "name": "Lateral Movement",
        "name_tr": "Yanal Hareket",
        "color": "#ec4899",
    },
    "TA0009": {"name": "Collection", "name_tr": "Toplama", "color": "#f43f5e"},
    "TA0010": {"name": "Exfiltration", "name_tr": "Veri Sızdırma", "color": "#a855f7"},
    "TA0011": {
        "name": "Command and Control",
        "name_tr": "Komuta ve Kontrol",
        "color": "#6366f1",
    },
    "TA0040": {"name": "Impact", "name_tr": "Etki", "color": "#dc2626"},
}

ATTACK_TO_MITRE = {
    "DDoS": {
        "tactic": "TA0040",
        "technique": "T1499",
        "name": "Endpoint Denial of Service",
    },
    "DoS": {
        "tactic": "TA0040",
        "technique": "T1499",
        "name": "Endpoint Denial of Service",
    },
    "SQL Injection": {
        "tactic": "TA0001",
        "technique": "T1190",
        "name": "Exploit Public-Facing Application",
    },
    "XSS": {"tactic": "TA0001", "technique": "T1189", "name": "Drive-by Compromise"},
    "Brute Force": {"tactic": "TA0006", "technique": "T1110", "name": "Brute Force"},
    "Port Scan": {
        "tactic": "TA0007",
        "technique": "T1046",
        "name": "Network Service Scanning",
    },
    "Probe": {
        "tactic": "TA0007",
        "technique": "T1046",
        "name": "Network Service Scanning",
    },
    "R2L": {
        "tactic": "TA0001",
        "technique": "T1133",
        "name": "External Remote Services",
    },
    "U2R": {
        "tactic": "TA0004",
        "technique": "T1068",
        "name": "Exploitation for Privilege Escalation",
    },
    "Botnet": {
        "tactic": "TA0011",
        "technique": "T1071",
        "name": "Application Layer Protocol",
    },
    "Malware": {
        "tactic": "TA0002",
        "technique": "T1059",
        "name": "Command and Scripting Interpreter",
    },
    "Phishing": {"tactic": "TA0001", "technique": "T1566", "name": "Phishing"},
}


@router.get("/mitre/tactics")
async def get_mitre_tactics():
    """MITRE ATT&CK taktiklerini getir"""
    return {"success": True, "data": [{"id": k, **v} for k, v in MITRE_TACTICS.items()]}


@router.get("/mitre/mapping")
async def get_mitre_mapping(hours: int = Query(24)):
    """Saldırıları MITRE ATT&CK'e eşle"""
    try:
        attacks = db.get_attacks(hours=hours)

        mapping = {}
        for attack in attacks:
            attack_type = attack.get("attack_type", "Unknown")
            if attack_type in ATTACK_TO_MITRE:
                mitre = ATTACK_TO_MITRE[attack_type]
                tactic_id = mitre["tactic"]

                if tactic_id not in mapping:
                    mapping[tactic_id] = {
                        **MITRE_TACTICS.get(tactic_id, {}),
                        "techniques": {},
                        "count": 0,
                    }

                technique = mitre["technique"]
                if technique not in mapping[tactic_id]["techniques"]:
                    mapping[tactic_id]["techniques"][technique] = {
                        "id": technique,
                        "name": mitre["name"],
                        "count": 0,
                        "attacks": [],
                    }

                mapping[tactic_id]["techniques"][technique]["count"] += 1
                mapping[tactic_id]["count"] += 1

                if len(mapping[tactic_id]["techniques"][technique]["attacks"]) < 5:
                    mapping[tactic_id]["techniques"][technique]["attacks"].append(
                        {
                            "id": attack.get("id"),
                            "source_ip": attack.get("source_ip"),
                            "timestamp": attack.get("timestamp"),
                        }
                    )

        return {
            "success": True,
            "data": {
                "mapping": list(mapping.values()),
                "total_mapped": sum(t["count"] for t in mapping.values()),
                "total_attacks": len(attacks),
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============= IOC (Indicator of Compromise) =============

# In-memory IOC store (production'da veritabanı kullanılmalı)
IOC_STORE: list[dict] = []


class IOCCreateRequest(BaseModel):
    type: str  # ip, domain, hash, url
    value: str
    severity: str = "medium"  # low, medium, high, critical
    description: str | None = None
    source: str | None = None
    tags: list[str] | None = []


@router.get("/ioc")
async def get_iocs(
    ioc_type: str | None = None,
    severity: str | None = None,
    limit: int = Query(100),
):
    """IOC listesini getir"""
    result = IOC_STORE.copy()

    if ioc_type:
        result = [i for i in result if i.get("type") == ioc_type]
    if severity:
        result = [i for i in result if i.get("severity") == severity]

    return {"success": True, "data": result[:limit], "total": len(result)}


@router.post("/ioc")
async def create_ioc(request: IOCCreateRequest):
    """Yeni IOC ekle"""
    ioc = {
        "id": f"IOC-{len(IOC_STORE)+1:06d}",
        "type": request.type,
        "value": request.value,
        "severity": request.severity,
        "description": request.description,
        "source": request.source,
        "tags": request.tags or [],
        "created_at": datetime.now().isoformat(),
        "hits": 0,
        "last_seen": None,
    }
    IOC_STORE.append(ioc)

    return {"success": True, "data": ioc}


@router.delete("/ioc/{ioc_id}")
async def delete_ioc(ioc_id: str):
    """IOC sil"""
    global IOC_STORE
    IOC_STORE = [i for i in IOC_STORE if i.get("id") != ioc_id]
    return {"success": True, "message": f"IOC {ioc_id} silindi"}


@router.post("/ioc/check")
async def check_ioc(values: list[str]):
    """Değerlerin IOC listesinde olup olmadığını kontrol et"""
    results = []
    for value in values:
        match = next((i for i in IOC_STORE if i.get("value") == value), None)
        if match:
            match["hits"] = match.get("hits", 0) + 1
            match["last_seen"] = datetime.now().isoformat()
            results.append({"value": value, "found": True, "ioc": match})
        else:
            results.append({"value": value, "found": False})

    return {"success": True, "data": results}


# ============= IP Reputation =============


@router.get("/ip-reputation/{ip}")
async def get_ip_reputation(ip: str):
    """IP reputation bilgisi (simüle edilmiş)"""
    # Gerçek uygulamada AbuseIPDB, VirusTotal API kullanılır
    import hashlib

    # Deterministik score (demo için)
    hash_val = int(hashlib.md5(ip.encode()).hexdigest()[:8], 16)
    risk_score = hash_val % 100

    # Bilinen kötü IP pattern'leri
    is_known_bad = ip.startswith("10.") or ip.startswith("192.168.") or risk_score > 80

    return {
        "success": True,
        "data": {
            "ip": ip,
            "risk_score": risk_score,
            "risk_level": (
                "critical"
                if risk_score > 80
                else (
                    "high"
                    if risk_score > 60
                    else "medium" if risk_score > 40 else "low"
                )
            ),
            "is_known_bad": is_known_bad,
            "country": (
                "TR" if hash_val % 3 == 0 else "US" if hash_val % 3 == 1 else "RU"
            ),
            "asn": f"AS{hash_val % 65535}",
            "reports": hash_val % 50,
            "last_reported": datetime.now().isoformat() if risk_score > 50 else None,
            "categories": (
                ["botnet", "scanner"]
                if risk_score > 70
                else ["suspicious"] if risk_score > 40 else []
            ),
        },
    }


# ============= Real-time IDS Status =============

# Global IDS instance reference
_realtime_ids_instance = None


def get_realtime_ids():
    """Real-time IDS instance'ı al"""
    global _realtime_ids_instance
    return _realtime_ids_instance


def set_realtime_ids(instance):
    """Real-time IDS instance'ı set et"""
    global _realtime_ids_instance
    _realtime_ids_instance = instance


@router.get("/realtime-ids/status")
async def get_ids_status():
    """Real-time IDS durumunu getir"""
    ids = get_realtime_ids()

    if ids is None:
        return {
            "success": True,
            "data": {
                "status": "not_initialized",
                "message": "Real-time IDS başlatılmadı",
            },
        }

    try:
        metrics = ids.get_metrics()
        drift_detected, drift_score = ids.check_drift()

        return {
            "success": True,
            "data": {
                "status": "running" if ids.is_running else "stopped",
                "metrics": metrics,
                "drift_detected": drift_detected,
                "drift_score": drift_score,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@router.get("/realtime-ids/alerts")
async def get_ids_alerts(limit: int = Query(20)):
    """Real-time IDS alert'lerini getir"""
    ids = get_realtime_ids()

    if ids is None:
        return {"success": True, "data": []}

    try:
        alerts = ids.get_recent_alerts(limit)
        return {"success": True, "data": alerts}
    except Exception as e:
        return {"success": False, "error": str(e)}
