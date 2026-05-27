"""
CyberGuard AI — Threat Decision Agent
======================================

Tehdit tespiti → Playbook önerisi → Otomatik Incident oluşturma akışı.

Mevcut LLM sağlayıcılarını (Groq, OpenAI, Claude, Gemini, Ollama) kullanır.
Sağlayıcı yoksa kural tabanlı fallback çalışır.

Kullanım:
    from src.ai_decision.threat_agent import ThreatDecisionAgent

    agent = ThreatDecisionAgent()
    result = await agent.handle_threat(threat_event)
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLAYBOOKS_FILE = PROJECT_ROOT / "data" / "playbooks.json"
INCIDENTS_FILE = PROJECT_ROOT / "data" / "incidents.json"


# ---------------------------------------------------------------------------
# Kural tabanlı Playbook seçici (LLM olmadan da çalışır)
# ---------------------------------------------------------------------------

_PLAYBOOK_RULES: dict[str, list[str]] = {
    "DDoS":          ["rate-limit-ip", "contact-isp", "activate-scrubbing"],
    "Brute Force":   ["block-ip-15min", "force-mfa", "notify-soc"],
    "SQL Injection": ["block-ip", "patch-waf-rule", "review-db-logs"],
    "XSS":           ["sanitize-input", "update-csp-header", "notify-dev"],
    "Malware":       ["quarantine-host", "run-av-scan", "isolate-network"],
    "Phishing":      ["block-domain", "notify-users", "reset-credentials"],
    "Port Scan":     ["log-event", "trigger-honeypot", "rate-limit-ip"],
    "Ransomware":    ["isolate-host", "snapshot-restore", "activate-ir-plan"],
}

_DEFAULT_PLAYBOOK = ["log-event", "notify-soc", "manual-review"]


def _rule_based_playbook(attack_type: str) -> list[str]:
    for key, steps in _PLAYBOOK_RULES.items():
        if key.lower() in attack_type.lower():
            return steps
    return _DEFAULT_PLAYBOOK


def _rule_based_severity(confidence: float, attack_type: str) -> str:
    critical_types = {"Ransomware", "Malware", "SQL Injection"}
    if any(t.lower() in attack_type.lower() for t in critical_types):
        return "CRITICAL" if confidence > 0.8 else "HIGH"
    if confidence > 0.9:
        return "HIGH"
    if confidence > 0.7:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# LLM entegrasyonu (opsiyonel)
# ---------------------------------------------------------------------------

def _build_prompt(threat_event: dict) -> str:
    return f"""You are a senior SOC analyst for a cybersecurity platform.
A threat has been detected. Respond ONLY with valid JSON (no markdown, no explanation).

Threat Event:
{json.dumps(threat_event, indent=2)}

Required JSON output:
{{
  "severity": "LOW|MEDIUM|HIGH|CRITICAL",
  "summary": "one-sentence description",
  "playbook_steps": ["step1", "step2", "step3"],
  "escalate": true|false
}}"""


async def _llm_analyze(threat_event: dict) -> dict | None:
    """LLM ile tehdit analizi. Başarısız olursa None döner."""
    try:
        provider = os.environ.get("LLM_PROVIDER", "").lower()
        api_key = os.environ.get("LLM_API_KEY", "")
        model = os.environ.get("LLM_MODEL", "")

        if not provider or not api_key:
            return None

        prompt = _build_prompt(threat_event)

        if provider == "groq":
            from groq import AsyncGroq
            client = AsyncGroq(api_key=api_key)
            response = await client.chat.completions.create(
                model=model or "llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content

        elif provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=api_key)
            response = await client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            raw = response.choices[0].message.content

        elif provider == "ollama":
            import httpx
            ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{ollama_url}/api/generate",
                    json={"model": model or "llama3", "prompt": prompt, "stream": False},
                )
                raw = resp.json().get("response", "")

        else:
            return None

        return json.loads(raw)

    except Exception as e:
        logger.warning(f"[ThreatAgent] LLM analizi başarısız: {e}")
        return None


# ---------------------------------------------------------------------------
# Incident yazıcısı
# ---------------------------------------------------------------------------

def _save_incident(incident: dict) -> None:
    try:
        incidents: list[dict] = []
        if INCIDENTS_FILE.exists():
            with open(INCIDENTS_FILE, encoding="utf-8") as f:
                data = json.load(f)
                incidents = data if isinstance(data, list) else data.get("incidents", [])

        incidents.append(incident)

        INCIDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(incidents, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[ThreatAgent] Incident kaydedilemedi: {e}")


# ---------------------------------------------------------------------------
# Ana ajan sınıfı
# ---------------------------------------------------------------------------

class ThreatDecisionAgent:
    """
    Tehdit olaylarını alır, LLM veya kural motoruyla analiz eder
    ve otomatik incident + playbook önerisi oluşturur.
    """

    async def handle_threat(self, threat_event: dict) -> dict:
        """
        Bir tehdit olayını işle ve karşılık döndür.

        Args:
            threat_event: {
                "attack_type": str,
                "source_ip": str,
                "confidence": float,   # 0.0 – 1.0
                "timestamp": str,      # ISO 8601
                "details": dict        # opsiyonel ek bilgi
            }

        Returns:
            {
                "incident_id": str,
                "severity": str,
                "summary": str,
                "playbook_steps": list[str],
                "escalate": bool,
                "source": "llm" | "rule-based"
            }
        """
        attack_type = threat_event.get("attack_type", "Unknown")
        confidence = float(threat_event.get("confidence", 0.5))

        # LLM analiz dene
        llm_result = await _llm_analyze(threat_event)

        if llm_result:
            severity       = llm_result.get("severity", "MEDIUM")
            summary        = llm_result.get("summary", "LLM analizi tamamlandı.")
            playbook_steps = llm_result.get("playbook_steps", _DEFAULT_PLAYBOOK)
            escalate       = bool(llm_result.get("escalate", False))
            source         = "llm"
        else:
            severity       = _rule_based_severity(confidence, attack_type)
            summary        = f"{attack_type} saldırısı tespit edildi (güven: {confidence:.0%})"
            playbook_steps = _rule_based_playbook(attack_type)
            escalate       = severity in ("HIGH", "CRITICAL")
            source         = "rule-based"

        # Incident oluştur
        incident = {
            "id": str(uuid.uuid4()),
            "title": f"[AUTO] {attack_type} — {severity}",
            "severity": severity,
            "status": "open",
            "source_ip": threat_event.get("source_ip", "unknown"),
            "attack_type": attack_type,
            "confidence": confidence,
            "summary": summary,
            "playbook_steps": playbook_steps,
            "escalate": escalate,
            "analysis_source": source,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "raw_event": threat_event,
        }

        _save_incident(incident)
        logger.info(
            f"[ThreatAgent] Incident oluşturuldu: {incident['id']} "
            f"({severity}, kaynak={source})"
        )

        return {
            "incident_id": incident["id"],
            "severity": severity,
            "summary": summary,
            "playbook_steps": playbook_steps,
            "escalate": escalate,
            "source": source,
        }
