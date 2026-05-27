"""
Context Builder - CyberGuard AI
===============================

AI Chatbot için zengin context oluşturur.

Özellikler:
    - Son saldırılar
    - Model sonuçları
    - IDS durumu
    - Smart action detection
"""

import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
logger = logging.getLogger("ContextBuilder")


class ContextBuilder:
    """
    AI Chatbot için zengin context oluşturur
    """

    # Smart action keywords
    SMART_ACTIONS = {
        "ddos analizi": "analyze_ddos",
        "ddos analiz": "analyze_ddos",
        "port tarama": "analyze_portscan",
        "portscan": "analyze_portscan",
        "ip kontrol": "check_ip",
        "ip sorgula": "check_ip",
        "model karşılaştır": "compare_models",
        "model öner": "recommend_model",
        "yara kuralı": "generate_yara",
        "firewall kural": "generate_firewall",
        "mitre": "mitre_mapping",
        "ids durumu": "ids_status",
    }

    def __init__(self):
        self.models_dir = PROJECT_ROOT / "models"
        self.data_dir = PROJECT_ROOT / "data"

    def build_context(
        self,
        include_attacks: bool = True,
        include_models: bool = True,
        include_ids: bool = True,
        max_attacks: int = 10,
    ) -> str:
        """
        Kapsamlı context oluştur
        """
        context_parts = []

        # Model bilgileri
        if include_models:
            model_context = self._get_model_context()
            if model_context:
                context_parts.append(model_context)

        # Son saldırılar
        if include_attacks:
            attack_context = self._get_attack_context(max_attacks)
            if attack_context:
                context_parts.append(attack_context)

        # IDS durumu
        if include_ids:
            ids_context = self._get_ids_context()
            if ids_context:
                context_parts.append(ids_context)

        if not context_parts:
            return ""

        return "\n\n".join(context_parts)

    def _get_model_context(self) -> str:
        """Model sonuçları context'i"""
        try:
            from src.chatbot.model_integration import get_integration

            integration = get_integration()

            if not integration.training_results:
                return ""

            lines = ["📊 **MEVCUT MODEL SONUÇLARI:**"]

            for name, results in list(integration.training_results.items())[:5]:
                if isinstance(results, dict):
                    acc = results.get("accuracy", 0)
                    if isinstance(acc, float) and acc < 1:
                        acc *= 100
                    lines.append(f"  - {name}: %{acc:.2f} accuracy")

            model_count = len(integration.get_available_models())
            lines.append(f"\n📦 Toplam {model_count} eğitilmiş model mevcut.")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Model context error: {e}")
            return ""

    def _get_attack_context(self, max_attacks: int = 10) -> str:
        """Son saldırılar context'i"""
        try:
            import sqlite3

            db_path = self.data_dir / "cyberguard.db"
            if not db_path.exists():
                return ""

            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()

                # Tablolar mevcut mu kontrol et
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='attacks'"
                )
                if not cursor.fetchone():
                    return ""

                cursor.execute(
                    """
                    SELECT attack_type, severity, source_ip, timestamp
                    FROM attacks
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (max_attacks,),
                )

                rows = cursor.fetchall()
                if not rows:
                    return ""

                lines = ["🚨 **SON SALDIRILAR:**"]
                for row in rows[:5]:
                    attack_type, severity, source_ip, timestamp = row
                    lines.append(f"  - {attack_type} ({severity}) - {source_ip}")

                # İstatistikler
                cursor.execute("SELECT COUNT(*) FROM attacks")
                total = cursor.fetchone()[0]

                cursor.execute(
                    """
                    SELECT attack_type, COUNT(*) as cnt
                    FROM attacks
                    GROUP BY attack_type
                    ORDER BY cnt DESC
                    LIMIT 3
                """
                )
                top_types = cursor.fetchall()

                if top_types:
                    lines.append(f"\n📈 Toplam {total} saldırı:")
                    for attack_type, cnt in top_types:
                        lines.append(f"  - {attack_type}: {cnt}")

                return "\n".join(lines)

        except Exception as e:
            logger.debug(f"Attack context error: {e}")
            return ""

    def _get_ids_context(self) -> str:
        """IDS durumu context'i"""
        try:
            from src.network_detection.realtime_ids import get_ids

            ids = get_ids()
            status = ids.get_status()

            if not status.get("is_running"):
                return ""

            lines = ["🛡️ **REAL-TIME IDS DURUMU:**"]
            lines.append(
                f"  - Durum: {'✅ Aktif' if status['is_running'] else '❌ Pasif'}"
            )
            lines.append(f"  - Model: {status.get('model_name', 'N/A')}")
            lines.append(f"  - Accuracy: %{status.get('model_accuracy', 0)*100:.1f}")

            if status.get("recent_alerts"):
                lines.append(f"  - Son {len(status['recent_alerts'])} alert mevcut")

            return "\n".join(lines)

        except Exception as e:
            logger.debug(f"IDS context error: {e}")
            return ""

    def detect_smart_action(self, message: str) -> dict | None:
        """
        Mesajdan smart action tespit et

        Returns:
            {"action": "action_name", "params": {...}} veya None
        """
        message_lower = message.lower()

        for keyword, action in self.SMART_ACTIONS.items():
            if keyword in message_lower:
                return {
                    "action": action,
                    "keyword": keyword,
                    "original_message": message,
                }

        return None

    def execute_smart_action(self, action: str, params: dict = None) -> str | None:
        """
        Smart action çalıştır ve sonuç döndür
        """
        params = params or {}

        try:
            if action == "analyze_ddos":
                return self._action_analyze_ddos()

            elif action == "analyze_portscan":
                return self._action_analyze_portscan()

            elif action == "compare_models":
                return self._action_compare_models()

            elif action == "recommend_model":
                return self._action_recommend_model()

            elif action == "ids_status":
                return self._action_ids_status()

            elif action == "mitre_mapping":
                return self._action_mitre_mapping()

            elif action == "generate_yara":
                # Bu AI'ın kendisi yapacak
                return None

            elif action == "check_ip":
                ip = params.get("ip")
                if ip:
                    return self._action_check_ip(ip)
                return None

        except Exception as e:
            logger.error(f"Smart action error: {e}")

        return None

    def _action_analyze_ddos(self) -> str:
        """DDoS analizi"""
        try:
            # DDoS model sonuçlarını al
            results_file = self.models_dir / "attack_specific_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)

                ddos = results.get("ddos", {})
                if ddos:
                    return f"""📊 **DDoS Model Analizi:**

| Metrik | Değer |
|--------|-------|
| Accuracy | %{ddos.get('accuracy', 0)*100:.2f} |
| Precision | %{ddos.get('precision', 0)*100:.2f} |
| Recall | %{ddos.get('recall', 0)*100:.2f} |
| F1-Score | %{ddos.get('f1_score', 0)*100:.2f} |
| Eğitim: {ddos.get('train_samples', 0):,} sample |"""

        except Exception as e:
            logger.error(f"DDoS analysis error: {e}")

        return ""

    def _action_analyze_portscan(self) -> str:
        """PortScan analizi"""
        try:
            results_file = self.models_dir / "attack_specific_results.json"
            if results_file.exists():
                with open(results_file) as f:
                    results = json.load(f)

                ps = results.get("portscan", {})
                if ps:
                    return f"""📊 **PortScan Model Analizi:**

| Metrik | Değer |
|--------|-------|
| Accuracy | %{ps.get('accuracy', 0)*100:.2f} |
| F1-Score | %{ps.get('f1_score', 0)*100:.2f} |"""

        except Exception:
            pass

        return ""

    def _action_compare_models(self) -> str:
        """Model karşılaştırması"""
        try:
            from src.chatbot.model_integration import get_integration

            return get_integration().get_model_comparison()
        except Exception:
            return ""

    def _action_recommend_model(self) -> str:
        """Model önerisi"""
        try:
            from src.chatbot.model_integration import get_integration

            integration = get_integration()

            if not integration.training_results:
                return "Henüz eğitilmiş model yok."

            # En iyi modeli bul
            best_name = None
            best_acc = 0

            for name, results in integration.training_results.items():
                if isinstance(results, dict):
                    acc = results.get("accuracy", 0)
                    if acc > best_acc:
                        best_acc = acc
                        best_name = name

            if best_name:
                return f"""🎯 **Önerilen Model:** {best_name}

**Neden?**
- En yüksek accuracy: %{best_acc*100:.2f}
- Makale mimarisi ile eğitildi (SSA-LSTMIDS)
- CICIDS2017 dataset ile validate edildi"""

        except Exception:
            pass

        return ""

    def _action_ids_status(self) -> str:
        """IDS durumu"""
        return self._get_ids_context()

    def _action_mitre_mapping(self) -> str:
        """MITRE ATT&CK mapping"""
        return """🎯 **MITRE ATT&CK Taktikleri:**

| Taktik | Teknik Sayısı |
|--------|---------------|
| Initial Access | 9 |
| Execution | 14 |
| Persistence | 19 |
| Privilege Escalation | 13 |
| Defense Evasion | 42 |
| Credential Access | 17 |
| Discovery | 31 |
| Lateral Movement | 9 |
| Collection | 17 |
| Command and Control | 16 |
| Exfiltration | 9 |
| Impact | 13 |"""

    def _action_check_ip(self, ip: str) -> str:
        """IP kontrol"""
        # Basit IP analizi
        return f"""🔍 **IP Analizi: {ip}**

⚠️ Detaylı analiz için IP Reputation API entegrasyonu gerekli.
Mevcut bilgiler veritabanından kontrol edilebilir."""


# Singleton
_context_builder: ContextBuilder | None = None


def get_context_builder() -> ContextBuilder:
    """Global context builder"""
    global _context_builder
    if _context_builder is None:
        _context_builder = ContextBuilder()
    return _context_builder


# Test
if __name__ == "__main__":
    print("🧪 Context Builder Test\n")

    builder = ContextBuilder()

    # Full context
    print("📋 Full Context:")
    context = builder.build_context()
    print(context if context else "(No context available)")

    # Smart action detection
    print("\n🎯 Smart Action Tests:")
    test_messages = [
        "DDoS analizi yap",
        "Model karşılaştır",
        "MITRE ATT&CK mapping göster",
        "Merhaba nasılsın",
    ]

    for msg in test_messages:
        action = builder.detect_smart_action(msg)
        if action:
            print(f"   ✅ '{msg}' → {action['action']}")
        else:
            print(f"   ❌ '{msg}' → No action")
