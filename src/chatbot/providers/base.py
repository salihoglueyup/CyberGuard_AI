"""
Base Provider - Abstract Interface
Tüm AI sağlayıcıları bu interface'i implement eder
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List


class BaseProvider(ABC):
    """
    Base AI Provider Interface

    Tüm provider'lar bu sınıftan türemeli
    """

    @abstractmethod
    def chat(
        self,
        message: str,
        context: Optional[str] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Kullanıcı mesajına yanıt ver

        Args:
            message: Kullanıcı mesajı
            context: Ek bağlam bilgisi
            system_prompt: Sistem promptu
            **kwargs: Ek parametreler

        Returns:
            AI yanıtı
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict:
        """Model bilgilerini döndür"""
        pass

    @classmethod
    @abstractmethod
    def list_models(cls) -> List[Dict]:
        """Mevcut modelleri listele"""
        pass

    def get_default_system_prompt(self) -> str:
        """Varsayılan sistem promptu"""
        return """Sen CyberGuard AI'ın siber güvenlik asistanısın.

Görevlerin:
1. Siber güvenlik sorularını yanıtla
2. Tehdit analizleri yap
3. Savunma önerileri sun
4. Saldırı verilerini yorumla
5. ML modelleri hakkında bilgi ver

Kurallar:
- Türkçe yanıt ver
- Teknik ama anlaşılır ol
- Somut öneriler sun
- Güvenlik odaklı ol
- Verilen context bilgilerini kullan"""
