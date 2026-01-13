# src/utils/config.py

"""
Configuration Manager
config.yaml dosyasını okur ve ayarları yönetir
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path


class Config:
    """
    CyberGuard AI Configuration Manager

    Özellikleri:
    - YAML config dosyasını okuma
    - .env dosyasından environment variables (manuel parse)
    - Nested dict'lere kolay erişim (config.get('app.name'))
    - Default değerler
    - Type safety
    - Singleton pattern
    """

    _instance = None
    _config_data = None
    _env_vars = {}
    _env_loaded = False

    def __new__(cls, config_file: str = "config.yaml"):
        """Singleton pattern - tek bir config instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config_file: str = "config.yaml"):
        """
        Config manager'ı başlat

        Args:
            config_file (str): YAML config dosyasının yolu
        """

        # Eğer zaten yüklenmişse tekrar yükleme
        if self._config_data is not None:
            return

        self.config_file = config_file

        # .env dosyasını yükle
        if not self._env_loaded:
            self._load_env_file()
            self._env_loaded = True

        # Config dosyasını yükle
        self._load_config()

    def _load_env_file(self, env_file: str = ".env"):
        """
        .env dosyasını manuel olarak oku ve parse et

        Args:
            env_file (str): .env dosyasının yolu
        """

        if not os.path.exists(env_file):
            print(f"⚠️  Warning: {env_file} not found, using system environment variables only")
            return

        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # Boş satırları ve yorumları atla
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # KEY=VALUE formatını parse et
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Tırnak işaretlerini temizle
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        # Environment variable'a ekle
                        self._env_vars[key] = value

                        # Sistem environment'a da ekle (opsiyonel)
                        os.environ[key] = value

            print(f"✅ Environment variables loaded from {env_file}")

        except Exception as e:
            print(f"⚠️  Warning: Error loading {env_file}: {e}")

    def _load_config(self):
        """YAML config dosyasını yükle"""

        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)

            print(f"✅ Config loaded: {self.config_file}")

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Config değerini getir (nested key desteği)

        Args:
            key (str): Config key (örn: 'app.name' veya 'database.type')
            default (Any): Bulunamazsa döndürülecek değer

        Returns:
            Any: Config değeri

        Örnek:
            config.get('app.name')  # "CyberGuard AI"
            config.get('network_detection.model.type')  # "LSTM"
        """

        keys = key.split('.')
        value = self._config_data

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def get_env(self, key: str, default: Any = None) -> Any:
        """
        Environment variable'ı getir

        Args:
            key (str): Environment variable adı
            default (Any): Bulunamazsa döndürülecek değer

        Returns:
            Any: Environment variable değeri
        """
        # Önce manuel yüklenen env vars'a bak
        if key in self._env_vars:
            return self._env_vars[key]

        # Sonra sistem environment'a bak
        return os.environ.get(key, default)

    def set(self, key: str, value: Any):
        """
        Config değerini runtime'da değiştir

        Args:
            key (str): Config key
            value (Any): Yeni değer

        Not: Bu değişiklik sadece runtime'da geçerli, dosyaya yazılmaz
        """

        keys = key.split('.')
        config = self._config_data

        # Son key'e kadar git
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Değeri set et
        config[keys[-1]] = value

    def reload(self):
        """Config dosyasını yeniden yükle"""
        self._config_data = None
        self._load_config()

    # ========================================
    # CONVENIENCE PROPERTIES
    # ========================================

    @property
    def app_name(self) -> str:
        """Uygulama adı"""
        return self.get('app.name', 'CyberGuard AI')

    @property
    def app_version(self) -> str:
        """Uygulama versiyonu"""
        return self.get('app.version', '1.0.0')

    @property
    def debug_mode(self) -> bool:
        """Debug mode aktif mi?"""
        debug_str = self.get_env('DEBUG', 'False')
        return debug_str.lower() in ('true', '1', 'yes')

    @property
    def log_level(self) -> str:
        """Log seviyesi"""
        return self.get('logging.level', 'INFO')

    @property
    def database_url(self) -> str:
        """Database URL"""
        return self.get_env('DATABASE_URL', 'sqlite:///src/database/cyberguard.db')

    @property
    def database_path(self) -> str:
        """SQLite database dosya yolu"""
        db_url = self.database_url
        if db_url.startswith('sqlite:///'):
            return db_url.replace('sqlite:///', '')
        return self.get('database.path', 'src/database/cyberguard.db')

    @property
    def gemini_api_key(self) -> Optional[str]:
        """Google Gemini API Key"""
        return self.get_env('GOOGLE_API_KEY')

    # ========================================
    # MODEL CONFIGS
    # ========================================

    def get_network_model_config(self) -> Dict:
        """Network detection model config"""
        return {
            'model_path': self.get_env('NETWORK_MODEL_PATH',
                                       self.get('network_detection.model_path',
                                                'models/network_detector/lstm_model.h5')),
            'sequence_length': self.get('network_detection.model.sequence_length', 10),
            'features': self.get('network_detection.model.features', 78),
            'num_classes': self.get('network_detection.model.num_classes', 15),
            'batch_size': self.get('network_detection.training.batch_size', 128),
            'epochs': self.get('network_detection.training.epochs', 50),
            'learning_rate': self.get('network_detection.training.learning_rate', 0.001),
            'threshold_normal': self.get('network_detection.thresholds.normal', 0.85),
            'threshold_suspicious': self.get('network_detection.thresholds.suspicious', 0.70),
            'threshold_malicious': self.get('network_detection.thresholds.malicious', 0.50),
        }

    def get_malware_model_config(self) -> Dict:
        """Malware detection model config"""
        return {
            'model_path': self.get_env('MALWARE_MODEL_PATH',
                                       self.get('malware_detection.model_path',
                                                'models/malware_classifier/cnn_model.h5')),
            'input_shape': self.get('malware_detection.model.input_shape', [224, 224, 3]),
            'num_classes': self.get('malware_detection.model.num_classes', 9),
            'batch_size': self.get('malware_detection.training.batch_size', 32),
            'epochs': self.get('malware_detection.training.epochs', 30),
            'learning_rate': self.get('malware_detection.training.learning_rate', 0.0001),
            'malware_threshold': self.get('malware_detection.thresholds.malware_threshold', 0.90),
            'max_file_size_mb': self.get('malware_detection.scan_settings.max_file_size_mb', 100),
            'timeout_seconds': self.get('malware_detection.scan_settings.timeout_seconds', 30),
        }

    def get_chatbot_config(self) -> Dict:
        """Chatbot config"""
        return {
            'model_name': self.get('chatbot.model.name', 'gemini-pro'),
            'temperature': self.get('chatbot.model.temperature', 0.3),
            'max_tokens': self.get('chatbot.model.max_output_tokens', 1024),
            'top_p': self.get('chatbot.model.top_p', 0.95),
            'top_k': self.get('chatbot.model.top_k', 40),
            'vector_store': self.get('chatbot.rag.vector_store', 'chromadb'),
            'embedding_model': self.get('chatbot.rag.embedding_model', 'models/embedding-001'),
            'chunk_size': self.get('chatbot.rag.chunk_size', 500),
            'chunk_overlap': self.get('chatbot.rag.chunk_overlap', 50),
            'retrieval_k': self.get('chatbot.rag.retrieval_k', 3),
            'max_history': self.get('chatbot.memory.max_history', 10),
            'save_to_database': self.get('chatbot.memory.save_to_database', True),
        }

    # ========================================
    # DATABASE CONFIGS
    # ========================================

    def get_database_config(self) -> Dict:
        """Database config"""
        return {
            'type': self.get('database.type', 'sqlite'),
            'path': self.database_path,
            'retention_attacks_days': self.get('database.retention.attacks_days', 90),
            'retention_logs_days': self.get('database.retention.logs_days', 30),
            'retention_chat_days': self.get('database.retention.chat_history_days', 30),
            'max_attacks': self.get('database.max_records.attacks', 100000),
            'max_logs': self.get('database.max_records.logs', 500000),
        }

    # ========================================
    # SECURITY CONFIGS
    # ========================================

    def get_security_config(self) -> Dict:
        """Security config"""
        return {
            'rate_limiting_enabled': self.get('security.rate_limiting.enabled', True),
            'max_requests_per_minute': self.get('security.rate_limiting.max_requests_per_minute', 60),
            'ip_blacklist_enabled': self.get('security.ip_blacklist.enabled', True),
            'auto_blacklist_threshold': self.get('security.ip_blacklist.auto_blacklist_threshold', 10),
            'max_concurrent_scans': self.get('security.file_scanning.max_concurrent_scans', 5),
            'quarantine_path': self.get('security.file_scanning.quarantine_path', 'data/quarantine/'),
        }

    # ========================================
    # MONITORING CONFIGS
    # ========================================

    def get_monitoring_config(self) -> Dict:
        """Monitoring config"""
        return {
            'real_time_analysis': self.get('monitoring.real_time_analysis', True),
            'update_interval_seconds': self.get('monitoring.update_interval_seconds', 5),
            'alerts_enabled': self.get('monitoring.alerts.enabled', True),
            'save_interval_minutes': self.get('monitoring.metrics.save_interval_minutes', 5),
            'retention_hours': self.get('monitoring.metrics.retention_hours', 168),
        }

    # ========================================
    # VISUALIZATION CONFIGS
    # ========================================

    def get_visualization_config(self) -> Dict:
        """Visualization config"""
        return {
            'refresh_rate_seconds': self.get('visualization.dashboard.refresh_rate_seconds', 10),
            'max_points_on_chart': self.get('visualization.dashboard.max_points_on_chart', 100),
            'default_zoom': self.get('visualization.maps.default_zoom', 2),
            'marker_cluster': self.get('visualization.maps.marker_cluster', True),
            'theme': self.get('visualization.charts.theme', 'plotly_dark'),
            'color_scheme': self.get('visualization.charts.color_scheme', 'viridis'),
        }

    # ========================================
    # DATA SOURCE CONFIGS
    # ========================================

    def get_data_sources(self) -> Dict:
        """Data sources config"""
        return {
            'network_dataset': {
                'name': self.get('data.network_dataset.name', 'CICIDS2017'),
                'url': self.get('data.network_dataset.url', ''),
                'local_path': self.get('data.network_dataset.local_path', 'data/raw/CICIDS2017/'),
            },
            'malware_dataset': {
                'name': self.get('data.malware_dataset.name', 'MalImg'),
                'url': self.get('data.malware_dataset.url', ''),
                'local_path': self.get('data.malware_dataset.local_path', 'data/raw/MalImg/'),
            },
            'security_docs': {
                'path': self.get('data.security_docs.path', 'data/docs/'),
                'formats': self.get('data.security_docs.formats', ['.txt', '.pdf', '.md']),
            }
        }

    # ========================================
    # UTILITIES
    # ========================================

    def validate(self) -> tuple[bool, list]:
        """
        Config'i validate et

        Returns:
            tuple: (is_valid, error_list)
        """
        errors = []

        # Gerekli environment variable'ları kontrol et
        if not self.gemini_api_key:
            errors.append("GOOGLE_API_KEY not found in .env file")

        # Config dosyası değerlerini kontrol et
        required_keys = [
            'app.name',
            'app.version',
            'database.type',
            'network_detection.model.type',
            'malware_detection.model.type',
            'chatbot.model.name',
        ]

        for key in required_keys:
            if self.get(key) is None:
                errors.append(f"Required config key missing: {key}")

        # Klasörlerin varlığını kontrol et
        required_dirs = [
            'data/logs',
            'models/network_detector',
            'models/malware_classifier',
            'models/chatbot',
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                errors.append(f"Required directory missing: {dir_path}")

        return (len(errors) == 0, errors)

    def print_config(self, section: Optional[str] = None):
        """
        Config'i pretty print et

        Args:
            section (str, optional): Sadece belirli bir section'ı göster
        """
        import json

        if section:
            data = self.get(section)
            if data is None:
                print(f"❌ Section not found: {section}")
                return
            print(f"\n📋 Config Section: {section}")
        else:
            data = self._config_data
            print("\n📋 Full Configuration")

        print("=" * 60)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("=" * 60)

    def export_to_dict(self) -> Dict:
        """Tüm config'i dict olarak export et"""
        return self._config_data.copy()

    def create_directories(self):
        """Config'de belirtilen tüm klasörleri oluştur"""
        directories = [
            'data/logs',
            'data/raw',
            'data/processed',
            'data/quarantine',
            'models/network_detector',
            'models/malware_classifier',
            'models/chatbot/vectorstore',
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")


# ========================================
# SINGLETON INSTANCE
# ========================================

# Global config instance
_config_instance = None


def get_config(config_file: str = "config.yaml") -> Config:
    """
    Config instance'ını getir (singleton)

    Args:
        config_file (str): Config dosya yolu

    Returns:
        Config: Config instance
    """
    global _config_instance

    if _config_instance is None:
        _config_instance = Config(config_file)

    return _config_instance


# ========================================
# TEST
# ========================================

if __name__ == "__main__":
    print("🧪 Config Manager Test\n")
    print("=" * 60)

    try:
        # Config'i yükle
        config = Config("config.yaml")

        # Temel bilgiler
        print("\n📱 Application Info:")
        print(f"  Name:    {config.app_name}")
        print(f"  Version: {config.app_version}")
        print(f"  Debug:   {config.debug_mode}")

        # Database
        print("\n💾 Database:")
        print(f"  Type: {config.get('database.type')}")
        print(f"  Path: {config.database_path}")

        # Models
        print("\n🤖 Models:")
        network_config = config.get_network_model_config()
        print(f"  Network Detection:")
        print(f"    - Type: {config.get('network_detection.model.type')}")
        print(f"    - Sequence Length: {network_config['sequence_length']}")
        print(f"    - Features: {network_config['features']}")
        print(f"    - Classes: {network_config['num_classes']}")

        malware_config = config.get_malware_model_config()
        print(f"  Malware Classification:")
        print(f"    - Type: {config.get('malware_detection.model.base_model')}")
        print(f"    - Input Shape: {malware_config['input_shape']}")
        print(f"    - Classes: {malware_config['num_classes']}")

        # Chatbot
        print("\n💬 Chatbot:")
        chatbot_config = config.get_chatbot_config()
        print(f"  Model: {chatbot_config['model_name']}")
        print(f"  Temperature: {chatbot_config['temperature']}")
        print(f"  Max Tokens: {chatbot_config['max_tokens']}")

        # Environment variables
        print("\n🔐 Environment Variables:")
        print(f"  GOOGLE_API_KEY: {'✅ Set' if config.gemini_api_key else '❌ Not set'}")
        print(f"  DATABASE_URL: {config.database_url}")

        # Validation
        print("\n✅ Validation:")
        is_valid, errors = config.validate()

        if is_valid:
            print("  ✅ All checks passed!")
        else:
            print("  ⚠️  Validation warnings:")
            for error in errors:
                print(f"     - {error}")

        # Nested key access test
        print("\n🔍 Nested Key Access Test:")
        test_keys = [
            'app.name',
            'network_detection.model.type',
            'chatbot.rag.chunk_size',
            'monitoring.update_interval_seconds',
            'non.existent.key',
        ]

        for key in test_keys:
            value = config.get(key, "NOT FOUND")
            print(f"  {key}: {value}")

        # Create directories
        print("\n📁 Creating Directories:")
        config.create_directories()

        print("\n" + "=" * 60)
        print("✅ Config test tamamlandı!")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Tip: config.yaml dosyasının proje kök dizininde olduğundan emin olun")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()