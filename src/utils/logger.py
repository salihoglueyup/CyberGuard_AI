# src/utils/logger.py

"""
Logging sistemi
Tüm uygulama loglarını yönetir
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys


class Logger:
    """
    CyberGuard AI Logger

    Özellikleri:
    - Console ve dosyaya loglama
    - Farklı log seviyeleri (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - Otomatik log rotation (10MB'da yeni dosya)
    - Renkli console çıktısı
    - Her modül için ayrı logger
    """

    # ANSI renk kodları
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    # Emoji'ler
    EMOJIS = {
        'DEBUG': '🔍',
        'INFO': '✅',
        'WARNING': '⚠️',
        'ERROR': '❌',
        'CRITICAL': '🚨'
    }

    _instances = {}

    def __new__(cls, name: str = "CyberGuard"):
        """Singleton pattern - her modül için tek bir logger"""
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, name: str = "CyberGuard", log_dir: str = "data/logs"):
        """
        Logger'ı başlat

        Args:
            name (str): Logger adı (genelde modül adı)
            log_dir (str): Log dosyalarının kaydedileceği klasör
        """

        # Eğer zaten initialize edilmişse, tekrar yapma
        if hasattr(self, '_initialized'):
            return

        self._initialized = True
        self.name = name
        self.log_dir = log_dir

        # Log klasörünü oluştur
        os.makedirs(log_dir, exist_ok=True)

        # Logger oluştur
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        # Önceki handler'ları temizle
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Console handler ekle
        self._add_console_handler()

        # File handler ekle
        self._add_file_handler()

    def _add_console_handler(self):
        """Console'a renkli çıktı veren handler"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Renkli formatter
        console_formatter = self._ColoredFormatter(
            fmt='%(emoji)s %(color)s[%(levelname)s]%(reset)s %(name)s - %(message)s',
            colors=self.COLORS,
            emojis=self.EMOJIS
        )

        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """Dosyaya log yazan handler (rotation ile)"""
        log_file = os.path.join(
            self.log_dir,
            f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )

        # Rotating file handler (10MB'da yeni dosya, max 5 dosya)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)

        # Detaylı formatter
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    class _ColoredFormatter(logging.Formatter):
        """Renkli console formatter"""

        def __init__(self, fmt, colors, emojis):
            super().__init__(fmt)
            self.colors = colors
            self.emojis = emojis

        def format(self, record):
            # Renk ve emoji ekle
            record.color = self.colors.get(record.levelname, '')
            record.reset = self.colors['RESET']
            record.emoji = self.emojis.get(record.levelname, '')

            return super().format(record)

    # ========================================
    # PUBLIC METHODS
    # ========================================

    def debug(self, message: str, **kwargs):
        """Debug seviyesi log"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs):
        """Info seviyesi log"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Warning seviyesi log"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, exc_info=True, **kwargs):
        """Error seviyesi log (exception bilgisi ile)"""
        self.logger.error(message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info=True, **kwargs):
        """Critical seviyesi log"""
        self.logger.critical(message, exc_info=exc_info, **kwargs)

    def exception(self, message: str, **kwargs):
        """Exception log (otomatik traceback ile)"""
        self.logger.exception(message, **kwargs)

    # ========================================
    # SPECIAL METHODS
    # ========================================

    def log_function_call(self, func_name: str, args=None, kwargs=None):
        """Fonksiyon çağrısını logla"""
        args_str = f"args={args}" if args else ""
        kwargs_str = f"kwargs={kwargs}" if kwargs else ""
        self.debug(f"🔧 Function call: {func_name}({args_str} {kwargs_str})")

    def log_performance(self, operation: str, duration: float):
        """Performans metriğini logla"""
        self.info(f"⏱️  {operation} completed in {duration:.3f}s")

    def log_attack(self, attack_type: str, source_ip: str, severity: str):
        """Saldırı tespitini logla"""
        emoji = "🚨" if severity == "CRITICAL" else "⚠️" if severity == "HIGH" else "ℹ️"
        self.warning(f"{emoji} ATTACK DETECTED: {attack_type} from {source_ip} [Severity: {severity}]")

    def log_scan(self, filename: str, is_malware: bool, malware_type: str = None):
        """Dosya taramasını logla"""
        if is_malware:
            self.warning(f"🦠 MALWARE DETECTED: {filename} - Type: {malware_type}")
        else:
            self.info(f"✅ FILE CLEAN: {filename}")

    def log_user_action(self, action: str, user_id: str = "unknown"):
        """Kullanıcı aksiyonunu logla"""
        self.info(f"👤 User [{user_id}]: {action}")

    def log_model_prediction(self, model_name: str, input_shape, prediction, confidence: float):
        """Model tahminini logla"""
        self.debug(
            f"🤖 Model [{model_name}]: input_shape={input_shape}, prediction={prediction}, confidence={confidence:.3f}")

    def log_database_operation(self, operation: str, table: str, records: int):
        """Database operasyonunu logla"""
        self.debug(f"💾 DB: {operation} on {table} - {records} records")

    # ========================================
    # CONTEXT MANAGER
    # ========================================

    def log_block(self, block_name: str):
        """
        Bir kod bloğunu logla

        Kullanım:
            with logger.log_block("Data Processing"):
                # kod...
        """
        return self._LogBlock(self, block_name)

    class _LogBlock:
        """Context manager for logging code blocks"""

        def __init__(self, logger, block_name):
            self.logger = logger
            self.block_name = block_name
            self.start_time = None

        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.info(f"▶️  START: {self.block_name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = (datetime.now() - self.start_time).total_seconds()

            if exc_type is not None:
                self.logger.error(f"❌ FAILED: {self.block_name} ({duration:.3f}s) - {exc_val}")
                return False
            else:
                self.logger.info(f"✅ COMPLETED: {self.block_name} ({duration:.3f}s)")
                return True

    # ========================================
    # UTILITIES
    # ========================================

    def set_level(self, level: str):
        """
        Log seviyesini değiştir

        Args:
            level (str): DEBUG, INFO, WARNING, ERROR, CRITICAL
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if level.upper() in level_map:
            self.logger.setLevel(level_map[level.upper()])
            self.info(f"Log level changed to {level.upper()}")
        else:
            self.warning(f"Invalid log level: {level}")

    def get_log_file(self):
        """Güncel log dosyasının yolunu döndür"""
        return os.path.join(
            self.log_dir,
            f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
        )

    def clear_old_logs(self, days: int = 7):
        """Eski logları temizle"""
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days)

        for filename in os.listdir(self.log_dir):
            if filename.endswith('.log'):
                file_path = os.path.join(self.log_dir, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))

                if file_time < cutoff_date:
                    os.remove(file_path)
                    self.info(f"🗑️  Deleted old log: {filename}")


# ========================================
# DECORATOR
# ========================================

def log_execution(logger: Logger = None):
    """
    Fonksiyon çalışmasını otomatik logla

    Kullanım:
        @log_execution(logger)
        def my_function():
            pass
    """
    import functools
    import time

    if logger is None:
        logger = Logger()

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"🔧 Calling: {func_name}")

            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.debug(f"✅ {func_name} completed in {duration:.3f}s")
                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(f"❌ {func_name} failed after {duration:.3f}s: {str(e)}")
                raise

        return wrapper

    return decorator


# ========================================
# MODULE-LEVEL LOGGER
# ========================================

# Default logger instance
default_logger = Logger("CyberGuard")

# Convenience functions
debug = default_logger.debug
info = default_logger.info
warning = default_logger.warning
error = default_logger.error
critical = default_logger.critical
exception = default_logger.exception

# ========================================
# TEST
# ========================================

if __name__ == "__main__":
    print("🧪 Logger Test\n")
    print("=" * 60)

    # Logger oluştur
    logger = Logger("TestModule")

    # Farklı log seviyeleri
    logger.debug("Bu bir DEBUG mesajı")
    logger.info("Bu bir INFO mesajı")
    logger.warning("Bu bir WARNING mesajı")
    logger.error("Bu bir ERROR mesajı", exc_info=False)
    logger.critical("Bu bir CRITICAL mesajı", exc_info=False)

    print("\n" + "=" * 60)

    # Özel log fonksiyonları
    logger.log_attack("DDoS", "192.168.1.100", "HIGH")
    logger.log_scan("malware.exe", True, "Trojan")
    logger.log_scan("document.pdf", False)
    logger.log_user_action("Login successful", "user123")

    print("\n" + "=" * 60)

    # Context manager
    with logger.log_block("Data Processing"):
        import time

        time.sleep(1)
        logger.info("Processing data...")

    print("\n" + "=" * 60)


    # Decorator test
    @log_execution(logger)
    def test_function(x, y):
        """Test fonksiyonu"""
        import time
        time.sleep(0.5)
        return x + y


    result = test_function(5, 3)
    print(f"\nFonksiyon sonucu: {result}")

    print("\n" + "=" * 60)

    # Log dosyası
    log_file = logger.get_log_file()
    print(f"\n📄 Log dosyası: {log_file}")

    if os.path.exists(log_file):
        print("\n📋 Son 10 log satırı:")
        print("-" * 60)
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[-10:]:
                print(line.strip())

    print("\n" + "=" * 60)
    print("✅ Logger test tamamlandı!")