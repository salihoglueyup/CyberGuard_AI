"""
Training API - CyberGuard AI
Background model training with real-time progress tracking

Dosya Yolu: src/api/training_api.py
"""

import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime

from app.paths import PROJECT_ROOT

project_root = str(PROJECT_ROOT)

from src.utils import Logger


class TrainingSession:
    """Bir training session'ı temsil eder"""

    def __init__(self, session_id: str, config: dict):
        self.session_id = session_id
        self.config = config
        self.status = 'pending'  # pending, running, completed, failed
        self.progress = 0.0
        self.current_epoch = 0
        self.total_epochs = config.get('epochs', 100)
        self.logs = []
        self.result = None
        self.error = None
        self.start_time = None
        self.end_time = None
        self.model_id = None

    def add_log(self, message: str, level: str = 'info'):
        """Log ekle"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        }
        self.logs.append(log_entry)

    def update_progress(self, epoch: int, total_epochs: int, metrics: dict = None):
        """Progress güncelle"""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
        self.progress = (epoch / total_epochs) * 100

        if metrics:
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            self.add_log(f"Epoch {epoch}/{total_epochs} - {metric_str}", 'info')

    def to_dict(self) -> dict:
        """Dictionary'ye çevir"""
        return {
            'session_id': self.session_id,
            'status': self.status,
            'progress': round(self.progress, 2),
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'config': self.config,
            'logs': self.logs[-10:],  # Son 10 log
            'model_id': self.model_id,
            'error': self.error,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self._calculate_duration()
        }

    def _calculate_duration(self) -> str | None:
        """Süre hesapla"""
        if not self.start_time:
            return None

        end = datetime.fromisoformat(self.end_time) if self.end_time else datetime.now()
        start = datetime.fromisoformat(self.start_time)
        duration = end - start

        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingAPI:
    """Background model training API"""

    def __init__(self):
        self.logger = Logger("TrainingAPI")
        self.sessions: dict[str, TrainingSession] = {}
        self.active_threads: dict[str, threading.Thread] = {}
        self.logger.info("✅ Training API initialized")

    def start_training(
        self,
        model_name: str,
        description: str,
        config: dict,
        callback: Callable | None = None
    ) -> str:
        """Training başlat"""

        # Session ID oluştur
        session_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}"

        # Session oluştur
        session = TrainingSession(session_id, config)
        session.config['model_name'] = model_name
        session.config['description'] = description
        self.sessions[session_id] = session

        # Training thread başlat
        thread = threading.Thread(
            target=self._run_training,
            args=(session_id, callback),
            daemon=True
        )

        self.active_threads[session_id] = thread
        thread.start()

        self.logger.info(f"🚀 Training started: {session_id}")
        return session_id

    def _run_training(self, session_id: str, callback: Callable | None = None):
        """Training'i çalıştır (background thread)"""

        session = self.sessions[session_id]
        session.status = 'running'
        session.start_time = datetime.now().isoformat()
        session.add_log(f"Training started: {session.config['model_name']}", 'info')

        try:
            # Config'den parametreleri al
            config = session.config

            session.add_log("Starting training in subprocess...", 'info')

            # Training script'i subprocess'te çalıştır
            cmd = [
                sys.executable,  # Python executable
                'src/models/train_tensorflow_model.py',
                '--db', config.get('db_path', 'src/database/cyberguard.db'),
                '--name', config['model_name'],
                '--desc', config['description'],
                '--limit', str(config.get('data_limit', 50000)),
                '--epochs', str(config.get('epochs', 100)),
                '--batch-size', str(config.get('batch_size', 32)),
                '--random'
            ]

            # CRITICAL FIX: Binary mode kullan, sonra decode et
            import subprocess

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=project_root,
                # Binary mode - encoding sorunu yok
                text=False
            )

            # Output'u byte olarak oku ve güvenli decode et
            def safe_decode(byte_data):
                """Byte'ları güvenli şekilde decode et"""
                if not byte_data:
                    return ""
                try:
                    return byte_data.decode('utf-8', errors='ignore')
                except UnicodeDecodeError:
                    try:
                        return byte_data.decode('cp1254', errors='ignore')
                    except UnicodeDecodeError:
                        return byte_data.decode('latin-1', errors='ignore')

            # Output'u real-time oku
            while True:
                line_bytes = process.stdout.readline()

                if not line_bytes and process.poll() is not None:
                    break

                if line_bytes:
                    line = safe_decode(line_bytes).strip()

                    if line:
                        # ASCII-safe hale getir
                        line = ''.join(char if ord(char) < 128 else '?' for char in line)
                        session.add_log(line, 'info')

                        # Progress parse et
                        if 'Epoch' in line and '/' in line:
                            try:
                                parts = line.split()
                                for part in parts:
                                    if '/' in part and part.replace('/', '').isdigit():
                                        current, total = map(int, part.split('/'))
                                        session.update_progress(current, total)
                                        break
                            except ValueError:
                                pass

                        if callback:
                            callback(session.to_dict())

            # Process tamamlandı mı? (timeout: 2 saat)
            try:
                return_code = process.wait(timeout=7200)
            except subprocess.TimeoutExpired:
                process.kill()
                session.add_log("Eğitim zaman aşımına uğradı (2 saat)", "error")
                session.status = "failed"
                return_code = -1

            # Stderr merged into stdout via STDOUT redirect
            stderr_text = ""

            if return_code == 0:
                # Başarılı
                session.add_log("Training completed successfully", 'success')

                # Registry'den son eklenen modeli bul
                time.sleep(2)  # Registry'nin güncellenmesi için bekle

                from src.models.model_manager import ModelManager
                mm = ModelManager()
                models = mm.list_models()

                # En son oluşturulan model
                if models:
                    latest = max(models, key=lambda m: m.get('created_at', ''))
                    session.model_id = latest.get('id')
                    session.result = {'summary': latest.get('metrics', {})}

                    session.add_log(f"Model ID: {session.model_id[:30]}...", 'success')

                    # Metrics
                    metrics = latest.get('metrics', {})
                    if metrics:
                        session.add_log(f"Accuracy: {metrics.get('accuracy', 0)*100:.2f}%", 'success')
                        session.add_log(f"Precision: {metrics.get('precision', 0)*100:.2f}%", 'success')
                        session.add_log(f"Recall: {metrics.get('recall', 0)*100:.2f}%", 'success')
                        session.add_log(f"F1-Score: {metrics.get('f1_score', 0)*100:.2f}%", 'success')

                session.status = 'completed'
                session.end_time = datetime.now().isoformat()

                self.logger.info(f"Training completed: {session_id}")
            else:
                # Hata
                session.status = 'failed'

                # Stderr'dan hata mesajını al
                if stderr_text:
                    # ASCII-safe hale getir
                    error_msg = ''.join(char if ord(char) < 128 else '?' for char in stderr_text)
                    session.error = error_msg[:500]  # İlk 500 karakter
                    session.add_log(f"Training failed: {error_msg[:200]}", 'error')
                else:
                    session.error = f"Process exited with code {return_code}"
                    session.add_log(f"Training failed with code {return_code}", 'error')

                session.end_time = datetime.now().isoformat()

            if callback:
                callback(session.to_dict())

        except Exception as e:
            session.status = 'failed'
            session.error = str(e)
            session.end_time = datetime.now().isoformat()
            session.add_log(f"Training failed: {str(e)}", 'error')

            self.logger.error(f"Training failed: {session_id} - {e}")

            if callback:
                callback(session.to_dict())

            import traceback
            traceback.print_exc()

    def get_session_status(self, session_id: str) -> dict | None:
        """Session durumunu al"""
        session = self.sessions.get(session_id)
        return session.to_dict() if session else None

    def list_sessions(self) -> list:
        """Tüm session'ları listele"""
        return [s.to_dict() for s in self.sessions.values()]

    def stop_training(self, session_id: str) -> bool:
        """Training'i durdur (soft stop - thread bitse de durduramayız)"""
        session = self.sessions.get(session_id)
        if session and session.status == 'running':
            session.add_log("⚠️ Stop requested (will complete current epoch)", 'warning')
            # Not: Threading.Thread'i zorla durduramayız, sadece flag set edebiliriz
            return True
        return False

    def clear_old_sessions(self, keep_last: int = 10):
        """Eski session'ları temizle"""
        if len(self.sessions) > keep_last:
            sorted_sessions = sorted(
                self.sessions.items(),
                key=lambda x: x[1].start_time or '',
                reverse=True
            )

            # İlk keep_last'ı sakla
            self.sessions = dict(sorted_sessions[:keep_last])
            self.logger.info(f"🧹 Old sessions cleared, keeping {keep_last}")


# Singleton instance
_training_api_instance = None

def get_training_api() -> TrainingAPI:
    """Training API singleton instance"""
    global _training_api_instance
    if _training_api_instance is None:
        _training_api_instance = TrainingAPI()
    return _training_api_instance


# Test
if __name__ == "__main__":
    print("🧪 Training API Test\n" + "="*60)

    api = get_training_api()

    # Test config
    test_config = {
        'db_path': 'src/database/cyberguard.db',
        'data_limit': 1000,
        'epochs': 5,
        'batch_size': 32,
        'hidden_layers': [64, 32],
        'dropout_rate': 0.3,
        'learning_rate': 0.001
    }

    # Callback
    def progress_callback(status):
        print("\n📊 Progress Update:")
        print(f"  Status: {status['status']}")
        print(f"  Progress: {status['progress']:.1f}%")
        print(f"  Epoch: {status['current_epoch']}/{status['total_epochs']}")
        if status.get('logs'):
            print(f"  Last log: {status['logs'][-1]['message']}")

    # Training başlat
    print("\n🚀 Starting training...")
    session_id = api.start_training(
        model_name="TestModel",
        description="Test training",
        config=test_config,
        callback=progress_callback
    )

    print(f"✅ Session ID: {session_id}")

    # Progress monitoring
    print("\n⏳ Monitoring progress...")
    while True:
        time.sleep(5)
        status = api.get_session_status(session_id)

        if status['status'] in ['completed', 'failed']:
            print(f"\n✅ Training {status['status']}!")
            if status.get('model_id'):
                print(f"🎉 Model ID: {status['model_id']}")
            if status.get('error'):
                print(f"❌ Error: {status['error']}")
            break

    print("\n" + "="*60 + "\n✅ Test completed!")
