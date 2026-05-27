"""
Model Manager - CyberGuard AI
ML modellerini yönetir (load, save, list, deploy)

Dosya Yolu: src/models/model_manager.py
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path


class ModelManager:
    """
    ML Model yönetim sistemi

    Özellikler:
    - Model kayıt/yükleme
    - Versiyonlama
    - Deploy/archive/delete
    - Model listesi
    - Metadata yönetimi
    """

    def __init__(self, base_dir: str = "model_artifacts"):
        """
        Args:
            base_dir: Model klasörü
        """
        # Proje root dizinini bul
        current_file = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

        # Base directory'yi proje root'a göre ayarla
        if not os.path.isabs(base_dir):
            self.base_dir = os.path.join(project_root, base_dir)
        else:
            self.base_dir = base_dir

        self.registry_file = os.path.join(self.base_dir, "model_registry.json")

        # Base directory oluştur (yoksa)
        os.makedirs(self.base_dir, exist_ok=True)

        # Registry yükle veya oluştur
        self.registry = self._load_registry()

        print("📦 Model Manager başlatıldı")
        print(f"📁 Base directory: {self.base_dir}")
        print(f"📄 Registry file: {self.registry_file}")

    # ========================================
    # REGISTRY OPERATIONS
    # ========================================

    def _load_registry(self) -> dict:
        """Registry dosyasını yükle"""

        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, encoding='utf-8') as f:
                    registry = json.load(f)

                # Format kontrolü - models dict ise list'e çevir
                if isinstance(registry.get('models'), dict):
                    print("⚠️ Registry format düzeltiliyor...")
                    models_dict = registry['models']
                    models_list = []

                    for model_id, model_data in models_dict.items():
                        # ID'yi ekle (yoksa)
                        if 'id' not in model_data:
                            model_data['id'] = model_data.get('model_id', model_id)

                        # Name'i ekle (yoksa)
                        if 'name' not in model_data:
                            model_data['name'] = model_data.get('model_name', model_id)

                        models_list.append(model_data)

                    registry['models'] = models_list

                    # Düzeltilmiş registry'yi kaydet
                    self.registry = registry
                    self._save_registry()
                    print("✅ Registry formatı düzeltildi!")

                return registry

            except Exception as e:
                print(f"⚠️ Registry yüklenemedi: {e}")
                return {'models': [], 'last_updated': None}
        else:
            # Yeni registry oluştur
            return {'models': [], 'last_updated': None}

    def _save_registry(self) -> bool:
        """Registry'yi kaydet"""

        try:
            self.registry['last_updated'] = datetime.now().isoformat()

            # Atomic write: write to temp file first, then replace
            tmp_file = self.registry_file + '.tmp'
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
            os.replace(tmp_file, self.registry_file)

            return True

        except Exception as e:
            print(f"❌ Registry kaydedilemedi: {e}")
            return False

    # ========================================
    # TRAINING INTEGRATION METHODS
    # ========================================

    def generate_model_id(self, model_name: str, model_type: str) -> str:
        """
        Unique model ID oluştur

        Args:
            model_name: Model adı
            model_type: Model tipi

        Returns:
            str: Model ID
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # UUID benzeri kısa hash
        import hashlib
        hash_part = hashlib.md5(f"{model_name}{timestamp}".encode()).hexdigest()[:8]

        return f"{model_type}_{model_name}_{timestamp}_{hash_part}"

    def create_model_directory(self, model_id: str, model_info: dict) -> Path:
        """
        Model klasörü oluştur

        Args:
            model_id: Model ID
            model_info: Model bilgileri

        Returns:
            Path: Model klasör yolu
        """
        model_dir = Path(self.base_dir) / model_id

        # Ana klasör
        model_dir.mkdir(parents=True, exist_ok=True)

        # Alt klasörler
        (model_dir / 'artifacts').mkdir(exist_ok=True)
        (model_dir / 'checkpoints').mkdir(exist_ok=True)
        (model_dir / 'logs').mkdir(exist_ok=True)
        (model_dir / 'evaluation').mkdir(exist_ok=True)
        (model_dir / 'plots').mkdir(exist_ok=True)

        # Metadata kaydet
        metadata = {
            'id': model_id,
            'name': model_info.get('name', 'Unknown'),
            'description': model_info.get('description', ''),
            'type': model_info.get('type', 'unknown'),
            'framework': model_info.get('framework', 'unknown'),
            'created_at': datetime.now().isoformat(),
            'status': 'training'
        }

        with open(model_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f"📁 Model klasörü oluşturuldu: {model_dir}")

        return model_dir

    def register_model(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        framework: str,
        hyperparameters: dict,
        training_config: dict,
        tags: list[str] = None,
        description: str = None
    ) -> bool:
        """
        Model kaydet (training için)

        Args:
            model_id: Model ID
            model_name: Model adı
            model_type: Model tipi
            framework: Framework (tensorflow, sklearn, etc.)
            hyperparameters: Hyperparameter'lar
            training_config: Training config
            tags: Etiketler
            description: Açıklama

        Returns:
            bool: Başarılı mı?
        """
        model_info = {
            'id': model_id,
            'model_id': model_id,  # Backward compatibility
            'name': model_name,
            'model_name': model_name,  # Backward compatibility
            'model_type': model_type,
            'framework': framework,
            'version': 1,
            'status': 'training',
            'hyperparameters': hyperparameters,
            'training_config': training_config,
            'tags': tags or [],
            'description': description or '',
            'created_at': datetime.now().isoformat(),
            'directory': model_id,  # Relative path
            'metrics': {}
        }

        # Mevcut model var mı?
        existing = self.get_model_info(model_id)

        if existing:
            # Güncelle
            for i, model in enumerate(self.registry['models']):
                if model.get('id') == model_id:
                    self.registry['models'][i] = model_info
                    break
        else:
            # Yeni ekle
            self.registry['models'].append(model_info)

        # Kaydet
        self._save_registry()

        print(f"✅ Model kaydedildi: {model_name}")

        return True

    def update_model_metrics(self, model_id: str, metrics: dict) -> bool:
        """
        Model metriklerini güncelle

        Args:
            model_id: Model ID
            metrics: Metrikler

        Returns:
            bool: Başarılı mı?
        """
        model_info = self.get_model_info(model_id)

        if not model_info:
            print(f"❌ Model bulunamadı: {model_id}")
            return False

        # Metrikleri güncelle
        model_info['metrics'] = metrics
        model_info['updated_at'] = datetime.now().isoformat()

        # Registry'ye kaydet
        for i, model in enumerate(self.registry['models']):
            if model.get('id') == model_id:
                self.registry['models'][i] = model_info
                break

        self._save_registry()

        return True

    # ========================================
    # MODEL OPERATIONS
    # ========================================

    def get_model_info(self, model_id: str) -> dict | None:
        """
        Model bilgilerini getir

        Args:
            model_id: Model ID

        Returns:
            Dict: Model bilgileri veya None
        """

        for model in self.registry['models']:
            if model.get('id') == model_id or model.get('model_id') == model_id:
                return model

        return None

    def load_model(self, model_id: str) -> dict:
        """
        Model yükle (dosya yolunu döndür)

        Args:
            model_id: Model ID

        Returns:
            Dict: Model bilgileri (model_path dahil)
        """

        model_info = self.get_model_info(model_id)

        if not model_info:
            raise ValueError(f"Model bulunamadı: {model_id}")

        # Model path'i al
        model_path = model_info.get('model_path')

        # Path mutlak mı değil mi kontrol et
        if model_path and not os.path.isabs(model_path):
            # Relative path ise, base_dir ile birleştir
            model_path = os.path.join(self.base_dir, model_path)

        # Model dosyası var mı kontrol et
        if model_path and os.path.exists(model_path):
            model_info['model_path'] = model_path
            return model_info

        # Model path yoksa veya dosya yoksa, model klasöründe ara
        model_dir = model_info.get('directory')

        # Directory zaten base_dir içeriyor mu kontrol et
        if model_dir:
            # Eğer directory "models\" ile başlıyorsa, sadece model_id kısmını al
            if model_dir.startswith('models\\') or model_dir.startswith('models/'):
                # models\ kısmını çıkar
                model_dir = model_dir.replace('models\\', '').replace('models/', '')

            # Şimdi base_dir ile birleştir
            if not os.path.isabs(model_dir):
                model_dir = os.path.join(self.base_dir, model_dir)
        else:
            # Directory yoksa model_id kullan
            model_dir = os.path.join(self.base_dir, model_id)

        # Olası dosya konumları (klasör + dosya adı)
        potential_locations = [
            # Ana klasörde
            ('', 'model.pkl'),
            ('', 'model.h5'),
            ('', 'model.keras'),
            ('', 'model.joblib'),
            ('', 'best_model.h5'),
            ('', 'final_model.h5'),
            # Artifacts klasöründe
            ('artifacts', 'model.pkl'),
            ('artifacts', 'model.h5'),
            ('artifacts', 'model.keras'),
            ('artifacts', 'best_model.h5'),
            # Checkpoints klasöründe
            ('checkpoints', 'best_model.h5'),
            ('checkpoints', 'model.h5'),
            ('checkpoints', 'final_model.h5'),
        ]

        # Önce metadata'dan model adını al
        model_name = model_info.get('name', model_info.get('model_name', ''))
        if model_name:
            # Model adıyla eşleşen dosyaları da dene
            potential_locations.extend([
                ('artifacts', f'{model_name}.h5'),
                ('artifacts', f'{model_name}.pkl'),
                ('artifacts', f'{model_name}.keras'),
                ('checkpoints', f'{model_name}.h5'),
                ('', f'{model_name}.h5'),
                ('', f'{model_name}.pkl'),
            ])

        # Her lokasyonu dene
        for subdir, filename in potential_locations:
            if subdir:
                path = os.path.join(model_dir, subdir, filename)
            else:
                path = os.path.join(model_dir, filename)

            if os.path.exists(path):
                model_path = path
                model_info['model_path'] = path
                print(f"✅ Model dosyası bulundu: {os.path.join(subdir, filename) if subdir else filename}")
                return model_info

        # Hiçbir dosya bulunamadı
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_id}")

    def list_models(self, status: str | None = None) -> list[dict]:
        """
        Modelleri listele

        Args:
            status: Durum filtresi (None ise tümü)

        Returns:
            List[Dict]: Model listesi
        """

        # Registry'yi yeniden yükle (güncel bilgiler için)
        self.registry = self._load_registry()

        models = self.registry.get('models', [])

        # Eğer models list değilse veya boşsa, klasör taraması yap
        if not models or not isinstance(models, list):
            print("⚠️ Registry boş, klasör taranıyor...")
            self.scan_model_directory()
            self.registry = self._load_registry()
            models = self.registry.get('models', [])

        # Status filtresi
        if status:
            models = [m for m in models if isinstance(m, dict) and m.get('status') == status]

        # Sadece dict olanları döndür (string olanları filtrele)
        models = [m for m in models if isinstance(m, dict)]

        return models

    def update_model_status(self, model_id: str, status: str) -> bool:
        """
        Model durumunu güncelle

        Args:
            model_id: Model ID
            status: Yeni durum (training/deployed/archived)

        Returns:
            bool: Başarılı mı?
        """

        model_info = self.get_model_info(model_id)

        if not model_info:
            print(f"❌ Model bulunamadı: {model_id}")
            return False

        # Status güncelle
        model_info['status'] = status
        model_info['updated_at'] = datetime.now().isoformat()

        # Registry'de güncelle
        for i, model in enumerate(self.registry['models']):
            if model.get('id') == model_id:
                self.registry['models'][i] = model_info
                break

        return self._save_registry()

    def deploy_model(self, model_id: str) -> bool:
        """
        Modeli deploy et

        Args:
            model_id: Model ID

        Returns:
            bool: Başarılı mı?
        """

        return self.update_model_status(model_id, 'deployed')

    def archive_model(self, model_id: str) -> bool:
        """
        Modeli arşivle

        Args:
            model_id: Model ID

        Returns:
            bool: Başarılı mı?
        """

        return self.update_model_status(model_id, 'archived')

    def delete_model(self, model_id: str, remove_files: bool = False) -> bool:
        """
        Modeli sil

        Args:
            model_id: Model ID
            remove_files: Dosyaları da sil

        Returns:
            bool: Başarılı mı?
        """

        try:
            model_info = self.get_model_info(model_id)

            if not model_info:
                print(f"❌ Model bulunamadı: {model_id}")
                return False

            # Dosyaları sil
            if remove_files:
                model_dir = os.path.join(self.base_dir, model_info.get('directory', model_id))

                if os.path.exists(model_dir):
                    shutil.rmtree(model_dir)
                    print(f"🗑️ Dosyalar silindi: {model_dir}")

            # Registry'den sil
            self.registry['models'] = [
                m for m in self.registry['models']
                if m.get('id') != model_id
            ]

            self._save_registry()

            print(f"✅ Model silindi: {model_info.get('name', model_id)}")

            return True

        except Exception as e:
            print(f"❌ Model silinemedi: {e}")
            return False

    # ========================================
    # SEARCH & FILTER
    # ========================================

    def search_models(self, query: str) -> list[dict]:
        """Model ara (isim, açıklama, tip)"""

        query_lower = query.lower()
        results = []

        for model in self.registry['models']:
            if (query_lower in model.get('name', '').lower() or
                query_lower in model.get('description', '').lower() or
                query_lower in model.get('model_type', '').lower()):
                results.append(model)

        return results

    def get_best_model(self, metric: str = 'accuracy') -> dict | None:
        """En iyi modeli bul"""

        models = self.list_models()

        if not models:
            return None

        # Metriği olan modelleri filtrele
        valid_models = [
            m for m in models
            if m.get('metrics', {}).get(metric) is not None
        ]

        if not valid_models:
            return None

        # En yüksek metriğe sahip model
        best = max(valid_models, key=lambda m: m.get('metrics', {}).get(metric, 0))

        return best

    # ========================================
    # STATISTICS
    # ========================================

    def get_stats(self) -> dict:
        """Model istatistikleri"""

        models = self.registry['models']

        stats = {
            'total_models': len(models),
            'by_status': {},
            'by_type': {},
            'best_accuracy': 0,
            'average_accuracy': 0
        }

        # Status dağılımı
        for model in models:
            status = model.get('status', 'unknown')
            stats['by_status'][status] = stats['by_status'].get(status, 0) + 1

            model_type = model.get('model_type', 'unknown')
            stats['by_type'][model_type] = stats['by_type'].get(model_type, 0) + 1

        # Accuracy istatistikleri
        accuracies = [
            m.get('metrics', {}).get('accuracy', 0)
            for m in models
            if m.get('metrics', {}).get('accuracy') is not None
        ]

        if accuracies:
            stats['best_accuracy'] = max(accuracies)
            stats['average_accuracy'] = sum(accuracies) / len(accuracies)

        return stats

    # ========================================
    # SCAN & SYNC
    # ========================================

    def scan_model_directory(self) -> int:
        """
        Model klasörünü tara ve registry ile senkronize et

        Returns:
            int: Eklenen model sayısı
        """

        added_count = 0

        try:
            # Model klasörlerini tara
            for item in os.listdir(self.base_dir):
                item_path = os.path.join(self.base_dir, item)

                # Klasör mü?
                if not os.path.isdir(item_path):
                    continue

                # Model ID çıkar (klasör adından)
                model_id = item

                # Registry'de var mı?
                if self.get_model_info(model_id):
                    continue

                # Metadata dosyası var mı?
                metadata_path = os.path.join(item_path, 'metadata.json')

                if os.path.exists(metadata_path):
                    try:
                        with open(metadata_path, encoding='utf-8') as f:
                            metadata = json.load(f)

                        # Eksik bilgileri tamamla
                        if 'id' not in metadata:
                            metadata['id'] = model_id

                        if 'name' not in metadata:
                            metadata['name'] = model_id.replace('_', ' ').title()

                        if 'status' not in metadata:
                            metadata['status'] = 'deployed'

                        # Directory ekle
                        metadata['directory'] = model_id

                        # Registry'ye ekle
                        self.registry['models'].append(metadata)
                        self._save_registry()
                        added_count += 1

                    except Exception as e:
                        print(f"⚠️ Metadata okunamadı ({item}): {e}")
                else:
                    # Metadata yoksa, basit bir kayıt oluştur
                    print(f"⚠️ Metadata bulunamadı, otomatik oluşturuluyor: {item}")

                    metadata = {
                        'id': model_id,
                        'name': model_id.replace('_', ' ').title(),
                        'model_type': 'unknown',
                        'version': 1,
                        'status': 'deployed',
                        'directory': model_id,
                        'metrics': {},
                        'created_at': datetime.now().isoformat(),
                        'description': f'Auto-imported model: {model_id}'
                    }

                    self.registry['models'].append(metadata)
                    self._save_registry()
                    added_count += 1

            if added_count > 0:
                print(f"✅ {added_count} model registry'ye eklendi")

            return added_count

        except Exception as e:
            print(f"❌ Tarama hatası: {e}")
            import traceback
            traceback.print_exc()
            return 0


# Test
if __name__ == "__main__":
    print("🧪 Model Manager Test\n")
    print("=" * 60)

    mm = ModelManager()

    # Klasörü tara
    print("\n🔍 Model klasörü taranıyor...")
    added = mm.scan_model_directory()
    print(f"Eklenen: {added}")

    # Modelleri listele
    print("\n📋 Mevcut Modeller:")
    models = mm.list_models()

    for model in models:
        print(f"\n• {model.get('name', 'Unknown')}")
        print(f"  ID: {model.get('id', 'unknown')}")
        print(f"  Type: {model.get('model_type', 'unknown')}")
        print(f"  Status: {model.get('status', 'unknown')}")

        metrics = model.get('metrics', {})
        if metrics:
            print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")

    # İstatistikler
    print("\n📊 İstatistikler:")
    stats = mm.get_stats()
    print(f"Toplam Model: {stats['total_models']}")
    print(f"Status Dağılımı: {stats['by_status']}")
    print(f"En İyi Accuracy: {stats['best_accuracy']:.2%}")

    # En iyi model
    best = mm.get_best_model()
    if best:
        print(f"\n🏆 En İyi Model: {best['name']}")
        print(f"   Accuracy: {best.get('metrics', {}).get('accuracy', 0):.2%}")

    print("\n" + "=" * 60)
    print("✅ Test tamamlandı!")
