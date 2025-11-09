"""
Feature Extractor - CyberGuard AI
Saldırı verilerinden ML modeli için özellikler çıkarır
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, Tuple
import pickle
import os

class FeatureExtractor:
    """Saldırı verilerinden ML özellikleri çıkarır"""

    def __init__(self):
        """Initialize encoders"""
        self.attack_encoder = LabelEncoder()
        self.severity_encoder = LabelEncoder()
        self.status_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_fitted = False

    def ip_to_int(self, ip: str) -> int:
        """
        IP adresini integer'a çevir

        Args:
            ip (str): IP adresi (örn: "192.168.1.1")

        Returns:
            int: IP'nin integer karşılığı
        """
        try:
            parts = ip.split('.')
            return (int(parts[0]) * 256**3 +
                   int(parts[1]) * 256**2 +
                   int(parts[2]) * 256 +
                   int(parts[3]))
        except:
            return 0

    def extract_time_features(self, timestamp: pd.Series) -> pd.DataFrame:
        """
        Timestamp'ten zaman özellikleri çıkar

        Args:
            timestamp: Timestamp serisi

        Returns:
            DataFrame: Zaman özellikleri
        """
        timestamp = pd.to_datetime(timestamp)

        features = pd.DataFrame({
            'hour': timestamp.dt.hour,
            'day_of_week': timestamp.dt.dayofweek,
            'is_weekend': (timestamp.dt.dayofweek >= 5).astype(int),
            'is_night': ((timestamp.dt.hour >= 22) | (timestamp.dt.hour <= 6)).astype(int)
        })

        return features

    def prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        DataFrame'den ML özellikleri hazırla

        Args:
            df (DataFrame): Saldırı verileri
            fit (bool): Encoder'ları fit et mi?

        Returns:
            np.ndarray: Hazırlanmış özellikler
        """
        df = df.copy()

        # IP'leri daha basit şekilde işle
        # Son oktet'i kullan (192.168.1.XXX -> XXX)
        df['source_ip_last'] = df['source_ip'].apply(
            lambda x: int(x.split('.')[-1]) if isinstance(x, str) else 0
        )
        df['dest_ip_last'] = df['destination_ip'].apply(
            lambda x: int(x.split('.')[-1]) if isinstance(x, str) else 0
        )

        # Port değerlerini normalize et
        df['port'] = df['port'].fillna(0).astype(int)
        df['port_normalized'] = df['port'] / 10000.0  # Normalize

        # Blocked 0/1
        df['blocked'] = df['blocked'].astype(int)

        # Zaman özellikleri
        time_features = self.extract_time_features(df['timestamp'])
        df = pd.concat([df, time_features], axis=1)

        # Kategorik değerleri encode et
        if fit:
            df['severity_encoded'] = self.severity_encoder.fit_transform(df['severity'])
            df['status_encoded'] = self.status_encoder.fit_transform(df['status'])
        else:
            df['severity_encoded'] = self.severity_encoder.transform(df['severity'])
            df['status_encoded'] = self.status_encoder.transform(df['status'])

        # Özellik seç - DAHA BASIT
        feature_columns = [
            'source_ip_last',      # Son oktet
            'dest_ip_last',        # Son oktet
            'port_normalized',     # Normalize port
            'severity_encoded',    # Severity (0-3)
            'blocked',             # 0 veya 1
            'hour',                # Saat (0-23)
            'is_night',            # Gece mi? (0/1)
            'is_weekend'           # Hafta sonu mu? (0/1)
        ]

        X = df[feature_columns].values

        # Normalizasyon - DAHA HAFIF
        if fit:
            X = self.scaler.fit_transform(X)
            self.is_fitted = True
        else:
            X = self.scaler.transform(X)

        return X

    def prepare_labels(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """
        Etiketleri hazırla (saldırı türleri)

        Args:
            df (DataFrame): Saldırı verileri
            fit (bool): Encoder'ı fit et mi?

        Returns:
            np.ndarray: Etiketler
        """
        if fit:
            y = self.attack_encoder.fit_transform(df['attack_type'])
        else:
            y = self.attack_encoder.transform(df['attack_type'])

        return y

    def get_attack_type_name(self, encoded_label: int) -> str:
        """
        Encoded label'dan saldırı tipi ismini al

        Args:
            encoded_label (int): Encoded label

        Returns:
            str: Saldırı tipi ismi
        """
        return self.attack_encoder.inverse_transform([encoded_label])[0]

    def save(self, filepath: str = 'models/feature_extractor.pkl'):
        """
        Feature extractor'ı kaydet

        Args:
            filepath (str): Kayıt yolu
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump({
                'attack_encoder': self.attack_encoder,
                'severity_encoder': self.severity_encoder,
                'status_encoder': self.status_encoder,
                'scaler': self.scaler,
                'is_fitted': self.is_fitted
            }, f)

        print(f"✅ Feature extractor saved: {filepath}")

    def load(self, filepath: str = 'models/feature_extractor.pkl'):
        """
        Feature extractor'ı yükle

        Args:
            filepath (str): Dosya yolu
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.attack_encoder = data['attack_encoder']
        self.severity_encoder = data['severity_encoder']
        self.status_encoder = data['status_encoder']
        self.scaler = data['scaler']
        self.is_fitted = data['is_fitted']

        print(f"✅ Feature extractor loaded: {filepath}")


# Test
if __name__ == "__main__":
    import sqlite3

    # Database'den veri çek
    conn = sqlite3.connect('cyberguard.db')
    df = pd.read_sql_query("SELECT * FROM attacks LIMIT 10", conn)
    conn.close()

    # Feature extractor test
    extractor = FeatureExtractor()

    print("📊 Ham veri:")
    print(df[['attack_type', 'source_ip', 'port', 'severity']].head())

    print("\n🔄 Özellikler çıkarılıyor...")
    X = extractor.prepare_features(df, fit=True)
    y = extractor.prepare_labels(df, fit=True)

    print(f"\n✅ X shape: {X.shape}")
    print(f"✅ y shape: {y.shape}")
    print(f"✅ Saldırı türleri: {extractor.attack_encoder.classes_}")