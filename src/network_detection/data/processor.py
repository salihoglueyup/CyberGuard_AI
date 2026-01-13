"""
Network Data Processor - CyberGuard AI
Ağ trafiği veri işleme ve özellik çıkarımı

Dosya Yolu: src/network_detection/data_processor.py
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import ipaddress


class NetworkDataProcessor:
    """
    Ağ trafiği veri işlemcisi
    
    Özellikler:
    - Paket özellik çıkarımı
    - Flow aggregation
    - IP analizi
    - Port analizi
    - Zaman serisi özellikleri
    """
    
    # Bilinen tehlikeli portlar
    DANGEROUS_PORTS = {
        21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 
        443, 445, 993, 995, 1433, 1521, 3306, 3389, 5432, 8080
    }
    
    # Protocol mapping
    PROTOCOL_MAP = {'TCP': 0, 'UDP': 1, 'ICMP': 2, 'OTHER': 3}
    
    def __init__(self):
        """Data Processor başlat"""
        self.feature_names = [
            'src_ip_numeric', 'dst_ip_numeric', 'src_port_norm', 'dst_port_norm',
            'protocol', 'packet_size_norm', 'is_dangerous_port', 'is_private_ip',
            'hour', 'is_night', 'is_weekend'
        ]
        print("🌐 Network Data Processor başlatıldı")
    
    def ip_to_numeric(self, ip: str) -> int:
        """IP adresini sayıya çevir"""
        try:
            return int(ipaddress.ip_address(ip))
        except:
            return 0
    
    def is_private_ip(self, ip: str) -> bool:
        """Private IP mi kontrol et"""
        try:
            return ipaddress.ip_address(ip).is_private
        except:
            return False
    
    def process_single_packet(self, packet: Dict) -> List[float]:
        """
        Tek paket işle
        
        Args:
            packet: Paket bilgileri dictionary
            
        Returns:
            Özellik vektörü
        """
        features = []
        
        # IP özellikleri
        src_ip = packet.get('source_ip', '0.0.0.0')
        dst_ip = packet.get('destination_ip', '0.0.0.0')
        
        features.append(self.ip_to_numeric(src_ip) / (2**32))  # Normalize
        features.append(self.ip_to_numeric(dst_ip) / (2**32))
        
        # Port özellikleri
        src_port = packet.get('source_port', 0)
        dst_port = packet.get('port', packet.get('destination_port', 0))
        
        features.append(src_port / 65535)  # Normalize
        features.append(dst_port / 65535)
        
        # Protocol
        protocol = packet.get('protocol', 'TCP')
        features.append(self.PROTOCOL_MAP.get(protocol.upper(), 3) / 3)
        
        # Packet size
        packet_size = packet.get('packet_size', 0)
        features.append(min(packet_size / 65535, 1.0))
        
        # Dangerous port
        features.append(1.0 if dst_port in self.DANGEROUS_PORTS else 0.0)
        
        # Private IP
        features.append(1.0 if self.is_private_ip(src_ip) else 0.0)
        
        # Zaman özellikleri
        timestamp = packet.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    dt = datetime.now()
            else:
                dt = timestamp
            
            features.append(dt.hour / 23)  # Hour normalized
            features.append(1.0 if dt.hour >= 22 or dt.hour <= 6 else 0.0)  # Night
            features.append(1.0 if dt.weekday() >= 5 else 0.0)  # Weekend
        else:
            features.extend([0.5, 0.0, 0.0])
        
        return features
    
    def process_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """
        DataFrame işle
        
        Args:
            df: Ağ trafiği DataFrame
            
        Returns:
            Özellik matrisi
        """
        print(f"🔄 {len(df)} paket işleniyor...")
        
        features = []
        
        for _, row in df.iterrows():
            packet = row.to_dict()
            features.append(self.process_single_packet(packet))
        
        X = np.array(features)
        print(f"✅ Özellik matrisi: {X.shape}")
        
        return X
    
    def extract_flow_features(self, packets: List[Dict]) -> Dict:
        """
        Flow-level özellikler çıkar
        
        Args:
            packets: Paket listesi
            
        Returns:
            Flow özellikleri
        """
        if not packets:
            return {}
        
        # Temel istatistikler
        sizes = [p.get('packet_size', 0) for p in packets]
        ports = [p.get('port', 0) for p in packets]
        
        features = {
            'packet_count': len(packets),
            'total_bytes': sum(sizes),
            'avg_packet_size': np.mean(sizes) if sizes else 0,
            'std_packet_size': np.std(sizes) if len(sizes) > 1 else 0,
            'unique_ports': len(set(ports)),
            'unique_src_ips': len(set(p.get('source_ip', '') for p in packets)),
            'unique_dst_ips': len(set(p.get('destination_ip', '') for p in packets))
        }
        
        # Zaman özellikleri
        timestamps = []
        for p in packets:
            ts = p.get('timestamp')
            if ts:
                try:
                    if isinstance(ts, str):
                        timestamps.append(datetime.fromisoformat(ts.replace('Z', '+00:00')))
                    else:
                        timestamps.append(ts)
                except:
                    pass
        
        if len(timestamps) >= 2:
            timestamps.sort()
            duration = (timestamps[-1] - timestamps[0]).total_seconds()
            features['flow_duration'] = duration
            features['packets_per_second'] = len(packets) / duration if duration > 0 else 0
        
        return features
    
    def detect_port_scan(self, packets: List[Dict], threshold: int = 10) -> bool:
        """Port scan tespiti"""
        src_ips = {}
        for p in packets:
            src = p.get('source_ip', '')
            port = p.get('port', 0)
            if src not in src_ips:
                src_ips[src] = set()
            src_ips[src].add(port)
        
        # Bir IP'den çok fazla farklı port erişimi
        for ip, ports in src_ips.items():
            if len(ports) >= threshold:
                return True
        return False
    
    def detect_ddos_pattern(self, packets: List[Dict], threshold: int = 100) -> bool:
        """DDoS pattern tespiti"""
        dst_ips = {}
        for p in packets:
            dst = p.get('destination_ip', '')
            dst_ips[dst] = dst_ips.get(dst, 0) + 1
        
        # Bir hedefe çok fazla paket
        for ip, count in dst_ips.items():
            if count >= threshold:
                return True
        return False


# Test
if __name__ == "__main__":
    print("🧪 Network Data Processor Test\n")
    
    processor = NetworkDataProcessor()
    
    # Test paketi
    packet = {
        'source_ip': '192.168.1.100',
        'destination_ip': '8.8.8.8',
        'port': 443,
        'protocol': 'TCP',
        'packet_size': 1024,
        'timestamp': datetime.now().isoformat()
    }
    
    features = processor.process_single_packet(packet)
    print(f"📊 Özellikler: {features}")
    print(f"   Boyut: {len(features)}")
