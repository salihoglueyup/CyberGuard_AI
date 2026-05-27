# 🌐 CyberGuard AI - WebSocket Rehberi

Bu dokümanda CyberGuard AI'ın WebSocket API'sini kullanarak gerçek zamanlı veri akışına nasıl bağlanacağınızı öğrenebilirsiniz.

---

## 📋 İçindekiler

1. [WebSocket Endpoint'leri](#websocket-endpointleri)
2. [Bağlantı Kurma](#bağlantı-kurma)
3. [Mesaj Formatları](#mesaj-formatları)
4. [Örnek Kodlar](#örnek-kodlar)
5. [Hata Yönetimi](#hata-yönetimi)

---

## 🔌 WebSocket Endpoint'leri

| Endpoint | Açıklama | Veri Tipi |
| -------- | -------- | --------- |
| `ws://localhost:8000/ws` | Sistem metrikleri | CPU, RAM, Disk |
| `ws://localhost:8000/ws/attacks` | Saldırı akışı | Attack + ML Prediction |
| `ws://localhost:8000/ws/events` | Olay aboneliği | Özelleştirilebilir |
| `ws://localhost:8000/ws/security` | Güvenlik metrikleri | Aktif bağlantılar |

---

## 🔗 Bağlantı Kurma

### JavaScript (Tarayıcı)

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/attacks');

ws.onopen = () => {
    console.log('✅ WebSocket bağlantısı kuruldu');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log('Mesaj:', data);
};

ws.onerror = (error) => {
    console.error('❌ WebSocket hatası:', error);
};

ws.onclose = () => {
    console.log('🔌 WebSocket bağlantısı kapandı');
};
```

### Python

```python
import asyncio
import websockets
import json

async def connect_to_attacks():
    uri = "ws://localhost:8000/ws/attacks"
    
    async with websockets.connect(uri) as websocket:
        print("✅ Bağlantı kuruldu")
        
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Mesaj: {data}")

asyncio.run(connect_to_attacks())
```

---

## 📨 Mesaj Formatları

### Saldırı Akışı (`/ws/attacks`)

**Bağlantı Mesajı:**

```json
{
    "type": "connected",
    "message": "Connected to attack stream",
    "ml_enabled": true,
    "geoip_enabled": true
}
```

**Saldırı Mesajı:**

```json
{
    "type": "attack",
    "data": {
        "id": "ATK-10042",
        "source": {
            "country": "CN",
            "ip": "185.220.101.1",
            "lat": 35.86,
            "lng": 104.19
        },
        "target": {
            "country": "TR",
            "ip": "192.168.1.100",
            "lat": 39.0,
            "lng": 35.0
        },
        "attack_type": "DDoS",
        "severity": "high",
        "ml_prediction": {
            "is_threat": true,
            "confidence": 0.92,
            "severity": "high",
            "suggested_action": "block"
        }
    },
    "timestamp": "2026-01-13T10:30:00.000Z"
}
```

**Heartbeat:**

```json
{
    "type": "heartbeat"
}
```

### Sistem Metrikleri (`/ws`)

```json
{
    "type": "metrics",
    "data": {
        "cpu_percent": 45.2,
        "memory_percent": 62.5,
        "disk_percent": 35.8,
        "network": {
            "bytes_sent": 1234567890,
            "bytes_recv": 9876543210
        },
        "timestamp": "2026-01-13T10:30:00.000Z"
    }
}
```

---

## 💻 Örnek Kodlar

### React Hook

```javascript
import { useState, useEffect, useRef } from 'react';

function useWebSocket(url) {
    const [messages, setMessages] = useState([]);
    const [connected, setConnected] = useState(false);
    const wsRef = useRef(null);

    useEffect(() => {
        const ws = new WebSocket(url);
        wsRef.current = ws;

        ws.onopen = () => setConnected(true);
        ws.onclose = () => {
            setConnected(false);
            // Auto-reconnect
            setTimeout(() => {
                wsRef.current = new WebSocket(url);
            }, 3000);
        };
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setMessages(prev => [data, ...prev].slice(0, 100));
        };

        return () => ws.close();
    }, [url]);

    const send = (data) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(data));
        }
    };

    return { messages, connected, send };
}

// Kullanım
function AttackMonitor() {
    const { messages, connected } = useWebSocket('ws://localhost:8000/ws/attacks');

    return (
        <div>
            <p>Durum: {connected ? '🟢 Bağlı' : '🔴 Bağlı Değil'}</p>
            <ul>
                {messages.map((msg, i) => (
                    <li key={i}>{JSON.stringify(msg)}</li>
                ))}
            </ul>
        </div>
    );
}
```

### Python Async Client

```python
import asyncio
import websockets
import json
from datetime import datetime

class AttackMonitor:
    def __init__(self, url="ws://localhost:8000/ws/attacks"):
        self.url = url
        self.attacks = []
        self.connected = False
    
    async def connect(self):
        while True:
            try:
                async with websockets.connect(self.url) as ws:
                    self.connected = True
                    print(f"✅ [{datetime.now()}] Bağlantı kuruldu")
                    
                    async for message in ws:
                        await self.handle_message(json.loads(message))
                        
            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                print(f"🔌 Bağlantı koptu, yeniden bağlanılıyor...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"❌ Hata: {e}")
                await asyncio.sleep(5)
    
    async def handle_message(self, data):
        msg_type = data.get("type")
        
        if msg_type == "attack":
            attack = data.get("data", {})
            self.attacks.append(attack)
            
            # Tehdit analizi
            ml = attack.get("ml_prediction", {})
            if ml.get("is_threat") and ml.get("confidence", 0) > 0.8:
                print(f"⚠️ YÜKSEK TEHDİT!")
                print(f"   Kaynak: {attack.get('source', {}).get('ip')}")
                print(f"   Tip: {attack.get('attack_type')}")
                print(f"   Güven: {ml.get('confidence'):.1%}")
        
        elif msg_type == "heartbeat":
            # Ping gönder
            pass

# Çalıştır
async def main():
    monitor = AttackMonitor()
    await monitor.connect()

asyncio.run(main())
```

### Node.js Client

```javascript
const WebSocket = require('ws');

class AttackClient {
    constructor(url = 'ws://localhost:8000/ws/attacks') {
        this.url = url;
        this.ws = null;
        this.reconnectInterval = 3000;
    }

    connect() {
        this.ws = new WebSocket(this.url);

        this.ws.on('open', () => {
            console.log('✅ Bağlantı kuruldu');
        });

        this.ws.on('message', (data) => {
            const message = JSON.parse(data);
            this.handleMessage(message);
        });

        this.ws.on('close', () => {
            console.log('🔌 Bağlantı kapandı, yeniden bağlanılıyor...');
            setTimeout(() => this.connect(), this.reconnectInterval);
        });

        this.ws.on('error', (error) => {
            console.error('❌ Hata:', error.message);
        });
    }

    handleMessage(message) {
        switch (message.type) {
            case 'attack':
                const attack = message.data;
                const ml = attack.ml_prediction || {};
                
                if (ml.is_threat && ml.confidence > 0.8) {
                    console.log(`⚠️ YÜKSEK TEHDİT: ${attack.source?.ip} -> ${attack.target?.ip}`);
                    console.log(`   Tip: ${attack.attack_type}, Güven: ${(ml.confidence * 100).toFixed(0)}%`);
                }
                break;
            
            case 'heartbeat':
                this.ws.send(JSON.stringify({ type: 'ping' }));
                break;
        }
    }
}

const client = new AttackClient();
client.connect();
```

---

## ⚠️ Hata Yönetimi

### Bağlantı Kopması

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/attacks');
let reconnectAttempts = 0;
const maxReconnectAttempts = 5;

ws.onclose = () => {
    if (reconnectAttempts < maxReconnectAttempts) {
        reconnectAttempts++;
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
        console.log(`Yeniden bağlanma denemesi ${reconnectAttempts}/${maxReconnectAttempts} (${delay}ms)`);
        setTimeout(connect, delay);
    } else {
        console.error('Maksimum deneme sayısına ulaşıldı');
    }
};

ws.onopen = () => {
    reconnectAttempts = 0; // Başarılı bağlantıda sıfırla
};
```

### Heartbeat Kontrolü

```javascript
let heartbeatTimeout;

function resetHeartbeat() {
    clearTimeout(heartbeatTimeout);
    heartbeatTimeout = setTimeout(() => {
        console.warn('Heartbeat timeout, bağlantı kontrol ediliyor...');
        ws.close();
    }, 45000); // 45 saniye
}

ws.onmessage = (event) => {
    resetHeartbeat();
    // ... mesaj işleme
};
```

---

## 📊 Globe3D Entegrasyonu

Globe3D bileşeni otomatik olarak `/ws/attacks` endpoint'ine bağlanır:

```javascript
// Globe3D.jsx içinde
useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/attacks');
    
    ws.onmessage = (event) => {
        const message = JSON.parse(event.data);
        
        if (message.type === 'attack') {
            // Saldırıyı haritaya ekle
            setWsAttacks(prev => [message.data, ...prev].slice(0, 50));
            
            // ML tahmini yüksekse ses çal
            if (message.data.ml_prediction?.confidence > 0.85) {
                playAlertSound();
            }
        }
    };
    
    return () => ws.close();
}, []);
```

---

## 🔒 Güvenlik Notları

1. **Production'da wss:// kullanın** (SSL/TLS)
2. **Token tabanlı kimlik doğrulama** ekleyin
3. **Rate limiting** uygulayın
4. **Input validation** yapın

```javascript
// Güvenli bağlantı örneği
const ws = new WebSocket('wss://your-domain.com/ws/attacks', {
    headers: {
        'Authorization': `Bearer ${token}`
    }
});
```

---

**⚡ Gerçek zamanlı güvenlik izleme!**
