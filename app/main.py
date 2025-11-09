# app/main.py

"""
CyberGuard AI - Ana Sayfa
Streamlit Web Uygulaması
"""

import streamlit as st
import sys
import os



# Path ayarı
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Pages klasörünü de ekle
pages_path = os.path.join(os.path.dirname(__file__), 'pages')
if pages_path not in sys.path:
    sys.path.insert(0, pages_path)

from src.utils import get_config, Logger, DatabaseManager

# ML Prediction import
try:
    from pages.ml_prediction import show_ml_prediction_page

    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML Prediction: {e}")

# Network Monitor import
try:
    from pages.network_monitor import show_network_monitor_page

    NETWORK_AVAILABLE = True
except ImportError as e:
    NETWORK_AVAILABLE = False
    print(f"⚠️ Network Monitor: {e}")

# Dashboard import
try:
    from pages.dashboard import show_dashboard_page

    DASHBOARD_AVAILABLE = True
except ImportError as e:
    DASHBOARD_AVAILABLE = False
    print(f"⚠️ Dashboard: {e}")

# Malware Scanner import
try:
    from pages.malware_scanner import show_malware_scanner_page

    MALWARE_AVAILABLE = True
except ImportError as e:
    MALWARE_AVAILABLE = False
    print(f"⚠️ Malware Scanner: {e}")

# Page config (HER ZAMAN İLK SATIRDA OLMALI!)
st.set_page_config(
    page_title="CyberGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }

    .metric-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }

    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }

    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# Initialize
@st.cache_resource
def init_app():
    """Uygulama başlatma"""
    config = get_config()
    logger = Logger("StreamlitApp")
    db = DatabaseManager(config.database_path)

    logger.info("🚀 Streamlit app initialized")

    return config, logger, db


config, logger, db = init_app()

# Header
st.markdown('<h1 class="main-header">🛡️ CyberGuard AI</h1>', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 1.2rem; color: gray;'>"
    "Yapay Zeka Destekli Siber Güvenlik Platformu"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# SIDEBAR MENÜ
st.sidebar.title("🧭 Navigasyon")
page = st.sidebar.radio(
    "Sayfa Seçin:",
    [
        "🏠 Ana Sayfa",
        "📊 Dashboard",
        "🔍 Network Monitor",
        "🦠 Malware Scanner",
        "🤖 AI Assistant",
        "🔮 ML Tahmin"
    ]
)

st.sidebar.markdown("---")

# Sayfa durumu
st.sidebar.info(f"📍 **Aktif Sayfa:** {page}")

# ============================================================
# SAYFA YÖNLENDİRME
# ============================================================

if page == "🏠 Ana Sayfa":
    # ANA SAYFA İÇERİĞİ

    # Ana içerik
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("## 👋 Hoş Geldiniz!")

        st.markdown("""
        **CyberGuard AI**, yapay zeka ve derin öğrenme teknolojileriyle 
        siber güvenlik tehditlerini tespit eden, analiz eden ve engelleyen 
        gelişmiş bir güvenlik platformudur.

        ### 🎯 Platform Özellikleri:

        - 🔍 **Network Saldırı Tespiti** - Gerçek zamanlı ağ trafiği analizi
        - 🦠 **Malware Taraması** - AI destekli zararlı yazılım tespiti  
        - 🤖 **AI Güvenlik Asistanı** - Gemini Pro ile akıllı danışman
        - 📊 **Gerçek Zamanlı Dashboard** - Anlık tehdit görselleştirme
        - 📈 **Detaylı Raporlama** - Kapsamlı güvenlik analizleri
        """)

        st.info("👈 Sol menüden istediğiniz sayfaya geçebilirsiniz!")

    with col2:
        st.markdown("## 📊 Sistem Durumu")

        # Database istatistikleri
        try:
            db_stats = db.get_database_stats()

            st.metric(
                label="📋 Toplam Saldırı",
                value=db_stats.get('attacks', 0),
                delta="Son 24 saat"
            )

            st.metric(
                label="📡 Network Logları",
                value=db_stats.get('network_logs', 0),
                delta="Aktif izleme"
            )

            st.metric(
                label="🔍 Tarama Sayısı",
                value=db_stats.get('scan_results', 0),
                delta="Güvenlik taramaları"
            )

            st.metric(
                label="💾 Database Boyutu",
                value=f"{db_stats.get('db_size_mb', 0)} MB",
                delta="Veri deposu"
            )

        except Exception as e:
            st.error(f"❌ Veri yüklenirken hata: {str(e)}")

    st.markdown("---")

    # Özellik kartları
    st.markdown("## 🚀 Platform Modülleri")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🏠 Dashboard</h3>
            <p>Gerçek zamanlı güvenlik durumu, saldırı istatistikleri ve sistem metrikleri</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🔍 Network Monitor</h3>
            <p>Ağ trafiği izleme, saldırı tespiti ve IP analizi</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>🦠 Malware Scanner</h3>
            <p>Dosya tarama, zararlı yazılım tespiti ve karantina</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 AI Assistant</h3>
            <p>Gemini Pro ile akıllı güvenlik danışmanı</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Hakkında
    with st.expander("ℹ️ CyberGuard AI Hakkında"):
        st.markdown("""
        ### 🛡️ Teknoloji Stack

        **Backend:**
        - Python 3.10+
        - TensorFlow / Keras (Deep Learning)
        - Google Gemini Pro (LLM)
        - SQLite (Database)

        **Frontend:**
        - Streamlit (Web Framework)
        - Plotly / Matplotlib (Visualizations)

        **AI Models:**
        - Random Forest - Network attack detection
        - CNN - Malware classification (yapım aşamasında)
        - Gemini Pro - Conversational AI

        ### 👨‍💻 Geliştirici

        **Proje:** CyberGuard AI - Yapay Zeka Destekli Siber Güvenlik Platformu

        **Versiyon:** 1.0.0

        **Lisans:** MIT
        """)

elif page == "📊 Dashboard":
    if DASHBOARD_AVAILABLE:
        show_dashboard_page()
    else:
        st.error("❌ Dashboard modülü yüklenemedi!")
        st.info("Dosya konumu: app/pages/dashboard.py")

elif page == "🔍 Network Monitor":
    if NETWORK_AVAILABLE:
        show_network_monitor_page()
    else:
        st.error("❌ Network Monitor modülü yüklenemedi!")
        st.info("Dosya konumu: app/pages/network_monitor.py")

elif page == "🦠 Malware Scanner":
    if MALWARE_AVAILABLE:
        show_malware_scanner_page()
    else:
        st.error("❌ Malware Scanner modülü yüklenemedi!")
        st.info("Dosya konumu: app/pages/malware_scanner.py")

elif page == "🤖 AI Assistant":
    st.title("🤖 AI Güvenlik Asistanı")
    st.info("🤖 AI Assistant sayfası yapım aşamasında...")
    st.markdown("""
    ### Gelecek Özellikler:
    - 💬 Gemini Pro entegrasyonu
    - 🧠 Context-aware yanıtlar
    - 📊 Veri analizi
    - 💡 Güvenlik önerileri
    """)

elif page == "🔮 ML Tahmin":
    if ML_AVAILABLE:
        show_ml_prediction_page()
    else:
        st.error("❌ ML Prediction modülü yüklenemedi!")
        st.warning("⚠️ Lütfen önce modeli eğitin:")
        st.code("python train_model.py", language="bash")

        st.info("""
        **Gerekli Adımlar:**
        1. Mock veri oluştur: `python src/utils/mock_data_generator.py`
        2. Modeli eğit: `python train_model.py`
        3. Uygulamayı yeniden başlat
        """)

# ============================================================
# FOOTER (TÜM SAYFALARDA)
# ============================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "© 2025 CyberGuard AI | Yapay Zeka Destekli Siber Güvenlik Platformu"
    "</p>",
    unsafe_allow_html=True
)