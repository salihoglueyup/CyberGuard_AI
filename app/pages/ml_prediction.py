"""
ML Prediction Page - CyberGuard AI Dashboard
Saldırı tahmini sayfası
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
import os

# Path ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.predictor import AttackPredictor


# Predictor'ı cache'le (her refresh'te yeniden yükleme)
@st.cache_resource
def load_predictor():
    """Predictor'ı yükle ve cache'le"""
    predictor = AttackPredictor()
    predictor.load_models()
    return predictor


def show_ml_prediction_page():
    """ML Tahmin sayfasını göster"""

    st.title("🤖 Saldırı Tahmin Sistemi")
    st.markdown("---")

    # Predictor yükle
    try:
        predictor = load_predictor()

        if not predictor.is_loaded:
            st.error("❌ Model yüklenemedi! Lütfen önce modeli eğitin.")
            st.code("python train_model.py")
            return

        # Model bilgisi sidebar'da
        with st.sidebar:
            st.subheader("📊 Model Bilgisi")
            model_info = predictor.get_model_info()
            st.metric("Model Türü", model_info.get('model_type', 'N/A'))
            st.metric("Sınıf Sayısı", model_info.get('n_classes', 0))
            st.metric("Özellik Sayısı", model_info.get('n_features', 0))

            with st.expander("Tespit Edilen Saldırı Türleri"):
                for attack_type in model_info.get('attack_types', []):
                    st.write(f"• {attack_type}")

        # Ana içerik
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎯 Yeni Saldırı Tahmini")

            # Input formu
            with st.form("prediction_form"):
                input_col1, input_col2 = st.columns(2)

                with input_col1:
                    source_ip = st.text_input(
                        "Kaynak IP",
                        value="192.168.1.105",
                        help="Saldırgan IP adresi"
                    )

                    destination_ip = st.text_input(
                        "Hedef IP",
                        value="192.168.0.10",
                        help="Hedef IP adresi"
                    )

                    port = st.number_input(
                        "Port",
                        min_value=1,
                        max_value=65535,
                        value=80,
                        help="Hedef port numarası"
                    )

                with input_col2:
                    severity = st.selectbox(
                        "Severity",
                        options=['low', 'medium', 'high', 'critical'],
                        index=2,
                        help="Tehdit seviyesi"
                    )

                    blocked = st.selectbox(
                        "Engellendi mi?",
                        options=[("Evet", 1), ("Hayır", 0)],
                        format_func=lambda x: x[0],
                        index=0
                    )[1]

                    timestamp = st.text_input(
                        "Zaman Damgası",
                        value=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        help="YYYY-MM-DD HH:MM:SS"
                    )

                submit_button = st.form_submit_button(
                    "🔮 Tahmin Yap",
                    use_container_width=True,
                    type="primary"
                )

        with col2:
            st.subheader("💡 Hızlı Test Örnekleri")

            if st.button("🌊 DDoS Saldırısı", use_container_width=True):
                st.session_state.test_data = {
                    'source_ip': '192.168.1.110',
                    'destination_ip': '192.168.0.5',
                    'port': 443,
                    'severity': 'critical',
                    'blocked': 1
                }
                st.rerun()

            if st.button("🔍 Port Scan", use_container_width=True):
                st.session_state.test_data = {
                    'source_ip': '192.168.1.125',
                    'destination_ip': '192.168.0.8',
                    'port': 22,
                    'severity': 'low',
                    'blocked': 0
                }
                st.rerun()

            if st.button("💉 SQL Injection", use_container_width=True):
                st.session_state.test_data = {
                    'source_ip': '192.168.1.145',
                    'destination_ip': '192.168.0.12',
                    'port': 3306,
                    'severity': 'high',
                    'blocked': 1
                }
                st.rerun()

            if st.button("🔐 Brute Force", use_container_width=True):
                st.session_state.test_data = {
                    'source_ip': '192.168.1.165',
                    'destination_ip': '192.168.0.15',
                    'port': 22,
                    'severity': 'medium',
                    'blocked': 0
                }
                st.rerun()

        # Tahmin yap
        if submit_button or 'test_data' in st.session_state:

            # Test data varsa onu kullan
            if 'test_data' in st.session_state:
                attack_data = st.session_state.test_data
                attack_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                del st.session_state.test_data
            else:
                attack_data = {
                    'source_ip': source_ip,
                    'destination_ip': destination_ip,
                    'port': port,
                    'severity': severity,
                    'blocked': blocked,
                    'timestamp': timestamp
                }

            # Tahmin yap
            with st.spinner("🔮 Model tahmin yapıyor..."):
                result = predictor.predict_single(attack_data)

            if 'error' in result:
                st.error(f"❌ Hata: {result['error']}")
            else:
                st.markdown("---")
                st.subheader("📊 Tahmin Sonuçları")

                # Ana sonuçlar
                result_col1, result_col2, result_col3 = st.columns(3)

                with result_col1:
                    st.metric(
                        "Tahmin Edilen Saldırı",
                        result['predicted_type'],
                        delta=None
                    )

                with result_col2:
                    confidence_pct = result['confidence'] * 100
                    st.metric(
                        "Güven Skoru",
                        f"{confidence_pct:.2f}%",
                        delta=None
                    )

                with result_col3:
                    risk_color = {
                        'low': '🟢',
                        'medium': '🟡',
                        'high': '🟠',
                        'critical': '🔴'
                    }
                    st.metric(
                        "Risk Seviyesi",
                        f"{risk_color.get(result['risk_level'], '⚪')} {result['risk_level'].upper()}",
                        delta=None
                    )

                # Olasılık grafiği
                st.subheader("📈 Olasılık Dağılımı")

                probs_df = pd.DataFrame(
                    list(result['probabilities'].items()),
                    columns=['Saldırı Türü', 'Olasılık']
                ).sort_values('Olasılık', ascending=True)

                fig = go.Figure(go.Bar(
                    x=probs_df['Olasılık'],
                    y=probs_df['Saldırı Türü'],
                    orientation='h',
                    marker=dict(
                        color=probs_df['Olasılık'],
                        colorscale='Reds',
                        showscale=False
                    ),
                    text=[f"{p * 100:.1f}%" for p in probs_df['Olasılık']],
                    textposition='auto'
                ))

                fig.update_layout(
                    title="Tüm Saldırı Türleri İçin Olasılıklar",
                    xaxis_title="Olasılık",
                    yaxis_title="Saldırı Türü",
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig, use_container_width=True)

                # Top 3 tahminler
                st.subheader("🏆 En Olası 3 Saldırı")

                top3_cols = st.columns(3)
                for idx, (attack_type, prob) in enumerate(result['top_3_predictions']):
                    with top3_cols[idx]:
                        st.info(f"**#{idx + 1}: {attack_type}**\n\n{prob * 100:.2f}%")

                # Detaylı bilgi
                with st.expander("🔍 Detaylı Bilgiler"):
                    st.json({
                        'Input Data': attack_data,
                        'Prediction Result': result
                    })

    except Exception as e:
        st.error(f"❌ Bir hata oluştu: {e}")
        import traceback
        st.code(traceback.format_exc())


# Test
if __name__ == "__main__":
    st.set_page_config(
        page_title="ML Prediction - CyberGuard AI",
        page_icon="🤖",
        layout="wide"
    )

    show_ml_prediction_page()