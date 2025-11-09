# app/pages/dashboard.py

"""
Dashboard - CyberGuard AI
Genel bakış ve güvenlik metrikleri
"""

import os
from typing import List, Dict
from datetime import datetime, timedelta
import sqlite3
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import sys
import time

# Path ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# PDF Generator import
try:
    from src.utils.pdf_generator import PDFReportGenerator

    PDF_AVAILABLE = True
except:
    PDF_AVAILABLE = False
    print("⚠️ PDF Generator yüklenemedi")


class DashboardData:
    """Dashboard veri yönetimi"""

    def __init__(self, db_path: str = 'cyberguard.db'):
        self.db_path = db_path

    def get_overview_stats(self) -> dict:
        """Genel istatistikler"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Toplam saldırı
        cursor.execute("SELECT COUNT(*) FROM attacks")
        total_attacks = cursor.fetchone()[0]

        # Son 24 saat
        cursor.execute("""
                       SELECT COUNT(*)
                       FROM attacks
                       WHERE timestamp >= datetime('now', '-24 hours')
                       """)
        last_24h = cursor.fetchone()[0]

        # Önceki 24 saat (karşılaştırma için)
        cursor.execute("""
                       SELECT COUNT(*)
                       FROM attacks
                       WHERE timestamp >= datetime('now'
                           , '-48 hours')
                         AND timestamp
                           < datetime('now'
                           , '-24 hours')
                       """)
        prev_24h = cursor.fetchone()[0]

        # Değişim yüzdesi
        change_24h = ((last_24h - prev_24h) / prev_24h * 100) if prev_24h > 0 else 0

        # Engellenen
        cursor.execute("""
                       SELECT COUNT(*)FROM attacks 
            WHERE blocked = 1 AND timestamp >= datetime('now', '-24 hours')
        """)
        blocked_24h = cursor.fetchone()[0]

        # Kritik
        cursor.execute("""
                       SELECT COUNT(*)
                       FROM attacks
                       WHERE severity = 'critical' AND timestamp >= datetime('now', '-24 hours')
                       """)
        critical_24h = cursor.fetchone()[0]

        # Benzersiz IP sayısı
        cursor.execute("""
            SELECT COUNT(DISTINCT source_ip)FROM attacks
            WHERE timestamp >= datetime('now', '-24 hours')
        """)
        unique_ips = cursor.fetchone()[0]

        # Ortalama saldırı/saat
        cursor.execute("""
                       SELECT COUNT(*) / 24.0
                       FROM attacks
                       WHERE timestamp >= datetime('now', '-24 hours')
                       """)
        avg_per_hour = cursor.fetchone()[0]

        conn.close()

        return {
            'total_attacks': total_attacks,
            'last_24h': last_24h,
            'change_24h': change_24h,
            'blocked_24h': blocked_24h,
            'critical_24h': critical_24h,
            'unique_ips': unique_ips,
            'avg_per_hour': avg_per_hour,
            'block_rate': (blocked_24h / last_24h * 100) if last_24h > 0 else 0
        }

    def get_hourly_trend(self, hours: int = 24) -> pd.DataFrame:
        """Saatlik trend"""
        conn = sqlite3.connect(self.db_path)

        query = f"""
        SELECT 
            strftime('%H:00', timestamp) as hour,
            COUNT(*) as total,
            SUM(CASE WHEN blocked = 1 THEN 1 ELSE 0 END) as blocked,
            SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical
        FROM attacks 
        WHERE timestamp >= datetime('now', '-{hours} hours')
        GROUP BY hour 
        ORDER BY hour
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_attack_distribution(self) -> pd.DataFrame:
        """Saldırı dağılımı"""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT attack_type, \
                       COUNT(*) as count,
            SUM(CASE WHEN blocked = 1 THEN 1 ELSE 0 END) as blocked,
            SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical
                FROM attacks
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY attack_type
                ORDER BY count DESC \
                """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_severity_trend(self) -> pd.DataFrame:
        """Severity trendi (son 7 gün)"""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT
                    DATE (timestamp) as date, severity, COUNT (*) as count
                FROM attacks
                WHERE timestamp >= datetime('now', '-7 days')
                GROUP BY date, severity
                ORDER BY date, severity \
                """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_top_attackers(self, limit: int = 5) -> pd.DataFrame:
        """En aktif saldırganlar"""
        conn = sqlite3.connect(self.db_path)

        query = f"""
        SELECT 
            source_ip,
            COUNT(*) as attacks,
            SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical,
            MAX(timestamp) as last_seen
        FROM attacks 
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY source_ip 
        ORDER BY attacks DESC 
        LIMIT {limit}
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_port_stats(self, limit: int = 10) -> pd.DataFrame:
        """Port istatistikleri"""
        conn = sqlite3.connect(self.db_path)

        query = f"""
        SELECT 
            port,
            COUNT(*) as count,
            GROUP_CONCAT(DISTINCT attack_type) as attack_types
        FROM attacks 
        WHERE timestamp >= datetime('now', '-24 hours')
        GROUP BY port 
        ORDER BY count DESC 
        LIMIT {limit}
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df


def show_dashboard_page():
    """Dashboard sayfasını göster"""

    st.title("📊 Security Dashboard")
    st.markdown("Gerçek zamanlı güvenlik durumu ve tehdit analizi")

    # Auto-refresh toggle
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("---")

    with col2:
        auto_refresh = st.toggle("🔄 Otomatik Yenileme", value=False)

    with col3:
        if auto_refresh:
            refresh_interval = st.selectbox(
                "Süre",
                options=[5, 10, 30, 60],
                format_func=lambda x: f"{x}s",
                index=1
            )
        else:
            refresh_interval = 10

    st.markdown("---")

    # Auto-refresh logic
    if auto_refresh:
        # Countdown ve refresh
        placeholder = st.empty()

        for remaining in range(refresh_interval, 0, -1):
            placeholder.info(f"🔄 Yenileniyor: {remaining} saniye...")
            time.sleep(1)

        placeholder.empty()
        st.rerun()

    # Son güncelleme zamanı
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Son Güncelleme")
    st.sidebar.info(datetime.now().strftime('%H:%M:%S'))

    try:
        dashboard = DashboardData()
        stats = dashboard.get_overview_stats()

        # ============================================================
        # KPI KARTLARI
        # ============================================================

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                "🎯 Toplam Saldırı",
                f"{stats['total_attacks']:,}",
                delta=None
            )

        with col2:
            delta_color = "inverse" if stats['change_24h'] > 0 else "normal"
            st.metric(
                "⏱️ Son 24 Saat",
                f"{stats['last_24h']:,}",
                delta=f"{stats['change_24h']:+.1f}%",
                delta_color=delta_color
            )

        with col3:
            st.metric(
                "🛡️ Engelleme Oranı",
                f"{stats['block_rate']:.1f}%",
                delta=f"{stats['blocked_24h']} engellendi"
            )

        with col4:
            st.metric(
                "🚨 Kritik Tehdit",
                f"{stats['critical_24h']:,}",
                delta="Yüksek Öncelik",
                delta_color="inverse"
            )

        with col5:
            st.metric(
                "🌐 Benzersiz IP",
                f"{stats['unique_ips']:,}",
                delta=f"~{stats['avg_per_hour']:.1f} saldırı/saat"
            )

        st.markdown("---")

        # ============================================================
        # ANA GRAFİKLER
        # ============================================================

        col1, col2 = st.columns([2, 1])

        # SOL: Saatlik Trend
        with col1:
            st.subheader("📈 Son 24 Saat Aktivite")

            df_trend = dashboard.get_hourly_trend(hours=24)

            if len(df_trend) > 0:
                fig = go.Figure()

                # Toplam saldırı
                fig.add_trace(go.Scatter(
                    x=df_trend['hour'],
                    y=df_trend['total'],
                    mode='lines+markers',
                    name='Toplam',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))

                # Engellenen
                fig.add_trace(go.Scatter(
                    x=df_trend['hour'],
                    y=df_trend['blocked'],
                    mode='lines+markers',
                    name='Engellenen',
                    line=dict(color='#00C851', width=2),
                    marker=dict(size=6)
                ))

                # Kritik
                fig.add_trace(go.Scatter(
                    x=df_trend['hour'],
                    y=df_trend['critical'],
                    mode='lines+markers',
                    name='Kritik',
                    line=dict(color='#ff4444', width=2),
                    marker=dict(size=6)
                ))

                fig.update_layout(
                    height=400,
                    xaxis_title="Saat",
                    yaxis_title="Saldırı Sayısı",
                    hovermode='x unified',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📭 Henüz veri yok")

        # SAĞ: Saldırı Türü Dağılımı
        with col2:
            st.subheader("🎯 Saldırı Türleri")

            df_dist = dashboard.get_attack_distribution()

            if len(df_dist) > 0:
                fig = px.pie(
                    df_dist,
                    values='count',
                    names='attack_type',
                    hole=0.5,
                    color_discrete_sequence=px.colors.sequential.RdBu_r
                )

                fig.update_traces(
                    textposition='inside',
                    textinfo='percent+label'
                )

                fig.update_layout(
                    height=400,
                    showlegend=True,
                    legend=dict(
                        orientation="v",
                        yanchor="middle",
                        y=0.5,
                        xanchor="left",
                        x=1.1
                    )
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📭 Henüz veri yok")

        st.markdown("---")

        # ============================================================
        # DETAYLI ANALİZ
        # ============================================================

        col1, col2 = st.columns(2)

        # SOL: Severity Trend (7 gün)
        with col1:
            st.subheader("⚠️ Tehdit Seviyesi Trendi (7 Gün)")

            df_severity = dashboard.get_severity_trend()

            if len(df_severity) > 0:
                # Pivot tablo oluştur
                df_pivot = df_severity.pivot(
                    index='date',
                    columns='severity',
                    values='count'
                ).fillna(0)

                fig = go.Figure()

                severity_colors = {
                    'critical': '#ff4444',
                    'high': '#ff8800',
                    'medium': '#ffbb33',
                    'low': '#00C851'
                }

                for severity in ['low', 'medium', 'high', 'critical']:
                    if severity in df_pivot.columns:
                        fig.add_trace(go.Bar(
                            x=df_pivot.index,
                            y=df_pivot[severity],
                            name=severity.upper(),
                            marker_color=severity_colors.get(severity, 'gray')
                        ))

                fig.update_layout(
                    height=400,
                    barmode='stack',
                    xaxis_title="Tarih",
                    yaxis_title="Saldırı Sayısı",
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📭 Henüz veri yok")

        # SAĞ: Top 5 Saldırganlar
        with col2:
            st.subheader("🚨 En Aktif Saldırganlar (24h)")

            df_attackers = dashboard.get_top_attackers(limit=5)

            if len(df_attackers) > 0:
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    y=df_attackers['source_ip'],
                    x=df_attackers['attacks'],
                    orientation='h',
                    marker=dict(
                        color=df_attackers['attacks'],
                        colorscale='Reds',
                        showscale=False
                    ),
                    text=df_attackers['attacks'],
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>' +
                                  'Saldırı: %{x}<br>' +
                                  '<extra></extra>'
                ))

                fig.update_layout(
                    height=400,
                    xaxis_title="Saldırı Sayısı",
                    yaxis_title="IP Adresi"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Detay tablosu
                st.markdown("##### 📋 Detaylı Bilgi")

                display_df = df_attackers.copy()
                display_df['critical'] = display_df['critical'].astype(int)
                display_df['attacks'] = display_df['attacks'].astype(int)

                st.dataframe(
                    display_df[['source_ip', 'attacks', 'critical', 'last_seen']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "source_ip": "IP Adresi",
                        "attacks": "Saldırı",
                        "critical": "Kritik",
                        "last_seen": "Son Görülme"
                    }
                )
            else:
                st.info("📭 Henüz veri yok")

        st.markdown("---")

        # ============================================================
        # PORT ANALİZİ
        # ============================================================

        st.subheader("🌐 En Çok Hedeflenen Portlar (24h)")

        df_ports = dashboard.get_port_stats(limit=10)

        if len(df_ports) > 0:
            # Port isimleri
            port_names = {
                21: 'FTP', 22: 'SSH', 25: 'SMTP', 53: 'DNS',
                80: 'HTTP', 443: 'HTTPS', 3306: 'MySQL',
                3389: 'RDP', 5432: 'PostgreSQL', 8080: 'HTTP-Alt',
                445: 'SMB', 1433: 'MSSQL'
            }

            df_ports['port_label'] = df_ports['port'].apply(
                lambda x: f"Port {x} ({port_names.get(x, 'Unknown')})"
            )

            col1, col2 = st.columns([2, 1])

            with col1:
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=df_ports['port_label'],
                    y=df_ports['count'],
                    marker=dict(
                        color=df_ports['count'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Saldırı<br>Sayısı")
                    ),
                    text=df_ports['count'],
                    textposition='auto'
                ))

                fig.update_layout(
                    height=350,
                    xaxis_title="Port",
                    yaxis_title="Saldırı Sayısı",
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("##### 📊 Port Detayları")

                display_ports = df_ports[['port', 'count']].copy()
                display_ports['port'] = display_ports['port'].astype(int)
                display_ports['count'] = display_ports['count'].astype(int)

                st.dataframe(
                    display_ports,
                    use_container_width=True,
                    hide_index=True,
                    height=315,
                    column_config={
                        "port": "Port",
                        "count": "Saldırı"
                    }
                )
        else:
            st.info("📭 Henüz veri yok")

        st.markdown("---")

        # ============================================================
        # ÖZET BİLGİLER
        # ============================================================

        with st.expander("ℹ️ Dashboard Bilgileri"):
            st.markdown("""
            ### 📊 Dashboard Metrikleri

            **KPI Kartları:**
            - 🎯 **Toplam Saldırı**: Veritabanındaki tüm saldırılar
            - ⏱️ **Son 24 Saat**: Önceki 24 saat ile karşılaştırma
            - 🛡️ **Engelleme Oranı**: Engellenen saldırı yüzdesi
            - 🚨 **Kritik Tehdit**: Kritik seviye saldırılar
            - 🌐 **Benzersiz IP**: Farklı kaynak IP sayısı

            **Grafikler:**
            - 📈 **Saatlik Trend**: Son 24 saatin detaylı analizi
            - 🎯 **Saldırı Dağılımı**: Saldırı türlerinin oranı
            - ⚠️ **Severity Trendi**: 7 günlük tehdit seviyesi
            - 🚨 **Aktif Saldırganlar**: En çok saldıran IP'ler
            - 🌐 **Port Analizi**: Hedeflenen port dağılımı

            **Veri Güncelleme**: Otomatik (sayfa yenilendiğinde)
            """)

    except Exception as e:
        st.error(f"❌ Dashboard yüklenirken hata: {e}")
        import traceback
        st.code(traceback.format_exc())


# Test
if __name__ == "__main__":
    st.set_page_config(
        page_title="Dashboard - CyberGuard AI",
        page_icon="📊",
        layout="wide"
    )

    show_dashboard_page()