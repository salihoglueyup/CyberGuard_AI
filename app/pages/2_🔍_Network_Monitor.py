# app/pages/network_monitor.py

"""
Network Monitor - CyberGuard AI
Ağ trafiği izleme ve saldırı analizi
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sqlite3
import sys
import os
import time

# Path ayarları
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class NetworkMonitor:
    """Network monitoring ve analiz sınıfı"""

    def __init__(self, db_path: str = 'cyberguard.db'):
        self.db_path = db_path

    def get_recent_attacks(self, limit: int = 100, hours: int = 24) -> pd.DataFrame:
        """Son saldırıları getir"""
        conn = sqlite3.connect(self.db_path)

        query = f"""
        SELECT * FROM attacks 
        WHERE timestamp >= datetime('now', '-{hours} hours')
        ORDER BY timestamp DESC 
        LIMIT {limit}
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_attack_stats(self) -> dict:
        """Saldırı istatistikleri"""
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

        # Engellenen saldırılar
        cursor.execute("SELECT COUNT(*) FROM attacks WHERE blocked = 1")
        blocked = cursor.fetchone()[0]

        # Kritik saldırılar
        cursor.execute("SELECT COUNT(*) FROM attacks WHERE severity = 'critical'")
        critical = cursor.fetchone()[0]

        conn.close()

        return {
            'total': total_attacks,
            'last_24h': last_24h,
            'blocked': blocked,
            'critical': critical
        }

    def get_attack_by_type(self) -> pd.DataFrame:
        """Saldırı türüne göre dağılım"""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT attack_type, COUNT(*) as count
                FROM attacks
                GROUP BY attack_type
                ORDER BY count DESC \
                """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_attack_by_severity(self) -> pd.DataFrame:
        """Severity'e göre dağılım"""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT severity, COUNT(*) as count
                FROM attacks
                GROUP BY severity
                ORDER BY
                    CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END \
                """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_top_ips(self, limit: int = 10) -> pd.DataFrame:
        """En çok saldıran IP'ler"""
        conn = sqlite3.connect(self.db_path)

        query = f"""
        SELECT source_ip, COUNT(*) as attack_count,
               SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_count
        FROM attacks 
        GROUP BY source_ip 
        ORDER BY attack_count DESC 
        LIMIT {limit}
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_port_activity(self) -> pd.DataFrame:
        """Port aktivite analizi"""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT port, COUNT(*) as count
                FROM attacks
                GROUP BY port
                ORDER BY count DESC
                    LIMIT 15 \
                """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_timeline(self, hours: int = 24) -> pd.DataFrame:
        """Zaman serisi analizi"""
        conn = sqlite3.connect(self.db_path)

        query = f"""
        SELECT 
            strftime('%Y-%m-%d %H:00:00', timestamp) as hour,
            COUNT(*) as count
        FROM attacks 
        WHERE timestamp >= datetime('now', '-{hours} hours')
        GROUP BY hour 
        ORDER BY hour
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def search_ip(self, ip: str) -> pd.DataFrame:
        """Belirli IP'yi sorgula"""
        conn = sqlite3.connect(self.db_path)

        query = """
                SELECT * \
                FROM attacks
                WHERE source_ip = ? \
                   OR destination_ip = ?
                ORDER BY timestamp DESC \
                """

        df = pd.read_sql_query(query, conn, params=(ip, ip))
        conn.close()

        return df


def show_network_monitor_page():
    """Network Monitor sayfasını göster"""

    st.title("🔍 Network Monitor")
    st.markdown("Gerçek zamanlı ağ trafiği izleme ve saldırı analizi")

    # Auto-refresh
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown("---")

    with col2:
        auto_refresh = st.toggle("🔄 Canlı İzleme", value=False, key="network_refresh")

    with col3:
        if auto_refresh:
            refresh_interval = st.selectbox(
                "Yenileme",
                options=[5, 10, 15, 30],
                format_func=lambda x: f"{x}s",
                index=1,
                key="network_interval"
            )
        else:
            refresh_interval = 10

    st.markdown("---")

    # Auto-refresh logic
    if auto_refresh:
        st.info(f"🔄 Canlı izleme aktif (Her {refresh_interval} saniyede)")

        # Basit counter ile yenileme
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

        elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()

        if elapsed >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            st.rerun()

        # Kalan süre göster
        remaining = int(refresh_interval - elapsed)
        st.caption(f"⏳ Yenilenecek: {remaining} saniye | Son: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

    # Son güncelleme
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⏱️ Son Güncelleme")
    st.sidebar.success(datetime.now().strftime('%d/%m/%Y %H:%M:%S'))

    # Monitor başlat
    try:
        monitor = NetworkMonitor()

        # İstatistikler
        stats = monitor.get_attack_stats()

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📊 Toplam Saldırı",
                f"{stats['total']:,}",
                delta=None
            )

        with col2:
            st.metric(
                "⏱️ Son 24 Saat",
                f"{stats['last_24h']:,}",
                delta=f"+{stats['last_24h']}"
            )

        with col3:
            blocked_pct = (stats['blocked'] / stats['total'] * 100) if stats['total'] > 0 else 0
            st.metric(
                "🛡️ Engellenen",
                f"{stats['blocked']:,}",
                delta=f"{blocked_pct:.1f}%"
            )

        with col4:
            st.metric(
                "🚨 Kritik",
                f"{stats['critical']:,}",
                delta="Yüksek Tehdit",
                delta_color="inverse"
            )

        st.markdown("---")

        # Tab'lar
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Canlı Saldırılar",
            "📊 Analiz & Grafikler",
            "🔍 IP Sorgulama",
            "🌐 Port Aktivitesi"
        ])

        # TAB 1: Canlı Saldırılar
        with tab1:
            st.subheader("📋 Son Saldırılar")

            col1, col2 = st.columns([3, 1])

            with col1:
                limit = st.slider("Gösterilecek kayıt sayısı", 10, 200, 50)

            with col2:
                auto_refresh = st.checkbox("🔄 Otomatik Yenile", value=False)

            # Son saldırılar
            df_attacks = monitor.get_recent_attacks(limit=limit)

            if len(df_attacks) > 0:
                # Renklendirme için
                def highlight_severity(row):
                    if row['severity'] == 'critical':
                        return ['background-color: #ff4444'] * len(row)
                    elif row['severity'] == 'high':
                        return ['background-color: #ff8800'] * len(row)
                    elif row['severity'] == 'medium':
                        return ['background-color: #ffbb33'] * len(row)
                    else:
                        return ['background-color: #00C851'] * len(row)

                # Seçili kolonlar
                display_cols = [
                    'timestamp', 'attack_type', 'source_ip',
                    'destination_ip', 'port', 'severity', 'blocked'
                ]

                st.dataframe(
                    df_attacks[display_cols].head(limit),
                    use_container_width=True,
                    height=400
                )

                # İndirme butonu
                csv = df_attacks.to_csv(index=False)
                st.download_button(
                    "📥 CSV İndir",
                    csv,
                    "attacks.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.info("📭 Henüz saldırı kaydı yok")

        # TAB 2: Analiz & Grafikler
        with tab2:
            st.subheader("📊 Saldırı Analizi")

            col1, col2 = st.columns(2)

            # Saldırı türü dağılımı
            with col1:
                st.markdown("#### 🎯 Saldırı Türü Dağılımı")

                df_types = monitor.get_attack_by_type()

                if len(df_types) > 0:
                    fig = px.pie(
                        df_types,
                        values='count',
                        names='attack_type',
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Veri yok")

            # Severity dağılımı
            with col2:
                st.markdown("#### ⚠️ Tehdit Seviyesi")

                df_severity = monitor.get_attack_by_severity()

                if len(df_severity) > 0:
                    colors = {
                        'critical': '#ff4444',
                        'high': '#ff8800',
                        'medium': '#ffbb33',
                        'low': '#00C851'
                    }

                    fig = go.Figure(go.Bar(
                        x=df_severity['severity'],
                        y=df_severity['count'],
                        marker_color=[colors.get(s, '#gray') for s in df_severity['severity']],
                        text=df_severity['count'],
                        textposition='auto'
                    ))

                    fig.update_layout(
                        height=400,
                        xaxis_title="Severity",
                        yaxis_title="Saldırı Sayısı"
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Veri yok")

            # Zaman serisi
            st.markdown("#### 📈 Son 24 Saat Aktivite")

            df_timeline = monitor.get_timeline(hours=24)

            if len(df_timeline) > 0:
                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=df_timeline['hour'],
                    y=df_timeline['count'],
                    mode='lines+markers',
                    name='Saldırı Sayısı',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=8),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.2)'
                ))

                fig.update_layout(
                    height=400,
                    xaxis_title="Zaman",
                    yaxis_title="Saldırı Sayısı",
                    hovermode='x unified'
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📭 Son 24 saatte veri yok")

            # En tehlikeli IP'ler
            st.markdown("#### 🚨 En Aktif Saldırganlar (Top 10)")

            df_top_ips = monitor.get_top_ips(limit=10)

            if len(df_top_ips) > 0:
                fig = go.Figure()

                fig.add_trace(go.Bar(
                    y=df_top_ips['source_ip'],
                    x=df_top_ips['attack_count'],
                    orientation='h',
                    marker=dict(
                        color=df_top_ips['attack_count'],
                        colorscale='Reds',
                        showscale=True
                    ),
                    text=df_top_ips['attack_count'],
                    textposition='auto'
                ))

                fig.update_layout(
                    height=400,
                    xaxis_title="Saldırı Sayısı",
                    yaxis_title="IP Adresi"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Tablo
                st.dataframe(
                    df_top_ips,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Veri yok")

        # TAB 3: IP Sorgulama
        with tab3:
            st.subheader("🔍 IP Adresi Sorgulama")

            col1, col2 = st.columns([3, 1])

            with col1:
                search_ip = st.text_input(
                    "IP Adresi Girin",
                    placeholder="örn: 192.168.1.105",
                    help="Kaynak veya hedef IP adresi"
                )

            with col2:
                search_btn = st.button("🔍 Sorgula", type="primary", use_container_width=True)

            if search_btn and search_ip:
                with st.spinner(f"🔍 {search_ip} sorgulanıyor..."):
                    results = monitor.search_ip(search_ip)

                if len(results) > 0:
                    st.success(f"✅ {len(results)} kayıt bulundu!")

                    # Özet istatistikler
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Toplam Saldırı", len(results))

                    with col2:
                        critical_count = len(results[results['severity'] == 'critical'])
                        st.metric("Kritik Saldırı", critical_count)

                    with col3:
                        blocked_count = len(results[results['blocked'] == 1])
                        st.metric("Engellenen", blocked_count)

                    # Detaylar
                    st.markdown("#### 📋 Detaylı Kayıtlar")
                    st.dataframe(results, use_container_width=True)

                else:
                    st.warning(f"⚠️ {search_ip} için kayıt bulunamadı")

        # TAB 4: Port Aktivitesi
        with tab4:
            st.subheader("🌐 Port Aktivite Analizi")

            df_ports = monitor.get_port_activity()

            if len(df_ports) > 0:
                # Grafik
                fig = go.Figure()

                # Port isimleri
                port_names = {
                    21: 'FTP', 22: 'SSH', 25: 'SMTP',
                    80: 'HTTP', 443: 'HTTPS', 3306: 'MySQL',
                    3389: 'RDP', 5432: 'PostgreSQL', 8080: 'HTTP-Alt'
                }

                df_ports['port_name'] = df_ports['port'].apply(
                    lambda x: f"{x} ({port_names.get(x, 'Unknown')})"
                )

                fig.add_trace(go.Bar(
                    x=df_ports['port_name'],
                    y=df_ports['count'],
                    marker=dict(
                        color=df_ports['count'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    text=df_ports['count'],
                    textposition='auto'
                ))

                fig.update_layout(
                    height=500,
                    xaxis_title="Port",
                    yaxis_title="Saldırı Sayısı",
                    xaxis_tickangle=-45
                )

                st.plotly_chart(fig, use_container_width=True)

                # Tablo
                st.markdown("#### 📊 Detaylı Port İstatistikleri")
                st.dataframe(
                    df_ports[['port', 'count']],
                    use_container_width=True,
                    hide_index=True
                )

            else:
                st.info("📭 Port aktivitesi verisi yok")

    except Exception as e:
        st.error(f"❌ Bir hata oluştu: {e}")
        import traceback
        st.code(traceback.format_exc())


# Test
if __name__ == "__main__":
    st.set_page_config(
        page_title="Network Monitor - CyberGuard AI",
        page_icon="🔍",
        layout="wide"
    )

    show_network_monitor_page()