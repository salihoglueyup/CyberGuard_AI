"""
PDF Report Generator - CyberGuard AI
Güvenlik raporları oluşturur
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from datetime import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import io
from typing import Dict, List
import os


class PDFReportGenerator:
    """PDF rapor oluşturucu"""

    def __init__(self, db_path: str = 'src/database/cyberguard.db'):
        self.db_path = db_path
        self.styles = getSampleStyleSheet()

        # Custom stiller
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        )

        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceAfter=12,
            spaceBefore=12
        )

        self.normal_style = self.styles['Normal']

    def get_attack_stats(self, days: int = 7) -> Dict:
        """Saldırı istatistikleri"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Toplam saldırı
        cursor.execute(f"""
            SELECT COUNT(*) FROM attacks 
            WHERE timestamp >= datetime('now', '-{days} days')
        """)
        total = cursor.fetchone()[0]

        # Türe göre
        cursor.execute(f"""
            SELECT attack_type, COUNT(*) as count 
            FROM attacks 
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY attack_type 
            ORDER BY count DESC 
            LIMIT 5
        """)
        by_type = cursor.fetchall()

        # Severity'e göre
        cursor.execute(f"""
            SELECT severity, COUNT(*) as count 
            FROM attacks 
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY severity
        """)
        by_severity = cursor.fetchall()

        # En tehlikeli IP'ler
        cursor.execute(f"""
            SELECT source_ip, COUNT(*) as count 
            FROM attacks 
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY source_ip 
            ORDER BY count DESC 
            LIMIT 10
        """)
        top_ips = cursor.fetchall()

        conn.close()

        return {
            'total': total,
            'by_type': by_type,
            'by_severity': by_severity,
            'top_ips': top_ips
        }

    def create_pie_chart(self, data: List[tuple], title: str):
        """Pie chart oluştur ve Image objesi döndür"""
        if not data:
            return None

        labels = [item[0] for item in data]
        sizes = [item[1] for item in data]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        plt.title(title)

        # BytesIO kullan (dosya kaydetmeden)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        img_buffer.seek(0)

        # Image objesi döndür
        return Image(img_buffer, width=4*inch, height=2.5*inch)

    def create_bar_chart(self, data: List[tuple], title: str, xlabel: str, ylabel: str):
        """Bar chart oluştur ve Image objesi döndür"""
        if not data:
            return None

        labels = [item[0] for item in data]
        values = [item[1] for item in data]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, values, color='#667eea')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # BytesIO kullan
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()
        img_buffer.seek(0)

        return Image(img_buffer, width=5*inch, height=2.5*inch)

    def generate_report(self,
                       output_filename: str = None,
                       days: int = 7,
                       include_charts: bool = True) -> str:
        """
        Tam rapor oluştur

        Args:
            output_filename: Çıktı dosya adı
            days: Kaç günlük veri
            include_charts: Grafik ekle mi?

        Returns:
            str: Oluşturulan dosya yolu
        """
        if output_filename is None:
            output_filename = f'CyberGuard_Report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'

        # PDF oluştur
        doc = SimpleDocTemplate(output_filename, pagesize=A4)
        story = []

        # Başlık
        title = Paragraph("🛡️ CyberGuard AI<br/>Güvenlik Raporu", self.title_style)
        story.append(title)
        story.append(Spacer(1, 0.3*inch))

        # Rapor bilgileri
        report_info = f"""
        <b>Rapor Tarihi:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}<br/>
        <b>Kapsam:</b> Son {days} gün<br/>
        <b>Oluşturan:</b> CyberGuard AI v1.0
        """
        story.append(Paragraph(report_info, self.normal_style))
        story.append(Spacer(1, 0.3*inch))

        # İstatistikleri al
        stats = self.get_attack_stats(days=days)

        # Özet istatistikler
        story.append(Paragraph("📊 Özet İstatistikler", self.heading_style))

        summary_data = [
            ['Metrik', 'Değer'],
            ['Toplam Saldırı', str(stats['total'])],
            ['Farklı Saldırı Türü', str(len(stats['by_type']))],
            ['İzlenen IP Sayısı', str(len(stats['top_ips']))],
        ]

        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))

        # Saldırı türleri
        if stats['by_type']:
            story.append(Paragraph("🎯 Saldırı Türü Dağılımı", self.heading_style))

            type_data = [['Saldırı Türü', 'Sayı', 'Oran']]
            for attack_type, count in stats['by_type']:
                percentage = (count / stats['total'] * 100) if stats['total'] > 0 else 0
                type_data.append([attack_type, str(count), f"{percentage:.1f}%"])

            type_table = Table(type_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch])
            type_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(type_table)
            story.append(Spacer(1, 0.2*inch))

            # Grafik ekle
            if include_charts:
                chart_img = self.create_pie_chart(stats['by_type'], 'Saldırı Türü Dağılımı')
                if chart_img:
                    story.append(chart_img)

        story.append(PageBreak())

        # Severity dağılımı
        if stats['by_severity']:
            story.append(Paragraph("⚠️ Tehdit Seviyesi", self.heading_style))

            severity_data = [['Severity', 'Sayı']]
            for severity, count in stats['by_severity']:
                severity_data.append([severity.upper(), str(count)])

            severity_table = Table(severity_data, colWidths=[3*inch, 2*inch])
            severity_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(severity_table)
            story.append(Spacer(1, 0.2*inch))

            if include_charts:
                chart_img = self.create_bar_chart(
                    stats['by_severity'],
                    'Tehdit Seviyesi Dağılımı',
                    'Severity',
                    'Saldırı Sayısı'
                )
                if chart_img:
                    story.append(chart_img)

        story.append(Spacer(1, 0.3*inch))

        # En tehlikeli IP'ler
        if stats['top_ips']:
            story.append(Paragraph("🚨 En Aktif Saldırganlar (Top 10)", self.heading_style))

            ip_data = [['Sıra', 'IP Adresi', 'Saldırı Sayısı']]
            for idx, (ip, count) in enumerate(stats['top_ips'], 1):
                ip_data.append([str(idx), ip, str(count)])

            ip_table = Table(ip_data, colWidths=[0.8*inch, 2.5*inch, 2*inch])
            ip_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff4444')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lavender),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            story.append(ip_table)

        # Footer
        story.append(Spacer(1, 0.5*inch))
        footer_text = f"""
        <i>Bu rapor CyberGuard AI tarafından otomatik oluşturulmuştur.<br/>
        © 2025 CyberGuard AI | Yapay Zeka Destekli Siber Güvenlik Platformu</i>
        """
        story.append(Paragraph(footer_text, self.normal_style))

        # PDF'i oluştur
        doc.build(story)

        print(f"✅ Rapor oluşturuldu: {output_filename}")
        return output_filename


# Test
if __name__ == "__main__":
    print("🧪 PDF Generator Test\n")

    generator = PDFReportGenerator()

    # Rapor oluştur
    filename = generator.generate_report(
        output_filename='test_report.pdf',
        days=7,
        include_charts=True
    )

    print(f"\n✅ Test tamamlandı: {filename}")