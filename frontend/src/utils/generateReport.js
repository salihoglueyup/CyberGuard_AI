import jsPDF from 'jspdf';
import 'jspdf-autotable';

/**
 * CyberGuard AI - PDF Rapor Oluşturucu
 */

// Renk tanımları
const colors = {
    primary: [59, 130, 246],      // Blue
    success: [34, 197, 94],       // Green
    warning: [245, 158, 11],      // Orange
    danger: [239, 68, 68],        // Red
    dark: [15, 23, 42],           // Slate-900
    light: [241, 245, 249],       // Slate-100
    text: [51, 65, 85],           // Slate-700
};

export function generateDashboardReport(data) {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();
    let yPos = 20;

    // === BAŞLIK ===
    doc.setFillColor(...colors.dark);
    doc.rect(0, 0, pageWidth, 45, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(24);
    doc.setFont('helvetica', 'bold');
    doc.text('CyberGuard AI', 20, 25);

    doc.setFontSize(12);
    doc.setFont('helvetica', 'normal');
    doc.text('Guvenlik Raporu', 20, 35);

    // Tarih
    doc.setFontSize(10);
    doc.text(new Date().toLocaleDateString('tr-TR', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }), pageWidth - 20, 25, { align: 'right' });

    yPos = 55;

    // === ÖZET KARTLAR ===
    doc.setTextColor(...colors.text);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text('Sistem Ozeti', 20, yPos);
    yPos += 10;

    const stats = data.stats || {};
    const summaryData = [
        ['Toplam Model', stats.total_models || 0, colors.primary],
        ['Dagitilan', stats.deployed_models || stats.blocked || 0, colors.success],
        ['En Iyi Dogruluk', `${((stats.best_accuracy || 0) * 100).toFixed(1)}%`, colors.primary],
        ['Egitimde', stats.training_models || 0, colors.warning],
    ];

    const cardWidth = (pageWidth - 50) / 4;
    summaryData.forEach((item, idx) => {
        const x = 20 + (idx * (cardWidth + 5));

        // Kart arkaplan
        doc.setFillColor(248, 250, 252);
        doc.roundedRect(x, yPos, cardWidth, 30, 3, 3, 'F');

        // Değer
        doc.setTextColor(...item[2]);
        doc.setFontSize(18);
        doc.setFont('helvetica', 'bold');
        doc.text(String(item[1]), x + cardWidth / 2, yPos + 15, { align: 'center' });

        // Başlık
        doc.setTextColor(...colors.text);
        doc.setFontSize(8);
        doc.setFont('helvetica', 'normal');
        doc.text(item[0], x + cardWidth / 2, yPos + 25, { align: 'center' });
    });

    yPos += 45;

    // === SON MODELLER TABLOSU ===
    doc.setTextColor(...colors.text);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text('Son Modeller', 20, yPos);
    yPos += 5;

    const models = data.recentModels || [];
    if (models.length > 0) {
        doc.autoTable({
            startY: yPos,
            head: [['Model Adi', 'Framework', 'Durum', 'Tarih']],
            body: models.map(m => [
                m.source_ip || m.name || 'Bilinmiyor',
                m.attack_type || m.framework || 'TensorFlow',
                m.blocked ? 'Dagitildi' : 'Hazir',
                m.timestamp ? new Date(m.timestamp).toLocaleDateString('tr-TR') : '-'
            ]),
            theme: 'striped',
            headStyles: {
                fillColor: colors.primary,
                textColor: [255, 255, 255],
                fontStyle: 'bold'
            },
            bodyStyles: {
                textColor: colors.text
            },
            alternateRowStyles: {
                fillColor: [248, 250, 252]
            },
            margin: { left: 20, right: 20 },
        });
        yPos = doc.lastAutoTable.finalY + 15;
    }

    // === TEHDİT İSTATİSTİKLERİ ===
    if (data.threatStats) {
        doc.setTextColor(...colors.text);
        doc.setFontSize(16);
        doc.setFont('helvetica', 'bold');
        doc.text('Tehdit Istatistikleri', 20, yPos);
        yPos += 10;

        const threatData = data.threatStats;
        doc.setFontSize(11);
        doc.setFont('helvetica', 'normal');

        doc.text(`Toplam Tehdit: ${threatData.total || 0}`, 25, yPos);
        yPos += 7;
        doc.text(`Engellenen: ${threatData.blocked || 0}`, 25, yPos);
        yPos += 7;
        doc.text(`Basari Orani: ${threatData.blockRate || 0}%`, 25, yPos);
        yPos += 15;
    }

    // === SİSTEM SAĞLIĞI ===
    doc.setTextColor(...colors.text);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text('Sistem Sagligi', 20, yPos);
    yPos += 10;

    const systemHealth = data.systemHealth || {
        cpu: 34,
        memory: 67,
        storage: 45,
        network: 89
    };

    const healthItems = [
        { label: 'CPU Kullanimi', value: systemHealth.cpu },
        { label: 'Bellek', value: systemHealth.memory },
        { label: 'Depolama', value: systemHealth.storage },
        { label: 'Ag', value: systemHealth.network },
    ];

    healthItems.forEach((item, idx) => {
        const barWidth = 100;
        const x = 25;
        const barY = yPos + (idx * 12);

        // Label
        doc.setFontSize(9);
        doc.text(item.label, x, barY);

        // Progress bar background
        doc.setFillColor(226, 232, 240);
        doc.roundedRect(x + 45, barY - 5, barWidth, 6, 2, 2, 'F');

        // Progress bar fill
        const fillColor = item.value > 80 ? colors.danger : item.value > 60 ? colors.warning : colors.success;
        doc.setFillColor(...fillColor);
        doc.roundedRect(x + 45, barY - 5, (item.value / 100) * barWidth, 6, 2, 2, 'F');

        // Value
        doc.text(`${item.value}%`, x + 150, barY);
    });

    yPos += 60;

    // === FOOTER ===
    const footerY = doc.internal.pageSize.getHeight() - 20;
    doc.setDrawColor(226, 232, 240);
    doc.line(20, footerY - 5, pageWidth - 20, footerY - 5);

    doc.setFontSize(8);
    doc.setTextColor(148, 163, 184);
    doc.text('Bu rapor CyberGuard AI tarafindan otomatik olusturulmustur.', 20, footerY);
    doc.text(`Sayfa 1/1`, pageWidth - 20, footerY, { align: 'right' });

    // === KAYDET ===
    const filename = `CyberGuard_Rapor_${new Date().toISOString().split('T')[0]}.pdf`;
    doc.save(filename);

    return filename;
}

export function generateThreatReport(threats) {
    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();

    // Başlık
    doc.setFillColor(...colors.danger);
    doc.rect(0, 0, pageWidth, 40, 'F');

    doc.setTextColor(255, 255, 255);
    doc.setFontSize(20);
    doc.setFont('helvetica', 'bold');
    doc.text('Tehdit Raporu', 20, 25);

    doc.setFontSize(10);
    doc.text(new Date().toLocaleDateString('tr-TR'), pageWidth - 20, 25, { align: 'right' });

    // Tablo
    if (threats.length > 0) {
        doc.autoTable({
            startY: 50,
            head: [['ID', 'Tur', 'Kaynak IP', 'Ulke', 'Seviye', 'Durum']],
            body: threats.map(t => [
                t.id || '-',
                t.threat_type || '-',
                t.source_ip || '-',
                t.country || '-',
                t.severity || '-',
                t.blocked ? 'Engellendi' : 'Aktif'
            ]),
            theme: 'grid',
            headStyles: {
                fillColor: colors.danger,
                textColor: [255, 255, 255]
            },
            columnStyles: {
                4: {
                    cellWidth: 20,
                    halign: 'center'
                },
                5: {
                    cellWidth: 25,
                    halign: 'center'
                }
            }
        });
    }

    const filename = `CyberGuard_Tehdit_${new Date().toISOString().split('T')[0]}.pdf`;
    doc.save(filename);

    return filename;
}

export default { generateDashboardReport, generateThreatReport };
