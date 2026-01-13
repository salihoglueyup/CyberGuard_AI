@echo off
chcp 65001 >nul
REM ========================================
REM CyberGuard AI - Kurulum Script
REM ========================================

echo.
echo ╔════════════════════════════════════════════╗
echo ║   🛡️  CYBERGUARD AI KURULUM BASLATIYOR    ║
echo ╚════════════════════════════════════════════╝
echo.

REM Python kontrolu
echo [1/7] Python versiyonu kontrol ediliyor...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi!
    echo Python 3.10+ yukleyin: https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [OK] Python bulundu
python --version

echo.
echo [2/7] Sanal ortam olusturuluyor...
if exist venv (
    echo [BILGI] venv klasoru zaten mevcut, atlaniyor...
) else (
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [HATA] Sanal ortam olusturulamadi!
        pause
        exit /b 1
    )
    echo [OK] Sanal ortam olusturuldu
)

echo.
echo [3/7] Sanal ortam aktive ediliyor...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam aktive edilemedi!
    pause
    exit /b 1
)
echo [OK] Sanal ortam aktif

echo.
echo [4/7] pip guncelleniyor...
python -m pip install --upgrade pip --quiet
echo [OK] pip guncellendi

echo.
echo [5/7] Requirements yukleniyor...
if not exist requirements.txt (
    echo [HATA] requirements.txt bulunamadi!
    pause
    exit /b 1
)
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [HATA] Bagimliliklar yuklenemedi!
    pause
    exit /b 1
)
echo [OK] Tum paketler yuklendi

echo.
echo [6/7] .env dosyasi kontrol ediliyor...
if exist .env (
    echo [BILGI] .env dosyasi zaten mevcut
) else (
    if exist .env.example (
        copy .env.example .env >nul
        echo [OK] .env dosyasi olusturuldu
        echo [UYARI] .env dosyasinda GOOGLE_API_KEY ayarlamayi unutmayin!
    ) else (
        echo [UYARI] .env.example bulunamadi
    )
)

echo.
echo [7/7] Gerekli klasorler olusturuluyor...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist data\logs mkdir data\logs
if not exist models mkdir models
if not exist uploads mkdir uploads
if not exist reports mkdir reports
echo [OK] Klasorler olusturuldu

echo.
echo ╔════════════════════════════════════════════╗
echo ║       ✅ KURULUM TAMAMLANDI!              ║
echo ╠════════════════════════════════════════════╣
echo ║                                            ║
echo ║  Sonraki adimlar:                          ║
echo ║  1. .env dosyasini duzenleyin              ║
echo ║     (GOOGLE_API_KEY ekleyin)               ║
echo ║  2. run.bat ile uygulamayi baslatin        ║
echo ║  3. http://localhost:8501 adresini acin    ║
echo ║                                            ║
echo ╚════════════════════════════════════════════╝
echo.
pause