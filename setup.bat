@echo off
REM ========================================
REM CyberGuard AI - Kurulum Script
REM ========================================

echo.
echo ========================================
echo   CyberGuard AI Kurulum Baslatiluyor
echo ========================================
echo.

REM Yönetici kontrolü
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Bu script yonetici olarak calistirilmali!
    echo Sag tiklayip "Yonetici olarak calistir" seciniz.
    pause
    exit /b 1
)

echo [1/8] Python versiyonu kontrol ediliyor...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [HATA] Python bulunamadi!
    echo Python 3.8+ yukleyin: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Python bulundu
python --version

echo.
echo [2/8] Sanal ortam olusturuluyor...
if exist venv (
    echo [UYARI] venv klasoru zaten mevcut. Siliniyor...
    rmdir /s /q venv
)
python -m venv venv
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam olusturulamadi!
    pause
    exit /b 1
)
echo [OK] Sanal ortam olusturuldu

echo.
echo [3/8] Sanal ortam aktive ediliyor...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam aktive edilemedi!
    pause
    exit /b 1
)
echo [OK] Sanal ortam aktif

echo.
echo [4/8] pip guncelleniyor...
python -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [UYARI] pip guncellenemedi ama devam ediliyor...
)
echo [OK] pip guncellendi

echo.
echo [5/8] Requirements yukleniyor...
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
echo [6/8] .env dosyasi olusturuluyor...
if exist .env (
    echo [BILGI] .env dosyasi zaten mevcut, atlanıyor...
) else (
    if exist .env.example (
        copy .env.example .env
        echo [OK] .env dosyasi .env.example'dan olusturuldu
        echo [UYARI] .env dosyasini duzenleyip API anahtarlarini ekleyin!
    ) else (
        echo [UYARI] .env.example bulunamadi
    )
)

echo.
echo [7/8] Gerekli klasorler olusturuluyor...
if not exist logs mkdir logs
if not exist data mkdir data
if not exist models mkdir models
if not exist uploads mkdir uploads
if not exist reports mkdir reports
echo [OK] Klasorler olusturuldu

echo.
echo [8/8] Veritabani baslatiliyor...
if exist src\database\init_db.py (
    python src\database\init_db.py
    if %errorlevel% equ 0 (
        echo [OK] Veritabani baslatildi
    ) else (
        echo [UYARI] Veritabani baslatma hatasi
    )
) else (
    echo [BILGI] init_db.py bulunamadi, atlanıyor...
)

echo.
echo ========================================
echo   KURULUM TAMAMLANDI!
echo ========================================
echo.
echo Sonraki adimlar:
echo   1. .env dosyasini duzenleyin
echo   2. run.bat ile uygulamayi baslatin
echo   3. http://localhost:5000 adresini ziyaret edin
echo.
echo Dokumantasyon: docs/user_guide.md
echo.
pause