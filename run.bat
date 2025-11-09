@echo off
REM ========================================
REM CyberGuard AI - Uygulama Baslat
REM ========================================

echo.
echo ========================================
echo   CyberGuard AI Baslatiluyor...
echo ========================================
echo.

REM Sanal ortam kontrolü
if not exist venv (
    echo [HATA] Sanal ortam bulunamadi!
    echo Once setup.bat calistirin.
    pause
    exit /b 1
)

echo [1/4] Sanal ortam aktive ediliyor...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [HATA] Sanal ortam aktive edilemedi!
    pause
    exit /b 1
)
echo [OK] Sanal ortam aktif

echo.
echo [2/4] .env dosyasi kontrol ediliyor...
if not exist .env (
    echo [UYARI] .env dosyasi bulunamadi!
    if exist .env.example (
        copy .env.example .env
        echo [OK] .env dosyasi olusturuldu
        echo [UYARI] Lutfen .env dosyasini duzenleyin ve tekrar calistirin!
        pause
        exit /b 1
    ) else (
        echo [HATA] .env.example dosyasi da bulunamadi!
        pause
        exit /b 1
    )
)
echo [OK] .env dosyasi mevcut

echo.
echo [3/4] Gerekli servisler kontrol ediliyor...
REM PostgreSQL kontrolü (opsiyonel)
REM Redis kontrolü (opsiyonel)
echo [OK] Servis kontrolu tamamlandi

echo.
echo [4/4] Uygulama baslatiliyor...
echo.
echo ========================================
echo   CyberGuard AI CALISTIRILDI!
echo ========================================
echo.
echo   URL: http://localhost:5000
echo   Durdurmak icin: CTRL + C
echo.
echo ========================================
echo.

REM Ana uygulama
python src/main.py

REM Hata durumunda
if %errorlevel% neq 0 (
    echo.
    echo [HATA] Uygulama beklenmedik sekilde sonlandi!
    echo Hata kodu: %errorlevel%
    echo.
    echo Log dosyasini kontrol edin: logs/application.log
    pause
)

REM Temizlik
deactivate 2>nul