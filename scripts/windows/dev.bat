@echo off
REM ========================================
REM CyberGuard AI - Development Mode
REM ========================================

echo.
echo ========================================
echo   CyberGuard AI Development Mode
echo ========================================
echo.

REM Sanal ortam kontrolü
if not exist venv (
    echo [HATA] Sanal ortam bulunamadi!
    echo Once setup.bat calistirin.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

echo [1/3] Development ortami hazirlaniyor...
set FLASK_ENV=development
set FLASK_DEBUG=1
set PYTHONDONTWRITEBYTECODE=1
echo [OK] Environment variables ayarlandi

echo.
echo [2/3] Development dependencies kontrol ediliyor...
if exist requirements-dev.txt (
    pip install -r requirements-dev.txt -q
    echo [OK] Dev dependencies yuklendi
) else (
    echo [UYARI] requirements-dev.txt bulunamadi
)

echo.
echo [3/3] Uygulama development modunda baslatiliyor...
echo.
echo ========================================
echo   DEVELOPMENT MODE AKTIF
echo ========================================
echo.
echo   URL: http://localhost:5000
echo   Hot Reload: Aktif
echo   Debug Mode: Aktif
echo   Auto-restart: Aktif
echo.
echo   Durdurmak icin: CTRL + C
echo ========================================
echo.

REM Flask development server ile başlat
python -m flask run --host=0.0.0.0 --port=5000 --reload --debugger

REM Alternatif: Watchdog ile auto-reload
REM watchmedo auto-restart --directory=./src --pattern=*.py --recursive -- python src/main.py

if %errorlevel% neq 0 (
    echo.
    echo [HATA] Uygulama beklenmedik sekilde durdu!
    echo Log dosyasini kontrol edin: logs/dev.log
    pause
)

deactivate 2>nul