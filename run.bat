@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion

:: ════════════════════════════════════════════════════════════════════════════
::                          CYBERGUARD AI LAUNCHER v2.0
:: ════════════════════════════════════════════════════════════════════════════

title CyberGuard AI - Siber Guvenlik Platformu
color 0B

:MAIN_MENU
cls
echo.
echo   ╔══════════════════════════════════════════════════════════════════════╗
echo   ║                                                                      ║
echo   ║   ██████╗██╗   ██╗██████╗ ███████╗██████╗  ██████╗ ██╗   ██╗ █████╗  ║
echo   ║  ██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗██╔════╝ ██║   ██║██╔══██╗ ║
echo   ║  ██║      ╚████╔╝ ██████╔╝█████╗  ██████╔╝██║  ███╗██║   ██║███████║ ║
echo   ║  ██║       ╚██╔╝  ██╔══██╗██╔══╝  ██╔══██╗██║   ██║██║   ██║██╔══██║ ║
echo   ║  ╚██████╗   ██║   ██████╔╝███████╗██║  ██║╚██████╔╝╚██████╔╝██║  ██║ ║
echo   ║   ╚═════╝   ╚═╝   ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝ ║
echo   ║                                                                      ║
echo   ║                    AI-POWERED CYBER SECURITY PLATFORM                ║
echo   ║                             v2.0.0                                   ║
echo   ╠══════════════════════════════════════════════════════════════════════╣
echo   ║                                                                      ║
echo   ║   [1] Tam Baslat       - Backend + Frontend                          ║
echo   ║   [2] Sadece Backend   - FastAPI Server                              ║
echo   ║   [3] Sadece Frontend  - React/Vite Dev Server                       ║
echo   ║   [4] Model Egitimi    - ML Model Training                           ║
echo   ║   [5] Database Ayarla  - Veritabani Islemleri                        ║
echo   ║   [6] Bagimliliklari Kur - pip install                               ║
echo   ║   [7] Sistem Durumu    - Health Check                                ║
echo   ║   [8] Loglar           - Canlı Log Goruntule                         ║
echo   ║   [9] Temizlik         - Cache ve Gecici Dosyalar                    ║
echo   ║   [0] Cikis                                                          ║
echo   ║                                                                      ║
echo   ╚══════════════════════════════════════════════════════════════════════╝
echo.

set /p choice="   Seciminiz [0-9]: "

if "%choice%"=="1" goto START_ALL
if "%choice%"=="2" goto START_BACKEND
if "%choice%"=="3" goto START_FRONTEND
if "%choice%"=="4" goto TRAIN_MODEL
if "%choice%"=="5" goto DATABASE_MENU
if "%choice%"=="6" goto INSTALL_DEPS
if "%choice%"=="7" goto HEALTH_CHECK
if "%choice%"=="8" goto VIEW_LOGS
if "%choice%"=="9" goto CLEANUP
if "%choice%"=="0" goto EXIT

echo.
echo   [!] Gecersiz secim!
timeout /t 2 >nul
goto MAIN_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              SANAL ORTAM AKTIVASYONU
:: ════════════════════════════════════════════════════════════════════════════
:ACTIVATE_VENV
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
    set VENV_ACTIVE=1
    echo   [OK] .venv aktive edildi
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    set VENV_ACTIVE=1
    echo   [OK] venv aktive edildi
) else (
    set VENV_ACTIVE=0
    echo   [!] Sanal ortam bulunamadi, global Python kullaniliyor
)
exit /b

:: ════════════════════════════════════════════════════════════════════════════
::                              TAM BASLAT
:: ════════════════════════════════════════════════════════════════════════════
:START_ALL
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║              CYBERGUARD AI - TAM BASLAT                        ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

call :ACTIVATE_VENV
echo.

:: Port temizle
call :KILL_PORT
echo.

:: .env kontrolu
if not exist .env (
    echo   [!] UYARI: .env dosyasi bulunamadi
    echo       Gemini API key ekleyin: GEMINI_API_KEY=your_key
    echo.
)

:: Database kontrolu
if not exist src\database\cyberguard.db (
    echo   [*] Database bulunamadi, olusturuluyor...
    python scripts\setup_database.py
    echo.
)

echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║  [1/2] FastAPI Backend baslatiliyor...                         ║
echo   ╚════════════════════════════════════════════════════════════════╝

start "CyberGuard API" cmd /k "title CyberGuard API - Port 8000 && color 0A && python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0"

timeout /t 3 /nobreak >nul

echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║  [2/2] React Frontend baslatiliyor...                          ║
echo   ╚════════════════════════════════════════════════════════════════╝

start "CyberGuard UI" cmd /k "title CyberGuard UI - Port 5173 && color 0D && cd frontend && npm run dev"

timeout /t 2 /nobreak >nul

echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║                   CYBERGUARD AI BASLATILDI!                    ║
echo   ╠════════════════════════════════════════════════════════════════╣
echo   ║                                                                ║
echo   ║   Frontend:   http://localhost:5173                            ║
echo   ║   Backend:    http://localhost:8000                            ║
echo   ║   API Docs:   http://localhost:8000/api/docs                   ║
echo   ║   WebSocket:  ws://localhost:8000/ws                           ║
echo   ║                                                                ║
echo   ╠════════════════════════════════════════════════════════════════╣
echo   ║   Demo Hesap: admin / admin123                                 ║
echo   ╠════════════════════════════════════════════════════════════════╣
echo   ║   Kapatmak icin: CTRL+C (acilan pencerelerde)                  ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

:: Tarayici ac
timeout /t 2 /nobreak >nul
start "" "http://localhost:5173"

echo   [*] Tarayici acildi. Ana menuye donmek icin bir tusa basin...
pause >nul
goto MAIN_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              SADECE BACKEND
:: ════════════════════════════════════════════════════════════════════════════
:START_BACKEND
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║              FASTAPI BACKEND BASLATILIYOR                      ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

call :ACTIVATE_VENV
echo.

:: Port temizle
call :KILL_PORT
echo.
echo   [*] http://localhost:8000 adresinde baslatiliyor...
echo   [*] API Docs: http://localhost:8000/api/docs
echo.
python -m uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
goto MAIN_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              SADECE FRONTEND
:: ════════════════════════════════════════════════════════════════════════════
:START_FRONTEND
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║              REACT FRONTEND BASLATILIYOR                       ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.
echo   [*] http://localhost:5173 adresinde baslatiliyor...
echo.
cd frontend
npm run dev
cd ..
goto MAIN_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              MODEL EGITIMI
:: ════════════════════════════════════════════════════════════════════════════
:TRAIN_MODEL
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║                    ML MODEL EGITIMI                            ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

call :ACTIVATE_VENV
echo.

echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║   [1] Interaktif Egitim (Menu ile)                             ║
echo   ║   [2] Hizli Egitim (Varsayilan: 150 epoch)                     ║
echo   ║   [3] Uzun Egitim (300 epoch, detayli)                         ║
echo   ║   [4] Test Egitimi (10 epoch, hizli test)                      ║
echo   ║   [0] Geri                                                     ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

set /p train_choice="   Seciminiz [0-4]: "

if "%train_choice%"=="1" (
    python src\models\train_tensorflow_model.py
)
if "%train_choice%"=="2" (
    python src\models\train_tensorflow_model.py --epochs 150 --batch-size 64 --limit 100000 --random
)
if "%train_choice%"=="3" (
    python src\models\train_tensorflow_model.py --epochs 300 --batch-size 128 --limit 200000 --random
)
if "%train_choice%"=="4" (
    python src\models\train_tensorflow_model.py --epochs 10 --batch-size 32 --limit 5000 --random
)
if "%train_choice%"=="0" goto MAIN_MENU

echo.
pause
goto MAIN_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              DATABASE MENU
:: ════════════════════════════════════════════════════════════════════════════
:DATABASE_MENU
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║                  VERITABANI ISLEMLERI                          ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

call :ACTIVATE_VENV
echo.

echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║   [1] Database Olustur/Guncelle                                ║
echo   ║   [2] Mock Veri Ekle                                           ║
echo   ║   [3] Database Istatistikleri                                  ║
echo   ║   [4] Database Yedekle                                         ║
echo   ║   [5] Index Olustur (Performans)                               ║
echo   ║   [0] Geri                                                     ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

set /p db_choice="   Seciminiz [0-5]: "

if "%db_choice%"=="1" (
    python scripts\setup_database.py
)
if "%db_choice%"=="2" (
    python scripts\generate_mock_data.py
)
if "%db_choice%"=="3" (
    python -c "from src.utils.database import DatabaseManager; db=DatabaseManager(); import json; print(json.dumps(db.get_database_stats(), indent=2))"
)
if "%db_choice%"=="4" (
    echo   [*] Database yedekleniyor...
    if not exist backups mkdir backups
    copy src\database\cyberguard.db "backups\cyberguard_%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%.db" >nul
    echo   [OK] Yedekleme tamamlandi: backups\ klasorune
)
if "%db_choice%"=="5" (
    python scripts\create_indexes.py
)
if "%db_choice%"=="0" goto MAIN_MENU

echo.
pause
goto DATABASE_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              BAGIMLILIKLARI KUR
:: ════════════════════════════════════════════════════════════════════════════
:INSTALL_DEPS
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║              BAGIMLILIKLARI KURMA                              ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

call :ACTIVATE_VENV
echo.

echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║   [1] Python Paketleri (requirements.txt)                      ║
echo   ║   [2] Node.js Paketleri (frontend)                             ║
echo   ║   [3] Tum Bagimliliklari Kur                                   ║
echo   ║   [4] Sanal Ortam Olustur                                      ║
echo   ║   [0] Geri                                                     ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

set /p dep_choice="   Seciminiz [0-4]: "

if "%dep_choice%"=="1" (
    echo   [*] Python paketleri kuruluyor...
    pip install -r requirements.txt
)
if "%dep_choice%"=="2" (
    echo   [*] Node.js paketleri kuruluyor...
    cd frontend
    npm install
    cd ..
)
if "%dep_choice%"=="3" (
    echo   [*] Tum bagimliliklari kuruluyor...
    pip install -r requirements.txt
    cd frontend
    npm install
    cd ..
    echo   [OK] Tum bagimliliklari kuruldu!
)
if "%dep_choice%"=="4" (
    echo   [*] Sanal ortam olusturuluyor...
    python -m venv .venv
    echo   [OK] .venv olusturuldu. Lutfen launcher'i yeniden baslatin.
)
if "%dep_choice%"=="0" goto MAIN_MENU

echo.
pause
goto INSTALL_DEPS

:: ════════════════════════════════════════════════════════════════════════════
::                              SISTEM DURUMU
:: ════════════════════════════════════════════════════════════════════════════
:HEALTH_CHECK
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║                    SISTEM DURUMU                               ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

call :ACTIVATE_VENV
echo.

echo   [*] Python Surumu:
python --version
echo.

echo   [*] Node.js Surumu:
node --version 2>nul || echo     Node.js yuklu degil!
echo.

echo   [*] pip Surumu:
pip --version
echo.

echo   [*] Kritik Paketler:
python -c "import tensorflow; print('    TensorFlow:', tensorflow.__version__)" 2>nul || echo     TensorFlow: YUKLU DEGIL
python -c "import fastapi; print('    FastAPI:', fastapi.__version__)" 2>nul || echo     FastAPI: YUKLU DEGIL
python -c "import psutil; print('    psutil:', psutil.__version__)" 2>nul || echo     psutil: YUKLU DEGIL
python -c "import sklearn; print('    scikit-learn:', sklearn.__version__)" 2>nul || echo     scikit-learn: YUKLU DEGIL
echo.

echo   [*] Database Durumu:
if exist src\database\cyberguard.db (
    echo     Database: MEVCUT
    for %%A in (src\database\cyberguard.db) do echo     Boyut: %%~zA bytes
) else (
    echo     Database: YOK
)
echo.

echo   [*] API Durumu:
curl -s http://localhost:8000/api/health 2>nul || echo     API: CALISIYOR DEGIL
echo.

pause
goto MAIN_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              CANLI LOG
:: ════════════════════════════════════════════════════════════════════════════
:VIEW_LOGS
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║                    CANLI LOG GORUNTULE                         ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

call :ACTIVATE_VENV
echo.

echo   [*] Son 50 log kaydi getiriliyor...
echo.

curl -s "http://localhost:8000/api/logs?limit=50" 2>nul | python -c "import sys,json; data=json.load(sys.stdin); [print(f\"[{log.get('level','').upper():8}] {log.get('timestamp','')[:19]} - {log.get('message','')}\") for log in data.get('data',[])]" 2>nul || echo   [!] API calismiyor veya log bulunamadi

echo.
pause
goto MAIN_MENU

:: ════════════════════════════════════════════════════════════════════════════
::                              PORT TEMIZLEME
:: ════════════════════════════════════════════════════════════════════════════
:KILL_PORT
:: Port 8000 temizle
echo   [*] Port 8000 kontrol ediliyor...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :8000 ^| findstr LISTENING') do (
    echo   [*] Port 8000 kullanan process bulundu: %%a
    taskkill /F /PID %%a >nul 2>&1
    echo   [OK] Process %%a sonlandirildi
)
:: Port 5173 temizle
echo   [*] Port 5173 kontrol ediliyor...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5173 ^| findstr LISTENING') do (
    echo   [*] Port 5173 kullanan process bulundu: %%a
    taskkill /F /PID %%a >nul 2>&1
    echo   [OK] Process %%a sonlandirildi
)
exit /b

:: ════════════════════════════════════════════════════════════════════════════
::                              TEMIZLIK
:: ════════════════════════════════════════════════════════════════════════════
:CLEANUP
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║                    TEMIZLIK ISLEMLERI                          ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║   [1] Python Cache Temizle (__pycache__)                       ║
echo   ║   [2] Node Modules Temizle (frontend)                          ║
echo   ║   [3] Upload Klasorunu Temizle                                 ║
echo   ║   [4] Tum Cache'leri Temizle                                   ║
echo   ║   [5] Port Temizle (8000/5173)                                 ║
echo   ║   [0] Geri                                                     ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.

set /p clean_choice="   Seciminiz [0-5]: "

if "%clean_choice%"=="1" (
    echo   [*] Python cache temizleniyor...
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
    del /s /q *.pyc 2>nul
    echo   [OK] Python cache temizlendi!
)
if "%clean_choice%"=="2" (
    echo   [*] Node modules temizleniyor...
    if exist frontend\node_modules rd /s /q frontend\node_modules
    echo   [OK] Node modules temizlendi! (npm install gerekli)
)
if "%clean_choice%"=="3" (
    echo   [*] Upload klasoru temizleniyor...
    if exist uploads rd /s /q uploads
    mkdir uploads
    echo   [OK] Upload klasoru temizlendi!
)
if "%clean_choice%"=="4" (
    echo   [*] Tum cache'ler temizleniyor...
    for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
    del /s /q *.pyc 2>nul
    if exist uploads rd /s /q uploads
    mkdir uploads
    echo   [OK] Tum cache'ler temizlendi!
)
if "%clean_choice%"=="5" (
    call :KILL_PORT
    echo   [OK] Port temizleme tamamlandi!
)
if "%clean_choice%"=="0" goto MAIN_MENU

echo.
pause
goto CLEANUP

:: ════════════════════════════════════════════════════════════════════════════
::                              CIKIS
:: ════════════════════════════════════════════════════════════════════════════
:EXIT
cls
echo.
echo   ╔════════════════════════════════════════════════════════════════╗
echo   ║                                                                ║
echo   ║   Gule gule! CyberGuard AI'i kullandiginiz icin tesekkurler.   ║
echo   ║                                                                ║
echo   ╚════════════════════════════════════════════════════════════════╝
echo.
timeout /t 2 >nul
exit