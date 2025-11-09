@echo off
REM ========================================
REM CyberGuard AI - Temizlik Script
REM ========================================

echo.
echo ========================================
echo   CyberGuard AI Temizlik
echo ========================================
echo.

echo [UYARI] Bu islem su dosyalari/klasorleri silecek:
echo   - __pycache__ klasorleri
echo   - .pyc, .pyo dosyalari
echo   - Test dosyalari (.pytest_cache, htmlcov)
echo   - Log dosyalari (logs/)
echo   - Gecici dosyalar (temp/, tmp/)
echo   - Build klasorleri (build/, dist/)
echo.

set /p confirm="Devam etmek istiyor musunuz? (E/H): "
if /i not "%confirm%"=="E" (
    echo Islem iptal edildi.
    pause
    exit /b 0
)

echo.
echo Temizlik baslatiliyor...
echo.

REM Python cache dosyalarını temizle
echo [1/10] Python cache dosyalari temizleniyor...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
del /s /q *.pyo 2>nul
del /s /q *.pyd 2>nul
echo [OK] Python cache temizlendi

REM .pytest_cache temizle
echo [2/10] Pytest cache temizleniyor...
if exist .pytest_cache rd /s /q .pytest_cache
echo [OK] Pytest cache temizlendi

REM Coverage dosyalarını temizle
echo [3/10] Coverage dosyalari temizleniyor...
if exist htmlcov rd /s /q htmlcov
if exist .coverage del /q .coverage
if exist coverage.xml del /q coverage.xml
echo [OK] Coverage dosyalari temizlendi

REM Log dosyalarını temizle
echo [4/10] Log dosyalari temizleniyor...
if exist logs (
    del /q logs\*.log 2>nul
    echo [OK] Log dosyalari temizlendi
) else (
    echo [INFO] logs/ klasoru bulunamadi
)

REM Geçici dosyaları temizle
echo [5/10] Gecici dosyalar temizleniyor...
if exist temp rd /s /q temp
if exist tmp rd /s /q tmp
del /q *.tmp 2>nul
del /q *.temp 2>nul
echo [OK] Gecici dosyalar temizlendi

REM Build klasörlerini temizle
echo [6/10] Build klasorleri temizleniyor...
if exist build rd /s /q build
if exist dist rd /s /q dist
if exist *.egg-info rd /s /q *.egg-info
echo [OK] Build klasorleri temizlendi

REM IDE cache temizle
echo [7/10] IDE cache temizleniyor...
if exist .idea rd /s /q .idea
if exist .vscode\*.log del /q .vscode\*.log 2>nul
if exist *.iml del /q *.iml 2>nul
echo [OK] IDE cache temizlendi

REM Test database temizle
echo [8/10] Test veritabani temizleniyor...
if exist test.db del /q test.db
if exist *.db-journal del /q *.db-journal 2>nul
echo [OK] Test veritabani temizlendi

REM Node modules temizle (varsa)
echo [9/10] Node modules temizleniyor...
if exist node_modules rd /s /q node_modules
echo [OK] Node modules temizlendi

REM Backup dosyaları temizle
echo [10/10] Backup dosyalari temizleniyor...
del /q *.bak 2>nul
del /q *.old 2>nul
del /q *~ 2>nul
echo [OK] Backup dosyalari temizlendi

echo.
echo ========================================
echo   TEMIZLIK TAMAMLANDI!
echo ========================================
echo.

REM Boyut hesaplama (opsiyonel)
echo Disk alanı kazanildi.
echo.

pause