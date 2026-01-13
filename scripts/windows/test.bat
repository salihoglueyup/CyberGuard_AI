@echo off
REM ========================================
REM CyberGuard AI - Test Runner
REM ========================================

echo.
echo ========================================
echo   CyberGuard AI Test Suite
echo ========================================
echo.

REM Sanal ortam kontrolü
if not exist venv (
    echo [HATA] Sanal ortam bulunamadi!
    echo Once setup.bat calistirin.
    pause
    exit /b 1
)

echo [1/3] Sanal ortam aktive ediliyor...
call venv\Scripts\activate.bat
echo [OK] Sanal ortam aktif

echo.
echo [2/3] Test ortami hazirlaniyor...
set TESTING=1
set DATABASE_URL=sqlite:///test.db
echo [OK] Test ortami hazir

echo.
echo [3/3] Testler calistiriliyor...
echo.

REM Argüman kontrolü
if "%1"=="unit" goto unit_tests
if "%1"=="integration" goto integration_tests
if "%1"=="e2e" goto e2e_tests
if "%1"=="coverage" goto coverage_tests
if "%1"=="quick" goto quick_tests

REM Default: Tüm testler
:all_tests
echo [TEST] Tum testler calistiriliyor...
pytest tests/ -v --tb=short
goto end

:unit_tests
echo [TEST] Unit testler calistiriliyor...
pytest tests/unit/ -v
goto end

:integration_tests
echo [TEST] Integration testler calistiriliyor...
pytest tests/integration/ -v
goto end

:e2e_tests
echo [TEST] End-to-end testler calistiriliyor...
pytest tests/e2e/ -v
goto end

:coverage_tests
echo [TEST] Coverage analizi yapiliyor...
pytest tests/ --cov=src --cov-report=html --cov-report=term
echo.
echo [OK] Coverage raporu olusturuldu: htmlcov/index.html
start htmlcov\index.html
goto end

:quick_tests
echo [TEST] Hizli testler calistiriliyor...
pytest tests/ -v -m "not slow"
goto end

:end
echo.
if %errorlevel% equ 0 (
    echo ========================================
    echo   TUM TESTLER BASARILI!
    echo ========================================
) else (
    echo ========================================
    echo   BAZI TESTLER BASARISIZ!
    echo   Hata Kodu: %errorlevel%
    echo ========================================
)
echo.

REM Kullanım bilgisi
if "%1"=="" (
    echo.
    echo Kullanim:
    echo   test.bat              - Tum testler
    echo   test.bat unit         - Sadece unit testler
    echo   test.bat integration  - Sadece integration testler
    echo   test.bat e2e          - Sadece e2e testler
    echo   test.bat coverage     - Coverage analizi
    echo   test.bat quick        - Hizli testler
    echo.
)

pause
deactivate 2>nul