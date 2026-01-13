@echo off
REM ========================================
REM CyberGuard AI - Database Management
REM ========================================

echo.
echo ========================================
echo   CyberGuard AI Database Manager
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

REM Komut kontrolü
if "%1"=="init" goto db_init
if "%1"=="migrate" goto db_migrate
if "%1"=="upgrade" goto db_upgrade
if "%1"=="downgrade" goto db_downgrade
if "%1"=="reset" goto db_reset
if "%1"=="backup" goto db_backup
if "%1"=="restore" goto db_restore
if "%1"=="seed" goto db_seed
if "%1"=="status" goto db_status

REM Default: Yardım göster
:show_help
echo Kullanim: db.bat [komut]
echo.
echo Komutlar:
echo   init       - Veritabanini ilk kez olustur
echo   migrate    - Yeni migration olustur
echo   upgrade    - Son migration'a guncelle
echo   downgrade  - Bir onceki migration'a don
echo   reset      - Veritabanini sifirla (TEHLIKELI!)
echo   backup     - Veritabani yedeği al
echo   restore    - Yedeği geri yukle
echo   seed       - Test verileri yukle
echo   status     - Veritabani durumunu goster
echo.
goto end

:db_init
echo [INIT] Veritabani olusturuluyor...
echo.

if exist src\database\init_db.py (
    python src\database\init_db.py
    if %errorlevel% equ 0 (
        echo.
        echo [OK] Veritabani basariyla olusturuldu!
    ) else (
        echo.
        echo [HATA] Veritabani olusturulamadi!
        pause
        exit /b 1
    )
) else (
    echo [HATA] init_db.py bulunamadi!
    pause
    exit /b 1
)
goto end

:db_migrate
echo [MIGRATE] Yeni migration olusturuluyor...
echo.

set /p message="Migration mesaji: "
if "%message%"=="" (
    echo [HATA] Migration mesaji bos olamaz!
    pause
    exit /b 1
)

REM Alembic kullanımı (varsa)
if exist alembic.ini (
    alembic revision --autogenerate -m "%message%"
    echo [OK] Migration olusturuldu
) else (
    echo [UYARI] Alembic konfigurasyonu bulunamadi
    echo Manual migration gerekiyor
)
goto end

:db_upgrade
echo [UPGRADE] Veritabani guncellenior...
echo.

if exist alembic.ini (
    alembic upgrade head
    if %errorlevel% equ 0 (
        echo [OK] Veritabani guncellendi
    ) else (
        echo [HATA] Guncelleme basarisiz!
        pause
        exit /b 1
    )
) else (
    echo [UYARI] Alembic bulunamadi
)
goto end

:db_downgrade
echo [DOWNGRADE] Veritabani onceki versiyona dondurucluyor...
echo.
echo [UYARI] Bu islem geri alinabilir!
set /p confirm="Devam etmek istiyor musunuz? (E/H): "
if /i not "%confirm%"=="E" (
    echo Islem iptal edildi.
    goto end
)

if exist alembic.ini (
    alembic downgrade -1
    if %errorlevel% equ 0 (
        echo [OK] Downgrade basarili
    ) else (
        echo [HATA] Downgrade basarisiz!
    )
) else (
    echo [UYARI] Alembic bulunamadi
)
goto end

:db_reset
echo [RESET] Veritabani SIFIRLANIYOR...
echo.
echo [UYARI] Bu islem TUM verileri silecek!
echo [UYARI] Bu islem geri alinamaz!
echo.
set /p confirm1="Emin misiniz? (EVET yazin): "
if /i not "%confirm1%"=="EVET" (
    echo Islem iptal edildi.
    goto end
)

echo.
echo Son bir kez soruluyor...
set /p confirm2="Veritabanini sifirla (SIFIRLA yazin): "
if /i not "%confirm2%"=="SIFIRLA" (
    echo Islem iptal edildi.
    goto end
)

echo.
echo [1/3] Mevcut veritabani siliniyor...
if exist data\cyberguard.db del /q data\cyberguard.db
if exist *.db del /q *.db
echo [OK] Veritabani silindi

echo [2/3] Yeni veritabani olusturuluyor...
python src\database\init_db.py
echo [OK] Yeni veritabani olusturuldu

echo [3/3] Migrations sifirlaniyor...
if exist alembic\versions rd /s /q alembic\versions
mkdir alembic\versions
echo [OK] Migrations sifirlandi

echo.
echo [OK] Veritabani sifirlama tamamlandi!
goto end

:db_backup
echo [BACKUP] Veritabani yedeği aliniyor...
echo.

REM Backup klasörü oluştur
if not exist data\backups mkdir data\backups

REM Tarih-saat damgası
for /f "tokens=2 delims==" %%a in ('wmic OS Get localdatetime /value') do set "dt=%%a"
set "timestamp=%dt:~0,4%%dt:~4,2%%dt:~6,2%_%dt:~8,2%%dt:~10,2%%dt:~12,2%"

REM SQLite için
if exist data\cyberguard.db (
    copy data\cyberguard.db data\backups\cyberguard_%timestamp%.db
    echo [OK] Yedek olusturuldu: data\backups\cyberguard_%timestamp%.db
) else (
    echo [UYARI] Veritabani dosyasi bulunamadi
)

REM PostgreSQL için (opsiyonel)
REM pg_dump -U postgres -d cyberguard > data\backups\cyberguard_%timestamp%.sql

echo.
echo Yedek listesi:
dir data\backups\*.db /b
goto end

:db_restore
echo [RESTORE] Veritabani geri yukleniyor...
echo.

if not exist data\backups (
    echo [HATA] Backup klasoru bulunamadi!
    pause
    exit /b 1
)

echo Mevcut yedekler:
echo.
dir /b data\backups\*.db

echo.
set /p backup_file="Geri yuklenecek dosya adi: "

if not exist "data\backups\%backup_file%" (
    echo [HATA] Dosya bulunamadi!
    pause
    exit /b 1
)

echo.
echo [UYARI] Mevcut veritabani uzerine yazilacak!
set /p confirm="Devam edilsin mi? (E/H): "
if /i not "%confirm%"=="E" (
    echo Islem iptal edildi.
    goto end
)

copy /Y "data\backups\%backup_file%" data\cyberguard.db
echo [OK] Veritabani geri yuklendi!
goto end

:db_seed
echo [SEED] Test verileri yukleniyor...
echo.

if exist src\database\seed.py (
    python src\database\seed.py
    if %errorlevel% equ 0 (
        echo [OK] Test verileri yuklendi
        echo.
        echo Eklenen veriler:
        echo   - Admin kullanicisi
        echo   - Ornek tarama sonuclari
        echo   - Test raporlari
    ) else (
        echo [HATA] Test verileri yuklenemedi!
    )
) else (
    echo [UYARI] seed.py bulunamadi
    echo Manual veri ekleme gerekiyor
)
goto end

:db_status
echo [STATUS] Veritabani durumu...
echo.

REM SQLite için
if exist data\cyberguard.db (
    echo [OK] Veritabani mevcut: data\cyberguard.db
    echo.
    echo Boyut:
    dir data\cyberguard.db

    echo.
    echo Tablo sayisi:
    sqlite3 data\cyberguard.db "SELECT COUNT(*) FROM sqlite_master WHERE type='table';"
) else (
    echo [UYARI] Veritabani dosyasi bulunamadi
)

echo.
REM Alembic migration durumu
if exist alembic.ini (
    echo Migration durumu:
    alembic current
    echo.
    echo Migration gecmisi:
    alembic history
) else (
    echo [INFO] Alembic konfigurasyonu yok
)

goto end

:end
echo.
pause
deactivate 2>nul