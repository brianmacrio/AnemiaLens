@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title Push AnemiaLens ke GitHub
cd /d "%~dp0"

rem ===== Pengaturan =====
set "REPO=https://github.com/brianmacrio/AnemiaLens.git"
if defined ANEMIALENS_REPO set "REPO=%ANEMIALENS_REPO%"
set "BRANCH=main"
set "MSG=%~1"
if "%MSG%"=="" set "MSG=Update AnemiaLens %date% %time:~0,5%"

echo.
echo  Upload AnemiaLens ke %REPO%
echo.

where git >nul 2>nul || goto nogit

rem ===== Identitas commit (hanya ditanya sekali) =====
set "GN="
for /f "delims=" %%a in ('git config user.name 2^>nul') do set "GN=%%a"
if not defined GN (
  set /p "GN=Nama untuk commit GitHub: "
  git config --global user.name "!GN!"
)
set "GE="
for /f "delims=" %%a in ('git config user.email 2^>nul') do set "GE=%%a"
if not defined GE (
  set /p "GE=Email akun GitHub: "
  git config --global user.email "!GE!"
)

rem ===== Repo lokal (pertama kali) =====
if not exist ".git" (
  git init -q -b %BRANCH% 2>nul || (git init -q && git symbolic-ref HEAD refs/heads/%BRANCH%)
  echo  [1/5] Repo git lokal dibuat.
)
git remote get-url origin >nul 2>nul && (git remote set-url origin "%REPO%") || (git remote add origin "%REPO%")

echo  [2/5] Mengambil isi GitHub...
git fetch -q origin || goto nonet
rem Pertama kali: sambungkan ke riwayat yang sudah ada di GitHub tanpa mengubah file lokal
git rev-parse --verify -q HEAD >nul 2>nul
if errorlevel 1 (
  git rev-parse --verify -q origin/%BRANCH% >nul 2>nul && git reset -q origin/%BRANCH%
)

echo  [3/5] Memeriksa file besar (lebih dari 95 MB tidak bisa di-push ke GitHub)...
set "BIG="
for /f "delims=" %%f in ('git ls-files -o -m --exclude-standard') do (
  if exist "%%f" if %%~zf GTR 99614720 (
    echo        Terlalu besar: %%f
    set "BIG=1"
  )
)
if defined BIG goto toobig

echo  [4/5] Commit...
git add -A
git diff --cached --quiet
if errorlevel 1 (
  git commit -q -m "%MSG%" || goto fail
  git log --oneline -1
) else (
  echo        Tidak ada perubahan baru.
)

echo  [5/5] Push... (jika diminta, login ke akun GitHub di jendela browser yang muncul)
git rev-parse --verify -q origin/%BRANCH% >nul 2>nul
if not errorlevel 1 (
  git pull -q --rebase origin %BRANCH%
  if errorlevel 1 (
    git rebase --abort >nul 2>nul
    goto conflict
  )
)
git push -u origin %BRANCH% || goto pushfail
echo.
echo  Selesai! Buka: %REPO:.git=%
echo.
pause
exit /b 0

:nogit
echo  Git belum terpasang. Install dari https://git-scm.com/download/win
echo  (pilihan default saja), lalu jalankan file ini lagi.
pause
exit /b 1

:nonet
echo  Gagal menghubungi GitHub. Periksa internet dan alamat repo: %REPO%
pause
exit /b 1

:toobig
echo  Ada file lebih dari 95 MB. Tambahkan namanya ke .gitignore, lalu jalankan lagi.
pause
exit /b 1

:conflict
echo  Ada perubahan di GitHub yang bentrok dengan file lokal. Tidak ada yang di-push.
echo  Buka repo di GitHub, cek file yang diubah di sana, lalu jalankan file ini lagi.
pause
exit /b 1

:pushfail
echo  Push gagal. Pastikan login GitHub berhasil dan akun Anda punya akses ke repo ini.
pause
exit /b 1

:fail
echo  Terjadi kesalahan git. Baca pesan di atas.
pause
exit /b 1
