@echo off
setlocal EnableExtensions
chcp 65001 >nul
title AnemiaLens - Link Publik
cd /d "%~dp0"

echo.
echo  AnemiaLens - membuat link publik (bisa dibuka dari HP mana pun, tanpa satu WiFi)
echo.

rem ===== 1. Unduh cloudflared (sekali saja) =====
if exist "tools\cloudflared.exe" goto server
echo  [1/3] Mengunduh cloudflared dari GitHub resmi Cloudflare (sekali saja, sekitar 60 MB)...
if not exist tools mkdir tools
curl -L --fail -o "tools\cloudflared.exe" https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe || goto dlfail

rem ===== 2. Jalankan server bila belum jalan =====
:server
curl -s -o nul http://localhost:8000/api/info && goto tunnel
echo  [2/3] Menjalankan server AnemiaLens di jendela lain...
start "AnemiaLens Web" cmd /k Jalankan_AnemiaLens_Web.bat --no-browser
set /a N=0
:wait
timeout /t 3 /nobreak >nul
curl -s -o nul http://localhost:8000/api/info && goto tunnel
set /a N+=1
if %N% GEQ 240 goto srvfail
goto wait

rem ===== 3. Buat link publik =====
:tunnel
echo  [3/3] Membuat link publik...
echo.
echo  ============================================================
echo   Cari baris berisi  https://....trycloudflare.com  di bawah,
echo   lalu kirim link itu ke teman Anda.
echo   Link aktif selama jendela ini DAN jendela server terbuka.
echo   Tutup jendela ini untuk mematikan link.
echo  ============================================================
echo.
"tools\cloudflared.exe" tunnel --url http://localhost:8000 --no-autoupdate
pause
exit /b 0

:dlfail
echo  Gagal mengunduh cloudflared. Periksa internet lalu jalankan lagi.
del "tools\cloudflared.exe" 2>nul
pause
exit /b 1

:srvfail
echo  Server tidak kunjung siap. Lihat pesan error di jendela "AnemiaLens Web".
pause
exit /b 1
