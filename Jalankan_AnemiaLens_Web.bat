@echo off
chcp 65001 >nul
title AnemiaLens Web
cd /d "%~dp0"

rem ---- cari Python ----
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY goto nopython

rem ---- siapkan lingkungan (hanya sekali) ----
if not exist ".venv\siap.txt" (
  echo.
  echo  [1/2] Menyiapkan lingkungan Python. Hanya sekali, sekitar 5-10 menit...
  echo.
  if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv || goto nopython
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r web\requirements.txt || goto pipfail
  echo ok> ".venv\siap.txt"
)

echo.
echo  [2/2] Menjalankan AnemiaLens Web. Browser akan terbuka otomatis.
echo        Jika Windows Firewall bertanya, pilih "Allow" agar HP bisa membuka web ini.
echo.
".venv\Scripts\python.exe" web\server.py %*
pause
exit /b 0

:nopython
echo.
echo  Python belum terpasang.
echo  Install Python 3.11 atau 3.12 dari https://www.python.org/downloads/
echo  (centang "Add python.exe to PATH"), lalu jalankan file ini lagi.
pause
exit /b 1

:pipfail
echo.
echo  Instalasi paket gagal. Periksa koneksi internet, lalu jalankan file ini lagi.
echo  Jika tetap gagal: hapus folder .venv di folder AnemiaLens, lalu jalankan ulang.
pause
exit /b 1
