#!/usr/bin/env bash
# Upload / update AnemiaLens ke GitHub (macOS / Linux / Git Bash).
# Pakai:  ./push_ke_github.sh            -> pesan commit otomatis
#         ./push_ke_github.sh "pesan"    -> pesan commit sendiri
set -e
cd "$(dirname "$0")"

REPO="${ANEMIALENS_REPO:-https://github.com/brianmacrio/AnemiaLens.git}"
BRANCH="main"
MSG="${1:-Update AnemiaLens $(date '+%Y-%m-%d %H:%M')}"

command -v git >/dev/null || { echo "Git belum terpasang: https://git-scm.com/downloads"; exit 1; }

# Identitas commit (sekali saja)
if [ -z "$(git config user.name || true)" ]; then read -rp "Nama untuk commit GitHub: " N; git config --global user.name "$N"; fi
if [ -z "$(git config user.email || true)" ]; then read -rp "Email akun GitHub: " E; git config --global user.email "$E"; fi

# Repo lokal (pertama kali)
if [ ! -d .git ]; then
  git init -q -b "$BRANCH"
  echo "[1/5] Repo git lokal dibuat."
fi
if git remote get-url origin >/dev/null 2>&1; then git remote set-url origin "$REPO"; else git remote add origin "$REPO"; fi

echo "[2/5] Mengambil isi GitHub..."
git fetch -q origin || { echo "Gagal menghubungi GitHub. Periksa internet / alamat repo."; exit 1; }
# Pertama kali: sambungkan ke riwayat yang sudah ada di GitHub tanpa mengubah file lokal
if ! git rev-parse --verify -q HEAD >/dev/null && git rev-parse --verify -q "origin/$BRANCH" >/dev/null; then
  git reset -q "origin/$BRANCH"
fi

echo "[3/5] Memeriksa file besar (> 95 MB tidak bisa di-push ke GitHub)..."
BIG=$(git ls-files -o -m --exclude-standard -z | xargs -0 -I{} find {} -maxdepth 0 -type f -size +95M 2>/dev/null || true)
if [ -n "$BIG" ]; then echo "File terlalu besar, tambahkan ke .gitignore dulu:"; echo "$BIG"; exit 1; fi

echo "[4/5] Commit..."
git add -A
if git diff --cached --quiet; then echo "      Tidak ada perubahan baru."; else git commit -q -m "$MSG"; git log --oneline -1; fi

echo "[5/5] Push ke $REPO ..."
if git rev-parse --verify -q "origin/$BRANCH" >/dev/null && ! git pull -q --rebase origin "$BRANCH"; then
  git rebase --abort 2>/dev/null || true
  echo "Ada perubahan di GitHub yang bentrok dengan file lokal. Tidak ada yang di-push."
  echo "Buka repo di GitHub, cek file yang diubah di sana, lalu jalankan script ini lagi."
  exit 1
fi
git push -u origin "$BRANCH"
echo "Selesai: ${REPO%.git}"
