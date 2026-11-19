#!/usr/bin/env bash
# Tải lại slide WSI (.svs, open-access) từ GDC theo wsi_gdc_manifest.tsv.
# Chạy: bash download_wsi.sh <thư_mục_đích> [số_luồng]
# Chạy lại an toàn: file đã khớp md5 thì bỏ qua, file dở dang được tải tiếp (curl -C -).
set -euo pipefail

MANIFEST="$(dirname "$0")/wsi_gdc_manifest.tsv"
DEST="${1:?Cần thư mục đích}"
JOBS="${2:-4}"
mkdir -p "$DEST"

fetch_one() {
  local file_id="$1" file_name="$2" md5="$3" dest="$4"
  local out="$dest/$file_name"
  if [[ -f "$out" ]] && [[ "$(md5sum "$out" | cut -d' ' -f1)" == "$md5" ]]; then
    echo "SKIP $file_name"; return 0
  fi
  for attempt in 1 2 3 4 5; do
    if curl -sS -L --fail -C - --retry 3 -o "$out" "https://api.gdc.cancer.gov/data/$file_id"; then
      if [[ "$(md5sum "$out" | cut -d' ' -f1)" == "$md5" ]]; then
        echo "OK $file_name"; return 0
      fi
      echo "MD5_MISMATCH $file_name (lần $attempt), tải lại từ đầu"; rm -f "$out"
    else
      echo "CURL_FAIL $file_name (lần $attempt)"
    fi
    sleep 5
  done
  echo "FAILED $file_name"; return 1
}
export -f fetch_one
export DEST

# Tên file TCGA không chứa khoảng trắng nên tách tham số theo khoảng trắng là an toàn.
tail -n +2 "$MANIFEST" | awk -F'\t' '{print $2, $3, $5}' \
  | xargs -P "$JOBS" -n 3 bash -c 'fetch_one "$1" "$2" "$3" "$DEST"' _ \
  || true

echo "=== Tổng kết ==="
echo "Đã có: $(ls "$DEST"/*.svs 2>/dev/null | wc -l) / $(($(wc -l < "$MANIFEST") - 1)) slide"
