# GITHUB_CLEANUP_LOG — Dọn rác repo shirmtry/htk_no1

> Thực hiện ngày 2026-11-13. Nguyên tắc: **chỉ xóa khỏi git index (`git rm --cached`), GIỮ NGUYÊN file trên máy local.**

## 1. Quét repo

Lệnh quét:

```bash
git ls-files | grep -iE '(^|/)(venv|\.venv|__pycache__|\.pytest_cache|\.mypy_cache|\.idea|\.vscode)/|\.(pyc|pyo|pyd|tmp|swp|swo|log)$|\.bak|error_log\.csv|\.gguf|faiss_index.*\.zip|^\.env|Thumbs\.db|\.DS_Store|coverage/|htmlcov/'
```

Kết quả: **3 file rác được track** trên GitHub.

| File | Lý do | Lệnh đã chạy |
|---|---|---|
| `app.py.bak_round0` | File backup `*.bak*` — không thuộc mã nguồn | `git rm --cached app.py.bak_round0` |
| `app.py.bak_round2` | File backup `*.bak*` — không thuộc mã nguồn | `git rm --cached app.py.bak_round2` |
| `error_log.csv` | Log lỗi sinh ra khi chạy eval, không thuộc deliverable | `git rm --cached error_log.csv` |

Sau khi xóa khỏi index, kiểm tra file local vẫn còn: `ls app.py.bak_round0 app.py.bak_round2 error_log.csv` → OK.

## 2. Không có trên GitHub (đã sạch từ trước)

Các mục sau **không hề được track** (đã nằm trong .gitignore từ trước): `venv/`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `*.pyc`, `*.gguf`, `faiss_index*.zip`, `.env`, `logs/`, `*.log`, `TEMP/`, `models/`, `data/`.

## 3. Cập nhật .gitignore

Đã ghi lại `.gitignore` ở gốc repo với đúng danh sách yêu cầu:

```
venv/  .venv/  __pycache__/  *.pyc  *.pyo  *.pyd
.pytest_cache/  .mypy_cache/
*.bak*  *.tmp  *.swp  *.swo  *~  .DS_Store  Thumbs.db  .idea/  .vscode/
logs/  *.log  chatbot.log  streamlit_console.log  *.error.log  error_log.csv  coverage/  htmlcov/
models/  data/  *.gguf  faiss_index*.zip  .env  .env.local  TEMP/
eval_out/  eval_out_agg/
```

Ghi chú: giữ thêm `models/` và `data/` (file nặng ~2,1 GB GGUF + JSONL + FAISS được phát hành qua GitHub Releases, không commit vào git) và `eval_out*/` (kết quả runtime của harness).

## 4. Commit & push

```bash
git commit -m "chore: untrack junk files (.bak, error_log) and harden .gitignore"
git push origin main
```

Kết quả: commit `f433a00` — `4 files changed, 35 insertions(+), 2031 deletions(-)` — push `6f85ef3..f433a00 main -> main` thành công.

## 5. File rác còn trên máy LOCAL (giữ nguyên, không đưa vào ZIP)

`app.py.bak_round0`, `app.py.bak_round2`, `error_log.csv`, `structure.txt`, `__pycache__/`, `venv/`, `logs/` (log cũ), `eval_out/`, `data/faiss_index_new_v1.zip`, `models/*.gguf`.
