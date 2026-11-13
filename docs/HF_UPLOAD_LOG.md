# HF_UPLOAD_LOG — Đóng gói Docker image & upload Hugging Face

> Ngày thực hiện: 2026-11-13 · Môi trường build: WSL2 Ubuntu + Docker Engine 29.1.3 (containerd image store, overlayfs)

## 1. Build image

| Bước | Lệnh | Kết quả |
|---|---|---|
| Build | `docker build -t htk-no1-app .` | `BUILD_EXIT=0` · context 64.67 MB (GGUF và FAISS zip bị `.dockerignore` loại) |
| Kiểm tra | `docker images htk-no1-app` | `htk-no1-app  3.57GB` |
| Layer chính | `docker history` | base python 3.11-slim + apt 405 MB + torch CPU 989 MB + requirements 1.17 GB + data 43 MB |

Ghi chú: `Dockerfile` cài torch CPU-only (`--index-url https://download.pytorch.org/whl/cpu`) trước khi cài `requirements.txt` để tránh tải ~5 GB wheel CUDA không dùng được trên máy CPU. Model GGUF **không** nằm trong image; `docker-compose.yml` mount `./models` read-only lúc chạy (đúng thiết kế trong spec Dockerfile + .dockerignore).

## 2. Save image + checksum

| Bước | Lệnh | Kết quả |
|---|---|---|
| Save | `docker save htk-no1-app -o htk-no1-app.tar` | `SAVE_EXIT=0` · 806.982.656 B (containerd image store lưu layer dạng blob nén) |
| SHA256 | `python -c "import hashlib; ..."` | `c4e917d24e4fed2bf90fc5da09b6e1c38641616003f572675d037e0002956485` |
| File checksum | `htk-no1-app.tar.sha256` | chứa đúng hash trên |

## 3. Upload Hugging Face — KẾT QUẢ: CHƯA THỰC HIỆN ĐƯỢC (thiếu credential)

| Hạng mục | Giá trị |
|---|---|
| Repo đích | `Huy1223/DPL302m` (repo_type=model) |
| File cần upload | `htk-no1-app.tar`, `htk-no1-app.tar.sha256` |
| Script | `scripts/upload_hf.py` (đọc `os.environ["HF_TOKEN"]`) |
| Kiểm tra biến môi trường | `HF_TOKEN` **không tồn tại** trong shell hiện tại (Windows user env + WSL env đều trống) |
| Chạy script | `ERROR: $HF_TOKEN không có trong môi trường.` (SystemExit) |
| HTTP status | không có (request chưa được gửi) |
| Timestamp thử | 2026-11-13 |

**Nguyên nhân:** môi trường chạy agent không có `HF_TOKEN`; repo GitHub `shirmtry/htk_no1` cũng chưa cấu hình secret `HF_TOKEN` (`gh api repos/shirmtry/htk_no1/actions/secrets` → `total_count: 0`). Agent không được phép xin lại token theo quy tắc nhiệm vụ, nên ghi nhận trung thực và để lại đường dẫn hoàn tất.

**Cách hoàn tất (1 lệnh, sau khi có token):**

```bash
# Windows PowerShell
$env:HF_TOKEN = "hf_xxx"
python scripts/upload_hf.py
```

hoặc thiết lập GitHub secret rồi chạy workflow `docker-build-upload.yml` (Actions → Run workflow). Workflow tự build, save, tính SHA256 và upload cả 2 file lên `Huy1223/DPL302m`.

## 4. Dọn file nặng

| File | Trạng thái |
|---|---|
| `htk-no1-app.tar` (Windows) | ĐÃ XÓA sau khi tính checksum |
| `/tmp/htk-no1-app.tar` (WSL) | ĐÃ XÓA |
| `htk-no1-app.tar.sha256` | GIỮ LẠI — đưa vào `source/` của ZIP nộp bài |

File `.tar` không nằm trong ZIP nộp bài theo quy tắc số 9.
