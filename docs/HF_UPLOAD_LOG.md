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

## 3. Upload Hugging Face DATASET — KẾT QUẢ: CHƯA THỰC HIỆN ĐƯỢC (thiếu credential)

> Cập nhật nhiệm vụ 6: repo đích là **dataset** (`repo_type="dataset"`), không phải model. `scripts/upload_hf.py` và `.github/workflows/docker-build-upload.yml` đã được sửa đúng chuẩn dataset.

| Hạng mục | Giá trị |
|---|---|
| Repo đích | `Huy1223/DPL302m` · `repo_type="dataset"` |
| Kiểm tra repo (anonymous) | `dataset_info("Huy1223/DPL302m")` → **EXISTS**, `private=False`, files: `.gitattributes` |
| File cần upload | `htk-no1-app.tar` (806.982.656 B), `htk-no1-app.tar.sha256` |
| SHA256 (tính lại 2026-11-13) | `c4e917d24e4fed2bf90fc5da09b6e1c38641616003f572675d037e0002956485` |
| Script | `scripts/upload_hf.py` (đọc `os.environ.get("HF_TOKEN")`, upload bằng `HfApi.upload_file`, `repo_type="dataset"`) |
| Kiểm tra biến môi trường | `HF_TOKEN` **không tồn tại** trong Git Bash, `cmd`, và WSL (`.env` chỉ chứa khoá Groq, không phải HF) |
| Chạy script | `venv/Scripts/python.exe scripts/upload_hf.py` → exit 1 |
| Lỗi thực tế | `huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error` tại `https://huggingface.co/api/datasets/Huy1223/DPL302m/preupload/main` — "Invalid username or password." |
| Ý nghĩa lỗi | Request đã đi đúng endpoint **datasets** (xác nhận `repo_type="dataset"`), nhưng không có token hợp lệ nên HF trả 401 |
| Log thô | `TEMP/hf_upload_attempt.log` (không nằm trong ZIP nộp bài) |

**Nguyên nhân:** môi trường agent không có `HF_TOKEN`; repo GitHub `shirmtry/htk_no1` chưa cấu hình secret `HF_TOKEN` (`gh api repos/shirmtry/htk_no1/actions/secrets` → `total_count: 0`). Theo quy tắc nhiệm vụ, agent không hỏi lại token mà ghi nhận trung thực và để sẵn đường hoàn tất 1 lệnh.

**Cách hoàn tất (sau khi có token, không cần sửa code):**

```bash
# Git Bash / WSL
export HF_TOKEN=hf_xxx
venv/Scripts/python.exe scripts/upload_hf.py
```

hoặc thiết lập GitHub secret `HF_TOKEN` rồi chạy workflow `docker-build-upload.yml` (Actions → Run workflow). Workflow tự build, save, tính SHA256 và upload cả 2 file lên dataset `Huy1223/DPL302m`.

## 4. Dọn file nặng

| File | Trạng thái |
|---|---|
| `htk-no1-app.tar` (project root) | GIỮ LẠI để upload ngay khi có token — bị loại khỏi ZIP nộp bài theo quy tắc |
| `htk-no1-app.tar.sha256` | GIỮ LẠI (64 B hex + tên file), bị loại khỏi ZIP nộp bài |

File `.tar`, `.sha256`, log, `.webm` không nằm trong ZIP nộp bài theo quy tắc nhiệm vụ 6.
