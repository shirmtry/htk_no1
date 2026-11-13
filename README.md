# Chatbot Pháp luật VN v2.0

Hệ thống RAG tra cứu 9.159 điều luật từ 65 Bộ luật/Luật Việt Nam.

- Repository: https://github.com/shirmtry/htk_no1
- Docker image (tar + sha256): https://huggingface.co/Huy1223/DPL302m
- Kết quả đánh giá: 23/23 metric PASS · pass_rate 1.0000 · citation_f1 1.0000 · refusal_accuracy 1.0000 · avg latency 110 ms

## Hướng dẫn cài đặt

**Bước 1**: Clone repo:

```bash
git clone https://github.com/shirmtry/htk_no1.git
cd htk_no1
```

**Bước 2**: Tải file `qwen2.5-3b-instruct-q4_k_m.gguf` (~2,1 GB) từ tab [Releases](https://github.com/shirmtry/htk_no1/releases/tag/v1.0.0) và bỏ vào thư mục `models/`.

**Bước 3**: Tải file `faiss_index_new_v1.zip` từ tab Releases, giải nén và bỏ vào thư mục `data/` (thành `data/faiss_index_new_v1/`). Tải thêm file `HUYTKING.cleaned.jsonl` từ Releases và bỏ vào thư mục `data/` — đây là dữ liệu gốc bắt buộc để chạy.

**Bước 4**: Cài đặt requirements và chạy:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Mở http://localhost:8501. Đánh giá tự động: `python -X utf8 run_eval.py --phase=full --tier=all --config=improved`.

## Cài đặt

```bash
cd C:\Users\dinht\Downloads\Bo_Cai_Chatbot_Luat_v2
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Cấu trúc
- `app.py` — single-file, ~770 dòng
- `data/HUYTKING.cleaned.jsonl` — 9.159 điều luật
- `data/faiss_index_new_v1/` — vector DB
- `models/qwen2.5-3b-instruct-q4_k_m.gguf` — LLM (2,10 GB)

## Chạy
```bash
streamlit run app.py
```

Mở http://localhost:8501

## Test
7 câu demo:
1. Con tôi bị khuyết tật, không được ghi trong di chúc, có được hưởng không?
2. Cháu nội tôi có được thừa kế thay bố không?
3. Công chứng di chúc tại UBND xã thủ tục thế nào?
4. Điều 12 Bộ luật Hình sự quy định gì?
5. Người 13 tuổi phạm tội giết người bị xử lý thế nào?
6. Người 15 tuổi phạm tội giết người bị xử lý thế nào?
7. Án treo là gì?

## Docker

```bash
docker compose up -d --build
```

Mở https://localhost (Caddy TLS nội bộ, reverse proxy tới Streamlit 8501). Kiểm tra:

```bash
docker compose ps
curl -k https://localhost/_stcore/health
```

Image đóng gói: `docker build -t htk-no1-app . && docker save htk-no1-app -o htk-no1-app.tar` — bản phát hành kèm SHA256 tại [Huy1223/DPL302m](https://huggingface.co/Huy1223/DPL302m). CI tự build và upload khi push `main` (`.github/workflows/docker-build-upload.yml`, dùng `secrets.HF_TOKEN`).
