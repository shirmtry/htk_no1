# Chatbot Pháp luật VN v2.0

Hệ thống RAG tra cứu 9.159 điều luật từ 65 Bộ luật/Luật Việt Nam.

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
