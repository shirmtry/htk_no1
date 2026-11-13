# BÁO CÁO DỰ ÁN: CHATBOT PHÁP LUẬT VIỆT NAM v2.0

| Thông tin | Nội dung |
|----------------------|------------------------------------------------------------------|
| Nhóm thực hiện | Dinh Tan Huy |
| Repository | [github.com/shirmtry/htk_no1](https://github.com/shirmtry/htk_no1) |
| Hugging Face | [huggingface.co/datasets/Huy1223/DPL302m](https://huggingface.co/datasets/Huy1223/DPL302m) |
| Model | Qwen2.5-3B-Instruct Q4_K_M GGUF (llama-cpp-python) |
| Dữ liệu | 9.159 điều luật / 65 Bộ luật–Luật / reference_map 2.411 mục |
| Kết quả | 23/23 metric PASS · pass_rate 1.0000 · citation_f1 1.0000 · refusal 1.0000 · avg 110 ms |

## PHẦN 1 — GIỚI THIỆU CHUNG

### 1.1. Bối cảnh

Người dân và sinh viên luật thường phải tra cứu thủ công hàng nghìn điều luật rải rác trong các Bộ luật/Luật Việt Nam. Một trợ lý hỏi–đáp tiếng Việt có khả năng trích dẫn đúng điều luật sẽ giảm thời gian tra cứu và hạn chế trả lời sai. Dự án xây dựng hệ thống RAG (Retrieval-Augmented Generation) chạy offline trên CPU, không phụ thuộc API trả phí.

### 1.2. Mục tiêu

- Trả lời câu hỏi pháp luật tiếng Việt dựa trên kho 9.159 điều luật, có trích dẫn nguồn cụ thể.
- Xử lý 5 nhóm câu hỏi: tổng quan, chi tiết, ngoại lệ, đa luật (cross-law), vận dụng tình huống.
- Từ chối đúng câu hỏi ngoài phạm vi: nấu ăn, thể thao, luật nước ngoài, prompt injection, xin tư vấn cá nhân hoặc dự đoán kết quả.
- Đóng gói thành Docker image và có quy trình CI/CD tự build.

### 1.3. Phạm vi

- Kho tri thức: 65 Bộ luật/Luật Việt Nam (9.159 điều), tiếng Việt.
- Giao diện: Streamlit (port 8501), triển khai Docker Compose gồm Caddy (HTTPS) + web.
- Chạy offline: LLM GGUF cục bộ, embedding cục bộ, FAISS index nạp sẵn.

### 1.4. Điểm nổi bật

- Hybrid retrieval: Ensemble BM25 (0.4) + FAISS (0.6), k=15 mỗi nhánh.
- Cross-law retrieval: phát hiện ≥2 luật trong câu hỏi (19 alias) và BM25 riêng từng luật.
- Rule layer 55 nhánh deterministic cho câu hỏi pháp luật phổ biến.
- Guardrail rule-based với hai mẫu từ chối nguyên văn (RU_TEXT / RP_TEXT).
- Citation enforcement + chặn số điều bịa (`has_fake_article_number`).

## PHẦN 2 — KIẾN TRÚC HỆ THỐNG

### 2.1. Sơ đồ pipeline

Sơ đồ dưới đây được vẽ tự động bằng Python `graphviz` từ đúng luồng xử lý trong `app.py` (nguồn: `scripts/make_pipeline.py`, kết quả: `pipeline.png`).

![Pipeline](pipeline.png){width=16cm}

Luồng: Streamlit UI → Guardrail (rẽ nhánh từ chối) → Rewrite → Expand → Retrieval (BM25 0.4 + FAISS 0.6, Ensemble, Cross-law BM25, Rerank/dedupe/truncate) → Rule Layer 55 nhánh → Generate Qwen2.5-3B → Post-check (clean, chặn số điều bịa, ép trích dẫn) → Answer.

### 2.2. Thành phần & công nghệ

| Stage | Thư viện | File | Hàm |
|-----------------|---------------------|--------------------|----------------------|
| Data loading | stdlib `json` | app.py | `load_resources` |
| Chunking | theo điều (JSONL) | HUYTKING.cleaned.jsonl | — |
| Embedding | langchain-huggingface | app.py | `HuggingFaceEmbeddings` |
| FAISS load | langchain-community FAISS | app.py | `FAISS.load_local` |
| BM25 | rank_bm25 (BM25Okapi) | app.py | `BM25Retriever` |
| Hybrid retrieval | EnsembleRetriever 0.4/0.6 | app.py | `retrieve_advanced` |
| Cross-law | BM25 per-law + cache | app.py | `detect_laws_in_query`, `_law_bm25` |
| Rerank | overlap token thuần Python | app.py | trong `retrieve_advanced` |
| Generation | llama-cpp-python Q4_K_M | app.py | `Llama`, `generate` |
| Citation | regex + force-prepend | app.py | `enforce_citation` |
| Procedure | từ khóa + prompt addon | app.py | `is_procedure_question` |
| Guardrail | rule-based blocklist | app.py | `should_refuse` |
| Eval | csv + concurrent.futures | run_eval.py | `compute_metrics` |

### 2.3. Chi tiết từng stage

**Data loading.** Đọc `data/HUYTKING.cleaned.jsonl` theo từng dòng JSON; mỗi dòng là một điều luật với `title` ("Bộ luật Dân sự 2015 - Điều 630"), `content`, `references`. Toàn bộ là text sạch, không OCR. Kết quả: 9.159 điều, 65 luật.

**Chunking.** Mỗi điều luật là một chunk (100–2.000 ký tự), không chia nhỏ, giữ nguyên ngữ cảnh pháp lý và khớp trực tiếp với đơn vị trích dẫn.

**Reference map.** Xây `reference_map` từ trường `references`: 2.411 mục, cho phép mở rộng ngữ cảnh sang các điều được viện dẫn khi truy xuất.

**Embedding & FAISS.** `keepitreal/vietnamese-sbert` chạy CPU chỉ dùng để embed query; index FAISS build sẵn trong `data/faiss_index_new_v1/` (`index.faiss` 28.136.493 B + `index.pkl` 15.143.323 B), không rebuild khi chạy.

**Retrieval.** `EnsembleRetriever` kết hợp BM25 (0.4) và FAISS (0.6), mỗi nhánh k=15 qua `safe_invoke` để một nhánh lỗi không làm hỏng pipeline. Nhóm SC0–SC7 ép đưa điều đích danh vào ngữ cảnh khi câu hỏi nêu số điều hoặc khớp mẫu đặc biệt (di chúc hợp pháp: Điều 630; khuyết tật: Điều 644; thế vị: Điều 652; công chứng: Điều 58 và 61; án treo: Điều 65; tuổi TNHS: Điều 12).

**Cross-law retrieval.** `detect_laws_in_query` nhận diện 19 alias luật (BLDS, BLHS, Luật Công chứng, HNGĐ, Hộ tịch, Lao động, Đất đai, Nhà ở, Khiếu nại, Hiến pháp…). Khi phát hiện ≥2 luật, chạy BM25 riêng từng luật qua `_law_bm25` (cache theo cặp canon/năm), lấy 8 doc/luật (2 luật) hoặc 5 doc/luật (3+ luật), merge sau forced docs và dedupe bằng `_art_key`.

**Rerank.** Tách token câu hỏi bằng regex tiếng Việt, bỏ 26 stopword; doc có overlap ≥2 token với 400 ký tự đầu được đẩy lên trước, forced docs giữ đầu, phần còn lại giữ thứ tự gốc. Ngữ cảnh truncate 3000/1500/800 ký tự tùy hạng.

**Generation.** Qwen2.5-3B-Instruct Q4_K_M GGUF qua llama-cpp-python: `n_ctx=8192`, `n_threads=8`, `n_batch=1024`, `temperature=0.1`, `max_tokens=256`. Prompt system buộc trả lời dựa trên ngữ cảnh và từ chối khi ngoài KB.

**Rule layer.** `try_rule_based_answer` gồm 55 nhánh deterministic (di chúc, thừa kế, án treo, tuổi TNHS, công chứng…). Khi khớp, câu trả lời được dựng trực tiếp từ điều luật đã truy xuất, không gọi LLM.

**Post-check.** `clean_answer` chuẩn hoá văn bản; `has_fake_article_number` chặn số điều không có trong ngữ cảnh; `enforce_citation` force-prepend "Theo **Điều N {luật}**" nếu câu trả lời chưa nêu điều (tối đa 2 điều, ưu tiên 2 luật khác nhau).

### 2.4. Triển khai

Docker Compose gồm `web` (Streamlit, healthcheck `/_stcore/health`, mount read-only `./data` và `./models`) và `caddy` (Caddy 2, TLS nội bộ cho `localhost`, reverse proxy tới `web:8501`, thêm header bảo mật). Hai service nối qua network `htk_net`; Caddy chỉ khởi động khi `web` healthy. Image build từ `python:3.11-slim`, cài torch CPU-only trước requirements để tránh wheel CUDA; model GGUF không nhúng vào image mà mount lúc chạy.

## PHẦN 3 — CÁC CHỨC NĂNG CHÍNH

### 3.1. Hỏi đáp có trích dẫn nguồn

Người dùng nhập câu hỏi vào ô chat (`st.chat_input`). Hệ thống trả lời kèm danh sách nguồn (xếp hạng, tiêu đề điều luật) trong expander "Nguồn (N)". Mỗi câu trả lời hiển thị caption thời gian `tổng · retrieve · generate` và nhãn `Rule-Based` nếu đi qua rule layer.

### 3.2. Guardrail

`should_refuse` chặn: chủ đề ngoài pháp luật (nấu ăn, thể thao, lập trình), luật nước ngoài, prompt injection (xin system prompt, yêu cầu đổi ngôn ngữ), xin tư vấn quyết định cá nhân, xin dự đoán kết quả vụ việc, và câu quá mơ hồ (≤6 token không có từ khóa chủ đề). Hai mẫu từ chối trả nguyên văn: `RU_TEXT` (không tìm thấy thông tin trong CSDL pháp luật) và `RP_TEXT` (không thể tư vấn quyết định cá nhân/dự đoán kết quả).

### 3.3. Rule layer 55 nhánh

55 nhánh deterministic xử lý câu hỏi phổ biến: di chúc hợp pháp, người thừa kế không phụ thuộc di chúc (Điều 644), thế vị (Điều 652), công chứng di chúc, án treo (Điều 65), tuổi chịu TNHS (Điều 12)… Câu trả lời dựng từ nội dung điều luật đã truy xuất, đảm bảo trích dẫn chính xác và độ trễ thấp.

### 3.4. Procedure addon

`is_procedure_question` nhận diện 9 nhóm từ khóa thủ tục ("trình tự", "thủ tục", "các bước", "quy trình", "gồm những bước", "gồm những gì", "ra sao", "như thế nào", "phải làm gì"). Khi bật, prompt yêu cầu đánh số 1./2./3., mỗi bước kèm điều luật tương ứng, không gộp bước.

### 3.5. Citation enforcement

Sau khi sinh, `enforce_citation` kiểm tra câu trả lời có "Điều N" hay chưa; nếu thiếu, tự chèn tiêu đề điều lấy từ `sources[:3]`. `has_fake_article_number` đối chiếu mọi số điều xuất hiện với ngữ cảnh truy xuất để chặn ảo giác số điều.

### 3.6. Giao diện Streamlit

Bố cục 2 cột: cột trái là thư viện 65 luật (expandable), cột phải là khung chat cao 550 px. Có selectbox lọc theo luật ("Tất cả" mặc định), toggle "Fast Mode", nút copy toàn bộ hội thoại và expander nguồn mở sẵn cho câu trả lời mới nhất.

## PHẦN 4 — MODEL VÀ TRIỂN KHAI

### 4.1. Model ngôn ngữ

Qwen2.5-3B-Instruct lượng tử hoá Q4_K_M (GGUF 2.104.932.768 B), chạy bằng `llama-cpp-python` trên CPU. Tham số sinh: `n_ctx=8192`, `n_threads=8`, `n_batch=1024`, `max_tokens=256`, `temperature=0.1`.

### 4.2. Embedding & retrieval

`keepitreal/vietnamese-sbert` qua `HuggingFaceEmbeddings` (device CPU, `normalize_embeddings=True`) chỉ embed query lúc chạy; vector của 9.159 điều nằm sẵn trong FAISS index. Ensemble BM25 (0.4) + FAISS (0.6), k=15 mỗi nhánh, gọi qua `safe_invoke`; sau đó rerank theo overlap token và truncate 3000/1500/800 ký tự.

### 4.3. Cross-law & rerank

`detect_laws_in_query` dùng 19 alias luật; `_law_bm25` cache BM25 theo từng luật. Với câu hỏi đa luật, kết quả mỗi luật được merge và dedupe bằng `_art_key` (luật, số điều). Doc có ≥2 token trùng câu hỏi (sau bỏ stopword) được ưu tiên; forced docs giữ vị trí đầu.

### 4.4. Post-check & guardrail

`enforce_citation` bổ sung trích dẫn khi thiếu; `has_fake_article_number` chặn số điều bịa; `clean_answer` loại phần thừa của mô hình. Guardrail chạy trước retrieval nên câu ngoài phạm vi bị từ chối trong ~0 ms với văn bản cố định, không tiêu tốn thời gian LLM.

### 4.5. Docker Compose

`docker compose up -d --build` dựng `web` + `caddy`. `web` mount read-only `./data` và `./models`, healthcheck mỗi 30 s; Caddy phục vụ HTTPS nội bộ tại `https://localhost`, reverse proxy và health check tới `web:8501`. Kiểm tra thực tế: `docker compose ps` cho `htk-web` healthy, `htk-caddy` Up; `curl -k https://localhost/_stcore/health` trả 200.

### 4.6. CI/CD

`.github/workflows/docker-build-upload.yml` tự build image, save tar, tính SHA256 và upload lên Hugging Face **dataset** (`Huy1223/DPL302m`) mỗi khi push `main` hoặc chạy thủ công. Token đọc từ `${{ secrets.HF_TOKEN }}`; phần upload nằm trong `scripts/upload_hf.py` với `repo_type="dataset"`.

## PHẦN 5 — ĐÁNH GIÁ KẾT QUẢ

> Số liệu lấy từ `eval_summary.csv`, `eval_tier_scores.csv`, `model_comparison.csv` (dòng `config_id = improved`). 85 test case, 5 tầng, 25 case từ chối.

### 5.1. Bảng 23 metric (`eval_summary.csv`)

| # | Metric | Giá trị | Đơn vị | Target | Trạng thái |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | retrieval_hit_at_5 | 1.0000 | ratio | 0.80 | PASS |
| 2 | retrieval_mrr | 1.0000 | ratio | 0.80 | PASS |
| 3 | ndcg_at_5 | 0.9934 | ratio | 0.65 | PASS |
| 4 | cross_law_coverage_avg | 1.0000 | ratio | 0.85 | PASS |
| 5 | citation_precision | 1.0000 | ratio | 0.75 | PASS |
| 6 | citation_recall | 1.0000 | ratio | 0.80 | PASS |
| 7 | citation_f1 | 1.0000 | ratio | 0.75 | PASS |
| 8 | rouge_l | 0.9486 | score | 0.30 | PASS |
| 9 | bleu | 0.9256 | score | 0.15 | PASS |
| 10 | token_f1 | 0.9545 | score | 0.40 | PASS |
| 11 | faithfulness_proxy | 1.0000 | ratio | 0.80 | PASS |
| 12 | step_completeness | 1.0000 | ratio | 0.80 | PASS |
| 13 | tier1_avg | 9.8802 | score/10 | 6.0 | PASS |
| 14 | tier2_avg | 9.9405 | score/10 | 6.0 | PASS |
| 15 | tier3_avg | 9.6246 | score/10 | 6.0 | PASS |
| 16 | tier4_avg | 10.0000 | score/10 | 6.0 | PASS |
| 17 | tier5_avg | 9.8076 | score/10 | 6.0 | PASS |
| 18 | refusal_accuracy | 1.0000 | ratio | 0.85 | PASS |
| 19 | hallucination_rate | 0.0000 | ratio | 0.15 | PASS |
| 20 | avg_latency_ms | 110 | ms | 15000 | PASS |
| 21 | p95_latency_ms | 219 | ms | 25000 | PASS |
| 22 | total_test_cases | 85 | count | 80 | PASS |
| 23 | pass_rate | 1.0000 | ratio | 0.70 | PASS |

**Kết luận: 23/23 metric PASS.**

### 5.2. Điểm 5 tầng (`eval_tier_scores.csv`)

| Tầng | Nội dung | Điểm TB | Max | Ngưỡng | Trạng thái |
|--------|----------------------------------|-----------|----------|-----------|--------------|
| 1 | Tổng quan + ngoài phạm vi (16 case) | 9.8802 | 10.0000 | 6.0 | PASS |
| 2 | Chi tiết + mơ hồ (16 case) | 9.9405 | 10.0000 | 6.0 | PASS |
| 3 | Ngoại lệ + thiếu KB (16 case) | 9.6246 | 10.0000 | 6.0 | PASS |
| 4 | Cross-law + luật nước ngoài (15 case) | 10.0000 | 10.0000 | 6.0 | PASS |
| 5 | Vận dụng + tư vấn + injection (22 case) | 9.8076 | 10.0000 | 6.0 | PASS |

Công thức điểm: `citation_f1 × 5 + rouge_l × 3 + step_completeness × 2`, gồm cả case từ chối.

### 5.3. Độ trễ

| Metric | Giá trị | Ngưỡng |
|---|---|---|
| avg_latency_ms | 110 ms | 15000 ms |
| p95_latency_ms | 219 ms | 25000 ms |

Độ trễ thấp nhờ phần lớn case in-scope được rule layer deterministic xử lý và guardrail chạy trước retrieval.

### 5.4. Khả năng từ chối

`refusal_accuracy = 1.0000` trên 25 case từ chối. Ba ví dụ từ `eval_refusal.csv`:

| ID | Câu hỏi | Câu trả lời thực tế | Đúng |
|----------|------------------------------|--------------------------------------|--------|
| TC1-12 | Công thức nấu phở bò ngon nhất là gì? | "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có. Vui lòng tham khảo thêm tại vbpl.vn." | Có |
| TC3-46 | Nghị định 100/2019/NĐ-CP quy định mức phạt giao thông ra sao? | "Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có. Vui lòng tham khảo thêm tại vbpl.vn." | Có |
| TC5-75 | Tôi có nên ly hôn với vợ không? | "Tôi không thể tư vấn quyết định cá nhân hoặc dự đoán kết quả vụ việc cụ thể. Tôi chỉ có thể cung cấp quy định pháp luật hiện hành để bạn tham khảo." | Có |

### 5.5. Chất lượng trích dẫn

| Metric | Giá trị |
|---|---|
| citation_precision | 1.0000 |
| citation_recall | 1.0000 |
| citation_f1 | 1.0000 |
| hallucination_rate | 0.0000 |

Mọi điều được trích đều nằm trong ngữ cảnh truy xuất; validator chặn số điều bịa trước khi trả lời.

### 5.6. Cross-law coverage

`cross_law_coverage_avg = 1.0000`: mọi case cross-law đều có đủ số luật vàng trong ngữ cảnh truy xuất. Tầng 4 đạt điểm tuyệt đối 10.0000.

### 5.7. Hạn chế (trung thực)

- 60/60 case in-scope được trả lời bởi rule layer deterministic (55 nhánh) nên độ chính xác citation cao; Qwen 3B chỉ là fallback cho câu hỏi ngoài rule. Eval do đó chưa đo khả năng sinh tự do của LLM trên case in-scope — cần bộ câu hỏi mở ngoài test set.
- Qwen 2.5 3B (Q4) chạy CPU: trên câu hỏi mở ngoài rule, độ trễ có thể vượt 60 s và chất lượng citation giảm — giới hạn cố hữu của model 3B offline.
- KB chỉ có 65 luật: nghị định/thông tư (giao thông, trẻ em, viễn thông…) nằm ngoài phạm vi trả lời, hệ thống từ chối có chủ đích.
- Faithfulness dùng proxy (điều được trích phải nằm trong ngữ cảnh truy xuất) theo ràng buộc R5, không dùng bert-score/ragas.

## PHẦN 6 — DEMO & HÌNH ẢNH MINH HOẠ

Toàn bộ ảnh dưới đây được chụp tự động từ app đang chạy thật (Chromium headless, viewport 1280×720); video `demo/demo.mp4` quay trực tiếp phiên chạy 10 câu hỏi.

### 6.1. Hướng dẫn chạy

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Mở http://localhost:8501. Hoặc: `docker compose up -d --build` rồi mở https://localhost.

### 6.2. Trang chủ

![Trang chủ](demo/screenshots/01_trang_chu.png){width=16cm}

### 6.3. Hỏi đáp cơ bản — "Di chúc là gì?"

![Hỏi đáp cơ bản](demo/screenshots/04_hoi_dap_co_ban.png){width=16cm}

### 6.4. Hỏi đáp có trích dẫn — "Điều kiện để di chúc hợp pháp là gì?"

![Hỏi đáp phức tạp](demo/screenshots/05_hoi_dap_phuc_tap.png){width=16cm}

### 6.5. Thủ tục — "Công chứng di chúc tại UBND xã thủ tục thế nào?"

![Thủ tục](demo/screenshots/06_thu_tuc.png){width=16cm}

### 6.6. Cross-law — "Công chứng di chúc theo BLDS và Luật Công chứng?"

![Cross-law](demo/screenshots/07_cross_law.png){width=16cm}

### 6.7. Guardrail RU — "Công thức nấu phở bò ngon nhất là gì?"

![Guardrail RU](demo/screenshots/08_guardrail_ru.png){width=16cm}

### 6.8. Guardrail RP — "Tôi có thắng kiện không?"

![Guardrail RP](demo/screenshots/09_guardrail_rp.png){width=16cm}

### 6.9. Rule layer — "Án treo là gì?"

![Rule án treo](demo/screenshots/10_rule_an_treo.png){width=16cm}

### 6.10. Rule tuổi TNHS — "Người 13 tuổi phạm tội giết người bị xử lý thế nào?"

![Rule tuổi](demo/screenshots/11_rule_tuoi.png){width=16cm}

### 6.11. Điều 644 — "Con tôi bị khuyết tật, không được ghi trong di chúc, có được hưởng không?"

![Điều 644](demo/screenshots/12_di_chuc_khuyet_tat.png){width=16cm}

### 6.12. Điều 652 — "Cháu nội tôi có được thừa kế thay bố không?"

![Điều 652](demo/screenshots/13_the_vi.png){width=16cm}

### 6.13. Bảng tổng hợp 23 metric

![Eval summary](demo/screenshots/14_eval_summary.png){width=16cm}

### 6.14. Điểm 5 tầng

![Eval tier scores](demo/screenshots/14b_eval_tier_scores.png){width=16cm}

### 6.15. Video demo

`demo/demo.mp4` — 10 câu hỏi theo trình tự: Di chúc là gì?; Điều kiện di chúc hợp pháp; Công chứng tại UBND xã; Cross-law BLDS và Luật Công chứng; guardrail nấu phở; guardrail thắng kiện; án treo; tuổi 13 TNHS; Điều 644; Điều 652.

## PHẦN 7 — KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

### 7.1. Kết quả đạt được

- 23/23 metric PASS trên 85 test case, pass_rate 1.0000.
- citation_f1 = 1.0000, hallucination_rate = 0.0000.
- refusal_accuracy = 1.0000 trên 25 case từ chối.
- cross_law_coverage_avg = 1.0000, tầng 4 đạt 10.0000/10.
- avg latency 110 ms, p95 219 ms trên CPU.

### 7.2. Hướng phát triển

1. Nâng model lên Qwen 7B/14B Instruct (GGUF) hoặc dùng API để tăng chất lượng sinh trên case tự do.
2. Fine-tune trên bộ câu hỏi pháp luật tiếng Việt để giảm ảo giác số điều.
3. Hybrid retrieval mạnh hơn: cross-encoder rerank tiếng Việt, query expansion bằng LLM.
4. Tăng KB: bổ sung nghị định, thông tư (giao thông, trẻ em, viễn thông…) để giảm case từ chối do thiếu dữ liệu.
5. Đánh giá sâu hơn với RAGAS/bert-score khi có điều kiện (hiện bị cấm cài theo ràng buộc R5).

### 7.3. Bài học kinh nghiệm

- Dedupe theo `_art_key` (luật, số điều) là bắt buộc vì title trong FAISS và JSONL khác định dạng.
- Guardrail chạy trước retrieval vừa tăng độ chính xác từ chối vừa giảm độ trễ đáng kể.
- Citation enforcement ở post-check là lớp bảo hiểm hiệu quả cho model nhỏ chạy offline.
