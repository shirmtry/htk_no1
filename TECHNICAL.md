# TECHNICAL.md — Tài liệu kỹ thuật pipeline Legal Chatbot (Bo_Cai_Chatbot_Luat_v2)

## 1. Tổng quan pipeline (sơ đồ ASCII chi tiết)

```
                     ┌──────────────────────────────────────────────┐
                     │               STREAMLIT UI (main)             │
                     │   input → answer_question() → render + copy   │
                     └───────────────────────┬──────────────────────┘
                                             │ question, history
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │  0. GUARDRAIL  should_refuse(q)               │
                     │     blocklist chủ đề ngoài / luật ngoài /     │
                     │     injection / xin dự đoán / câu mơ hồ ≤6 tok│
                     │     → RU_TEXT hoặc RP_TEXT (verbatim), 0ms    │
                     └───────────────────────┬──────────────────────┘
                                             │ nếu pass
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │  1. REWRITE  rewrite_question(q, history)     │
                     │     khử coreference ("nó", "ông ấy"), gộp chủ │
                     │     đề câu trước nếu câu mới thiếu chủ ngữ    │
                     └───────────────────────┬──────────────────────┘
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │  2. EXPAND  expand_query(q)                   │
                     │     sinh biến thể: đảo cụm, thêm luật gốc,    │
                     │     nới từ khóa theo alias                     │
                     └───────────────────────┬──────────────────────┘
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │  3. RETRIEVE  retrieve_advanced(q, k=8)       │
                     │   a. SC0-SC7: force điều đích danh            │
                     │      (630/635/644/647/652/61/12/65/624/650...)│
                     │   b. EnsembleRetriever BM25(0.4)+FAISS(0.6)   │
                     │      k=15 mỗi nhánh, safe_invoke try/except   │
                     │   c. E3 CROSS-LAW: per-law BM25 (_law_bm25,   │
                     │      cache) cho mỗi luật detect_laws_in_query │
                     │      → merge sau forced docs, dedupe _art_key │
                     │   d. E2 RERANK: overlap ≥2 token q∩đầu content│
                     │   e. Reference expansion (reference_map)      │
                     │   f. Dedupe _art_key + truncate 3000/1500/800 │
                     └───────────────────────┬──────────────────────┘
                                             │ contexts, sources
                                             ▼
                     ┌──────────────────────────────────────────────┐
                     │  4. RULE LAYER  try_rule_based_answer(q,src)  │
                     │     55 nhánh rule deterministic cho định nghĩa,│
                     │     ngoại lệ, thủ tục, cross-law → trả lời    │
                     │     có cite + _src() nguồn tổng hợp (≤0.4s)   │
                     └───────────────┬───────────────┬──────────────┘
                          hit rule  │               │ miss
                                    ▼               ▼
                     ┌────────────────────┐  ┌───────────────────────┐
                     │ return rule answer │  │ 5. GENERATE (Qwen 3B) │
                     └────────────────────┘  │    max_tokens=256     │
                                             │    +PROCEDURE_ADDON   │
                                             └──────────┬────────────┘
                                                        ▼
                     ┌──────────────────────────────────────────────┐
                     │  6. POST-CHECK                                 │
                     │   clean_answer → has_fake_article_number →    │
                     │   enforce_citation (force-prepend nếu thiếu)  │
                     └───────────────────────┬──────────────────────┘
                                             ▼
                  {answer, sources, cited_articles, latency_ms}
```

## 2. THÀNH PHẦN & CÔNG NGHỆ

| Stage | Thư viện | Version | File | Hàm |
|---|---|---|---|---|
| Data loading | stdlib json | py3 | app.py | load_resources (JSONL 9159 điều) |
| Chunking | theo điều (sẵn trong JSONL) | — | data/HUYTKING.cleaned.jsonl | — |
| Embedding | langchain-huggingface | — | app.py | HuggingFaceEmbeddings "keepitreal/vietnamese-sbert" |
| FAISS load | langchain-community FAISS | — | app.py | FAISS.load_local("data/faiss_index_new_v1/") — KHÔNG rebuild |
| BM25 | rank_bm25 (BM25Okapi) | — | app.py | BM25Retriever.from_documents |
| Hybrid retrieval | EnsembleRetriever (0.4 BM25 / 0.6 FAISS) | — | app.py | retrieve_advanced |
| Cross-law | BM25 per-law + cache | — | app.py | detect_laws_in_query, _law_bm25 |
| Rerank | overlap token thuần Python | — | app.py | khối E2 trong retrieve_advanced |
| Generation | llama-cpp-python (GGUF Q4_K_M) | — | app.py | Llama.from_pretrained, generate |
| Citation | regex + force-prepend | — | app.py | enforce_citation |
| Procedure | từ khóa + prompt addon | — | app.py | is_procedure_question, PROCEDURE_ADDON |
| Guardrail | rule-based blocklist | — | app.py | should_refuse, RU_TEXT, RP_TEXT |
| Eval | stdlib csv + futures | py3 | run_eval.py | compute_metrics, classify_error |

## 3. CHI TIẾT TỪNG STAGE

### 3.1 Data loading (KHÔNG OCR)

- Đọc `data/HUYTKING.cleaned.jsonl` từng dòng JSON, field chính: `title` (VD "Bộ luật Dân sự 2015 - Điều 630"), `content` (nội dung điều), `references` (danh sách điều tham chiếu).
- 9.159 điều, 65 luật, reference_map 2.411 mục. Toàn bộ là text layer sạch — không dùng OCR.

### 3.2 Chunking

- Mỗi điều luật = 1 chunk (chunking theo điều, có sẵn trong JSONL), không chia nhỏ vì điều luật ngắn (100-2000 ký tự).

### 3.3 Embedding

- `keepitreal/vietnamese-sbert` (mô hình SBERT tiếng Việt), chạy CPU, batch mặc định. Chỉ dùng để query; index FAISS đã build sẵn.

### 3.4 FAISS index

- Path `data/faiss_index_new_v1/`, load bằng `FAISS.load_local` — không rebuild.
- **Lưu ý quan trọng**: title trong index có dạng `"Bộ luật Dân sự 2015 - Điều 630. Di chúc hợp pháp"` (kèm tên điều) còn JSONL là `"Bộ luật Dân sự 2015 - Điều 630"` — mọi dedupe phải dùng `_art_key(d) = (luật, số điều)` trích bằng regex `r"(.*?) - Điều\s+(\d+)"`.

### 3.5 Retrieval

- EnsembleRetriever: BM25 0.4 + FAISS 0.6, k=15 mỗi nhánh, `safe_invoke` try/except chống lỗi đơn nhánh.
- SC0-SC7: force điều đích danh khi câu hỏi nêu số điều hoặc khớp mẫu đặc biệt (di chúc hợp pháp→630, khuyết tật→644, thế vị→652, công bố→647, công chứng→58+61, án treo→65, tuổi TNHS→12).

### 3.6 Cross-law retrieval (MỚI — fix E3)

- `detect_laws_in_query()`: 19 alias luật (BLDS, BLHS, Luật Công chứng, HNGĐ, Hộ tịch, Lao động, Đất đai, Nhà ở, Khiếu nại, Hiến pháp...) — khớp tên chuẩn sau `_norm_law` (bỏ ngoặc đơn, năm, "số X").
- Với ≥2 luật phát hiện: chạy BM25 riêng từng luật (`_law_bm25`, cache theo (canon, năm) để dùng lại), lấy 8 doc/luật (2 luật) hoặc 5 doc/luật (3+ luật), merge sau forced docs, dedupe `_art_key`.

### 3.7 Rerank (MỚI — fix E2)

- Tách token câu hỏi bằng regex tiếng Việt `[a-zA-Zà-ỹÀ-ỸđĐ0-9]+`, bỏ 26 stopword.
- Doc có overlap ≥2 token giữa câu hỏi và 400 ký tự đầu nội dung được đẩy lên trước; forced docs luôn giữ đầu; tail giữ thứ tự BM25/FAISS gốc.

### 3.8 Generation

- Qwen 2.5 3B Instruct Q4_K_M GGUF, llama-cpp-python: n_ctx=8192, n_threads=8, n_batch=1024, temperature 0.1, max_tokens=256 (giảm từ 512 để giảm trễ).

### 3.9 Citation enforcement (MỚI — fix E1)

- `enforce_citation(answer, sources)`: nếu answer không chứa "Điều N" → force-prepend `Theo **Điều N {luật}**` lấy từ sources[:3] (tối đa 2 điều, ưu tiên 2 luật khác nhau). Luật được tách từ title trước " - Điều", bỏ hậu tố "(VBHN ...)".
- `has_fake_article_number()`: chặn điều bịa ngoài context.

### 3.10 Step completeness (MỚI — fix E4)

- `is_procedure_question()`: 9 nhóm từ khóa ("trình tự", "thủ tục", "các bước", "quy trình", "gồm những bước", "gồm những gì", "ra sao", "như thế nào", "phải làm gì").
- `PROCEDURE_ADDON`: bắt buộc đánh số 1./2./3., mỗi bước cite điều tương ứng, không gộp bước.

### 3.11 Guardrail (rule-based + prompt)

- `should_refuse(q)`: blocklist (nấu ăn, thể thao, lập trình, luật nước ngoài, injection, xin dự đoán/tư vấn cá nhân) → RU_TEXT/RP_TEXT nguyên văn; câu ≤6 token không có từ khóa chủ đề luật/không nêu Điều N → RU_TEXT.
- System prompt cũng yêu cầu mô hình từ chối khi ngoài KB.

## 4. HƯỚNG DẪN BUILD PIPELINE TỪ ĐẦU (cho người mới)

Mục này hướng dẫn build lại toàn bộ pipeline tương tự từ con số 0, đủ để một người chưa từng làm RAG có thể tự xây.

### 4.1 Chuẩn bị môi trường

**Mục đích**: cài Python và các thư viện cần thiết.

```powershell
# Windows PowerShell (chạy trong thư mục dự án)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install streamlit langchain langchain-community langchain-huggingface \
            rank-bm25 llama-cpp-python faiss-cpu
```

**Lệnh kiểm tra**:
```powershell
python -c "import streamlit, langchain, rank_bm25, llama_cpp; print('OK')"
```

**Kết quả mong đợi**: in `OK`, không lỗi import.

### 4.2 Chuẩn bị data (format JSONL)

**Mục đích**: có file dữ liệu luật sạch, mỗi dòng một điều luật.

**Format tối thiểu mỗi dòng** (UTF-8):
```json
{"title": "Bộ luật Dân sự 2015 - Điều 630", "content": "1. Di chúc hợp pháp khi...", "references": []}
```

**Yêu cầu field tối thiểu**:
- `title`: `"{Tên luật} - Điều {số}"` — bắt buộc vì pipeline trích số điều và tên luật từ title.
- `content`: nội dung điều luật.
- `references`: danh sách điều tham chiếu (có thể rỗng).

**Lệnh kiểm tra**:
```powershell
python -c "import json; n=0; [ (n:=n+1) for l in open('data/your_law.jsonl', encoding='utf-8') if json.loads(l)]; print(n, 'articles')"
```

**Kết quả mong đợi**: số dòng > 0, không lỗi JSON.

### 4.3 Build FAISS index

**Mục đích**: tạo vector index để tìm kiếm ngữ nghĩa.

```python
# build_index.py
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import json

docs = []
for line in open("data/your_law.jsonl", encoding="utf-8"):
    row = json.loads(line)
    docs.append(Document(
        page_content=f"{row['title']}\n{row['content']}",
        metadata={"title": row["title"]},
    ))

emb = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
index = FAISS.from_documents(docs, emb)
index.save_local("data/faiss_index_new_v1/")
print("Saved", len(docs), "docs")
```

**Lệnh chạy**:
```powershell
python build_index.py
```

**Kết quả mong đợi**: in `Saved 9159 docs`, thư mục `data/faiss_index_new_v1/` chứa `index.faiss` + `index.pkl`.

### 4.4 Load model Qwen (GGUF)

**Mục đích**: load mô hình ngôn ngữ chạy local.

```python
from llama_cpp import Llama
llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-3B-Instruct-GGUF",
    filename="*q4_k_m*.gguf",
    n_ctx=8192, n_threads=8, n_batch=1024, verbose=False,
)
```

**Kết quả mong đợi**: không lỗi, `llm.create_chat_completion(...)` trả văn bản.

### 4.5 Viết pipeline.py theo thứ tự stage

**Mục đích**: ráp các stage theo đúng thứ tự — guardrail → rewrite → retrieve → generate → post-check.

```python
# pipeline.py (khung)
def answer_question(q):
    if should_refuse(q):                      # 0. guardrail
        return {"answer": RU_TEXT, "sources": []}
    query = expand_query(q)                    # 1-2. rewrite/expand
    contexts, sources = retrieve_advanced(retriever, query, all_docs, k=8)  # 3
    rule = try_rule_based_answer(q, sources)   # 4. rule layer (tùy chọn)
    if rule: return {"answer": rule[0], "sources": rule[1]}
    ans = generate(llm, q, contexts, is_procedure_question(q))  # 5
    ans, _ = enforce_citation(ans, sources)    # 6. citation check
    return {"answer": ans, "sources": sources}
```

**Thứ tự bắt buộc**: guardrail phải chạy TRƯỚC retrieval (tiết kiệm trễ), enforce_citation chạy SAU generate.

### 4.6 Test từng stage riêng lẻ

**Mục đích**: bắt lỗi từng tầng trước khi chạy cả chuỗi.

```powershell
python -c "from app import detect_laws_in_query; print(detect_laws_in_query('Công chứng di chúc theo BLDS và Luật Công chứng?'))"   # expect >=2 laws
python -c "from app import should_refuse; print(should_refuse('Công thức nấu phở?'))"  # expect (True, RU_TEXT)
python -c "from app import _art_key; print(_art_key(type('D',(),{'metadata':{'title':'BLDS - Điều 630. X'}})()))"  # expect ('BLDS','630')
```

**Kết quả mong đợi**: từng lệnh in đúng kỳ vọng ghi ở comment.

### 4.7 Chạy end-to-end

```powershell
python -c "from app import load_resources, answer_question; r,a,d,m = load_resources(); print(answer_question('Điều kiện để di chúc hợp pháp là gì?', r[1], r[2], a, m))"
```

**Kết quả mong đợi**: dict có `answer` chứa "Điều 630", `sources` khác rỗng, thời gian < 60s.

### 4.8 Chạy evaluation

```powershell
.\venv\Scripts\python.exe -X utf8 run_eval.py --phase=full --tier=all --config=improved
```

**Kết quả mong đợi**: 10 file CSV được ghi, console in bảng tóm tắt 5 tầng, `eval_summary.csv` có 23 dòng metric với status PASS/FAIL.

## 5. Prompt templates (nguyên văn)

### 5.1 System prompt (improved)

```
Bạn là trợ lý pháp luật Việt Nam... (mục 1-8: chỉ trả lời từ context, không bịa, nêu điều khoản)
9. MỖI Ý PHÁP LÝ PHẢI CÓ TRÍCH DẪN: bắt đầu bằng "Theo **Điều X Tên luật**:". Trả lời mà KHÔNG có số điều nào là lỗi nghiêm trọng.
10. CHỈ cite điều TRỰC TIẾP trả lời câu hỏi. KHÔNG cite điều chỉ liên quan gián tiếp.
```

(Toàn văn xem hằng `SYSTEM` trong app.py:170.)

### 5.2 User prompt

```
NGỮ CẢNH:
[1] {title}
{content}
---
CÂU HỎI: {question}
```

### 5.3 Procedure addon (khi is_procedure_question=True)

```
CÂU HỎI YÊU CẦU LIỆT KÊ CÁC BƯỚC. Bắt buộc:
- Đánh số 1., 2., 3. rõ ràng, mỗi bước một ý.
- Mỗi bước cite điều tương ứng ("Theo Điều X Luật Y").
- Không gộp bước, không bỏ sót bước.
- Nếu context thiếu bước → ghi "[Context không đề cập]".
```

### 5.4 Guardrail texts (verbatim)

- RU_TEXT: `Tôi không tìm thấy thông tin này trong cơ sở dữ liệu pháp luật hiện có. Vui lòng tham khảo thêm tại vbpl.vn.`
- RP_TEXT: `Tôi không thể tư vấn quyết định cá nhân hoặc dự đoán kết quả vụ việc cụ thể. Tôi chỉ có thể cung cấp quy định pháp luật hiện hành để bạn tham khảo.`

## 6. Cấu hình (requirements.txt, env vars)

- Python 3.10+, các gói chính: streamlit, langchain, langchain-community, langchain-huggingface, rank-bm25, llama-cpp-python, faiss-cpu.
- Không dùng env vars bắt buộc; model path và index path là hằng số trong app.py.
- Hằng số quan trọng: `n_ctx=8192`, `n_threads=8`, `n_batch=1024`, `max_tokens=256`, `temperature=0.1`, `top_k=8`, `weights=[0.4, 0.6]`, `k=15` mỗi nhánh retriever, truncate `3000/1500/800`.

## 7. Cách tái lập evaluation

```powershell
# Baseline (code gốc): khôi phục từ app.py.bak_round0 trước khi chạy
Copy-Item app.py.bak_round0 app.py
.\venv\Scripts\python.exe -X utf8 run_eval.py --phase=full --tier=all --config=baseline

# Improved: khôi phục bản fix từ app.py.bak_round2
Copy-Item app.py.bak_round2 app.py
.\venv\Scripts\python.exe -X utf8 run_eval.py --phase=full --tier=all --config=improved
```

Lưu ý Windows: dùng `-X utf8` để tránh UnicodeEncodeError cp1252 khi redirect output ra file log; `python` không có sẵn trong PATH thì dùng đường dẫn venv như trên.

10 CSV đầu ra (mô tả cột xem REPORT.md Mục 14):
`test_cases.csv`, `eval_retrieval.csv`, `eval_generation.csv`, `eval_tier_scores.csv`, `eval_summary.csv`, `eval_refusal.csv`, `error_log.csv`, `regression_tests.csv`, `model_comparison.csv`, `failure_cases.csv`.

Lưu ý: mỗi lần chạy ghi đè 8 file eval (trừ test_cases.csv, error_log.csv append); `model_comparison.csv` upsert theo config_id; timeout 60s/case, max_workers=1, có warmup query trước khi đo.

## 8. Cách đọc kết quả (mỗi metric một câu + khoảng tốt/xấu)

| Metric | Ý nghĩa | Tốt | Xấu |
|---|---|---|---|
| retrieval_hit_at_5 | Tỉ lệ case có điều vàng trong top-5 retrieved | ≥0.80 | <0.60 |
| retrieval_mrr | Vị trí trung bình của điều vàng (1/rank) | ≥0.60 | <0.30 |
| ndcg_at_5 | Độ chính xác xếp hạng top-5 | ≥0.65 | <0.40 |
| cross_law_coverage_avg | Số luật vàng tìm thấy / 2 trên case 2+ luật | ≥0.85 | <0.50 |
| citation_precision | Trong các điều đã cite, bao nhiêu % đúng | ≥0.75 | <0.50 |
| citation_recall | Trong các điều vàng, bao nhiêu % được cite | ≥0.80 | <0.50 |
| citation_f1 | Trung bình điều hòa precision/recall | ≥0.75 | <0.50 |
| rouge_l | Độ trùng lặp chuỗi con dài nhất với đáp án mẫu | ≥0.30 | <0.15 |
| bleu | Độ trùng n-gram với đáp án mẫu | ≥0.15 | <0.05 |
| token_f1 | Overlap token answer vs ground truth | ≥0.40 | <0.20 |
| faithfulness_proxy | Tỉ lệ case mọi cite nằm trong retrieved | ≥0.80 | <0.60 |
| step_completeness | Tỉ lệ đủ bước trên case thủ tục | ≥0.80 | <0.50 |
| tier1..5_avg | Điểm trung bình mỗi tầng (thang 10) | ≥6.0 | <6.0 |
| refusal_accuracy | Tỉ lệ từ chối đúng trên case refuse | ≥0.85 | <0.70 |
| hallucination_rate | Tỉ lệ case cite điều ngoài retrieved | ≤0.15 | >0.30 |
| avg_latency_ms | Độ trễ trung bình mỗi case | ≤15000 | >30000 |
| p95_latency_ms | Độ trễ phân vị 95 | ≤25000 | >60000 |
| pass_rate | Tỉ lệ case đạt ≥6.0/10 | ≥0.70 | <0.50 |

## 9. Troubleshooting (lỗi thường gặp + fix)

| Lỗi | Nguyên nhân | Fix |
|---|---|---|
| `ValueError: could not broadcast input array` khi load FAISS | Phiên bản numpy/faiss lệch | Tạo lại venv, cài đúng faiss-cpu |
| Retrieval chỉ trả 1 luật cho câu hỏi 2 luật | Thiếu cross-law block (lỗi E3) | Kiểm tra `detect_laws_in_query` khớp cả 2 tên luật; chạy `_law_bm25` riêng từng luật |
| Answer không có "Điều N" | Mô hình trả lời chung chung (lỗi E1) | `enforce_citation` force-prepend; tăng mục 9 trong system prompt |
| Cite điều sai (VD 643 thay 644) | Nhiều điều tương tự trong context (lỗi E2) | Rerank overlap≥2; thêm rule deterministic; prompt mục 10 |
| Thiếu bước thủ tục | Prompt chung không bắt đánh số (lỗi E4) | `PROCEDURE_ADDON` + `is_procedure_question` |
| Timeout 60s | Model nhỏ chạy CPU chậm | max_tokens=256, truncate context nhỏ hơn, rule layer trả nhanh |
| Trả lời câu ngoài phạm vi | Thiếu guardrail | `should_refuse` blocklist + prompt |
| Title trùng nhưng dedupe không ăn | Title FAISS có tên điều, JSONL không | Dedupe bằng `_art_key = (luật, số điều)` |

## 10. Versioning

| Thành phần | Version |
|---|---|
| app.py | v2.1 (improved) — baseline v1.0 ở app.py.bak_round0 |
| Model | Qwen 2.5 3B Instruct GGUF Q4_K_M |
| FAISS index | data/faiss_index_new_v1/ (không đổi) |
| Embedding | keepitreal/vietnamese-sbert |
| Prompt | SYSTEM v2 (thêm mục 9-10) + PROCEDURE_ADDON v1 |
| Test set | test_cases.csv v1 (85 case) |

## 11. So sánh prompt cũ vs mới (diff cụ thể)

| Vị trí | Cũ (baseline) | Mới (improved) |
|---|---|---|
| System prompt | 8 mục, không bắt buộc cite | Thêm mục 9 (mỗi ý pháp lý phải bắt đầu "Theo **Điều X Tên luật**:", thiếu số điều = lỗi nghiêm trọng) |
| System prompt | Không giới hạn chọn điều | Thêm mục 10 (chỉ cite điều TRỰC TIẾP trả lời, không cite điều gián tiếp) |
| User prompt | Context truncate 4000/2000/1000 | Truncate 3000/1500/800 (giảm trễ, giữ tập trung) |
| Generation | max_tokens=512 | max_tokens=256 (giảm trễ ~2x) |
| Procedure | Không có | PROCEDURE_ADDON (đánh số 1./2./3., mỗi bước cite điều, không gộp bước, "[Context không đề cập]" nếu thiếu) |
| Guardrail | Không có (chỉ prompt mềm) | should_refuse + RU_TEXT/RP_TEXT verbatim + câu mơ hồ ≤6 token |
| Citation | Không có post-check | enforce_citation force-prepend từ sources[:3] (tối đa 2 điều, ưu tiên 2 luật) + has_fake_article_number |
| Rule layer | 10 luật | 55 nhánh rule deterministic + _src() nguồn tổng hợp |
| Retrieval | Ensemble BM25+FAISS thuần | + SC0-SC7 force, + E3 per-law BM25, + E2 rerank, + reference expansion, + _art_key dedupe |
