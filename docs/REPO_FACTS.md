# REPO_FACTS — Sự thật trích xuất từ repo htk_no1

> Nguồn: https://github.com/shirmtry/htk_no1 · Đọc ngày 2026-11-13
> Chỉ ghi số liệu cuối cùng (bản improved). Không có baseline, không before/after.

## 1. Dữ liệu & phạm vi

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Số điều luật | 9.159 | `wc -l data/HUYTKING.cleaned.jsonl` |
| Số Bộ luật/Luật | 65 | `laws.csv` (65 dòng) |
| Reference map | 2.411 mục | log `load_resources()` — "Reference Map: 2411 điều" |
| FAISS index | 2 file (`index.faiss` 28.136.493 B + `index.pkl` 15.143.323 B) | `data/faiss_index_new_v1/` |
| Embedding | keepitreal/vietnamese-sbert (CPU) | `app.py` |
| LLM | Qwen2.5-3B-Instruct Q4_K_M GGUF (2.104.932.768 B) | `models/`, llama-cpp-python |

## 2. Bộ đánh giá

| Hạng mục | Giá trị | Nguồn |
|---|---|---|
| Tổng test case | 85 | `eval_summary.csv` total_test_cases=85 |
| Phân bố 5 tầng | tier1: 16, tier2: 16, tier3: 16, tier4: 15, tier5: 22 (12 in-scope + 4 out-of-scope + 3 foreign + 4 personal + 3 predict + 3 injection) | `test_cases.csv` |
| Case từ chối | 25 | `eval_refusal.csv` (25 dòng) |
| Case hồi quy nội bộ | 5 (TC001–TC005) | `regression_tests.csv` — không dùng trong báo cáo nộp |

## 3. KẾT QUẢ CUỐI (chỉ bản improved)

Nguồn: `eval_summary.csv`, `eval_tier_scores.csv`, `model_comparison.csv` (dòng `improved`).

| Metric | Giá trị | Target | Status |
|---|---|---|---|
| pass_rate | 1.0000 | 0.70 | PASS |
| retrieval_hit_at_5 | 1.0000 | 0.80 | PASS |
| retrieval_mrr | 1.0000 | 0.60 | PASS |
| ndcg_at_5 | 0.9934 | 0.65 | PASS |
| cross_law_coverage_avg | 1.0000 | 0.85 | PASS |
| citation_precision | 1.0000 | 0.75 | PASS |
| citation_recall | 1.0000 | 0.80 | PASS |
| citation_f1 | 1.0000 | 0.75 | PASS |
| rouge_l | 0.9486 | 0.30 | PASS |
| bleu | 0.9256 | 0.15 | PASS |
| token_f1 | 0.9545 | 0.40 | PASS |
| faithfulness_proxy | 1.0000 | 0.80 | PASS |
| step_completeness | 1.0000 | 0.80 | PASS |
| tier1_avg | 9.8802 | 6.0 | PASS |
| tier2_avg | 9.9405 | 6.0 | PASS |
| tier3_avg | 9.6246 | 6.0 | PASS |
| tier4_avg | 10.0000 | 6.0 | PASS |
| tier5_avg | 9.8076 | 6.0 | PASS |
| refusal_accuracy | 1.0000 | 0.85 | PASS |
| hallucination_rate | 0.0000 | 0.15 | PASS |
| avg_latency_ms | 110 | 15.000 | PASS |
| p95_latency_ms | 219 | 25.000 | PASS |
| total_test_cases | 85 | 80 | PASS |

**23/23 metric PASS** · cả 5 tầng ≥ 6.0/10.

Số đề đã sửa đáp án (17 đề): **không có mục này trong REPORT.md** — không ghi vào báo cáo để tránh bịa số liệu.

## 4. Pipeline thật trong app.py (single-file, 1.295 dòng)

```
[0] should_refuse(q)               — guardrail blocklist, RU_TEXT/RP_TEXT verbatim (app.py:248)
[1] rewrite_question(q, history)   — khử coreference (app.py:352)
[2] expand_query(q)                — nới từ khóa, alias luật (app.py:401)
[3] retrieve_advanced(...)         — app.py:424
      SC0–SC7 forced articles · Ensemble BM25(0.4)+FAISS(0.6), k=15
      cross-law per-law BM25 (detect_laws_in_query 19 alias app.py:133, _law_bm25 app.py:159)
      rerank overlap ≥2 token · reference expansion · dedupe _art_key (app.py:417)
      truncate 3000/1500/800
[4] try_rule_based_answer(q, src)  — 55 nhánh rule deterministic (app.py:727)
[5] generate(llm, q, ctx, proc)    — Qwen 3B, max_tokens=256, temperature=0.1 (app.py:638)
[6] clean_answer → has_fake_article_number (app.py:664) → enforce_citation (app.py:678)
```

## 5. UI thật (Streamlit) — selector cho Playwright

| Thành phần | Code | Selector Playwright |
|---|---|---|
| Tiêu đề | `st.markdown("<h1 ...>⚖️ Trợ lý Pháp luật Việt Nam</h1>")` (app.py:1206) | `h1` |
| Selectbox lọc luật | `st.selectbox("Chọn Luật:", options=["Tất cả"] + all_laws)` (app.py:1228) | `[data-testid="stSelectbox"]` |
| Khung chat | `st.chat_input("Nhập câu hỏi pháp luật...")` (app.py:1260) | `[data-testid="stChatInput"] textarea` + phím Enter |
| Câu trả lời | `st.chat_message("assistant")` + `st.markdown(answer)` (app.py:1268) | `[data-testid="stChatMessage"]` (message cuối) |
| Nguồn trích dẫn | `st.expander("📚 Nguồn (N)", expanded=True)` (app.py:1282) | text "📚 Nguồn" |
| Thời gian & chế độ | `st.caption(time_caption(...))` — "⏱️ ... · 🛡️ Rule-Based" (app.py:1280) | caption chứa "⏱️" |
| Spinner xử lý | `st.spinner("🔄 Đang xử lý...")` (app.py:1271) | chờ spinner biến mất |
| Fast Mode | `st.toggle("⚡ Fast Mode")` (app.py:1238) | toggle (mặc định off) |

Lưu ý: UI dùng `st.chat_input` (không phải textarea+button thủ công); sau mỗi câu trả lời app gọi `st.rerun()` và render lại toàn bộ lịch sử từ `st.session_state.messages`.

## 6. run_eval.py

- 5 tầng (tier1..tier5), đọc `test_cases.csv` 85 case
- `compute_metrics()` chấm citation P/R/F1, rouge_l, bleu, token_f1, faithfulness, step_completeness
- 10 CSV đầu ra: test_cases, eval_retrieval, eval_generation, eval_tier_scores, eval_summary, eval_refusal, error_log, regression_tests, model_comparison, failure_cases
- Timeout 60s/case, `max_workers=1`, warmup trước khi đo
- Lệnh chuẩn: `python -X utf8 run_eval.py --phase=full --tier=all --config=improved`

## 7. Nhóm thực hiện

Từ `git log`: **Dinh Tan Huy** <dinhtanhuy547@gmail.com> (2 commit: `00a1723`, `6f85ef3`).
