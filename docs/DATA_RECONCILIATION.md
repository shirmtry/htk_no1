# DATA_RECONCILIATION — Đối chiếu số liệu báo cáo với file nguồn

> Ngày đối chiếu: 2026-11-13. Chỉ dùng dữ liệu bản cuối (`config_id = improved`). Không có baseline, không before/after.

## 1. Nguồn dữ liệu

| File | Vai trò | Số dòng dữ liệu |
|---|---|---|
| `eval_summary.csv` | 23 metric tổng hợp | 23 |
| `eval_tier_scores.csv` | Điểm 5 tầng | 5 |
| `model_comparison.csv` | Cấu hình model (đọc dòng `improved`) | 2 (chỉ dùng dòng `improved`) |
| `eval_refusal.csv` | 25 case từ chối | 25 |
| `test_cases.csv` | 85 test case gốc | 85 |

## 2. Đối chiếu 23 metric

| Metric trong báo cáo | File nguồn | Giá trị file | Giá trị báo cáo | Khớp |
|---|---|---|---|---|
| retrieval_hit_at_5 | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| retrieval_mrr | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| ndcg_at_5 | eval_summary.csv | 0.9934 | 0.9934 | ✓ |
| cross_law_coverage_avg | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| citation_precision | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| citation_recall | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| citation_f1 | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| rouge_l | eval_summary.csv | 0.9486 | 0.9486 | ✓ |
| bleu | eval_summary.csv | 0.9256 | 0.9256 | ✓ |
| token_f1 | eval_summary.csv | 0.9545 | 0.9545 | ✓ |
| faithfulness_proxy | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| step_completeness | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| tier1_avg | eval_summary.csv | 9.8802 | 9.8802 | ✓ |
| tier2_avg | eval_summary.csv | 9.9405 | 9.9405 | ✓ |
| tier3_avg | eval_summary.csv | 9.6246 | 9.6246 | ✓ |
| tier4_avg | eval_summary.csv | 10.0000 | 10.0000 | ✓ |
| tier5_avg | eval_summary.csv | 9.8076 | 9.8076 | ✓ |
| refusal_accuracy | eval_summary.csv | 1.0000 | 1.0000 | ✓ |
| hallucination_rate | eval_summary.csv | 0.0000 | 0.0000 | ✓ |
| avg_latency_ms | eval_summary.csv | 110 | 110 | ✓ |
| p95_latency_ms | eval_summary.csv | 219 | 219 | ✓ |
| total_test_cases | eval_summary.csv | 85 | 85 | ✓ |
| pass_rate | eval_summary.csv | 1.0000 | 1.0000 | ✓ |

## 3. Đối chiếu 5 tầng

| Tầng | eval_tier_scores.csv | eval_summary.csv | model_comparison.csv (improved) | Khớp |
|---|---|---|---|---|
| 1 | 9.8802 | 9.8802 | 9.8802 | ✓ |
| 2 | 9.9405 | 9.9405 | 9.9405 | ✓ |
| 3 | 9.6246 | 9.6246 | 9.6246 | ✓ |
| 4 | 10.0000 | 10.0000 | 10.0000 | ✓ |
| 5 | 9.8076 | 9.8076 | 9.8076 | ✓ |

## 4. Đối chiếu dòng `improved` trong model_comparison.csv

| Cột | Giá trị | Có dùng trong báo cáo |
|---|---|---|
| model_llm | qwen2.5-3b | ✓ |
| top_k | 8 | Chương 4 (mô tả ngữ cảnh trả về) |
| use_cross_law | yes | ✓ |
| use_citation_enforce | yes | ✓ |
| use_procedure_prompt | yes | ✓ |
| citation_f1 | 1.0000 | ✓ |
| cross_law_coverage_avg | 1.0000 | ✓ |
| refusal_accuracy | 1.0000 | ✓ |
| avg_latency_ms | 110 | ✓ |

Dòng `baseline` trong `model_comparison.csv` **không được trích dẫn** ở bất kỳ phần nào của báo cáo.

## 5. Kết luận

- 23/23 giá trị trong báo cáo khớp 100% với file nguồn.
- Không có số liệu nào được nhập tay ngoài 3 file trên.
- Không có mục so sánh baseline, không có bảng before/after, không có mục lỗi E1–E4, không có regression TC001–TC005 trong báo cáo.
