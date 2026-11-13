# DEMO_LOG — Nhật ký tự chạy app, chụp ảnh, quay video

> Ngày chạy: 2026-11-13 · Máy: Windows 10 (x64), CPU-only · Agent tự thực hiện 100%, không có thao tác thủ công của nhóm.

## 1. Cách chạy app

| Bước | Lệnh thực tế | Kết quả |
|---|---|---|
| Khởi động | `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 venv/Scripts/python.exe -X utf8 -m streamlit run app.py --server.headless true --server.port 8501` | Uvicorn server started on `:::8501` |
| Health check | `curl http://localhost:8501/_stcore/health` | `200` |
| Load resources | log server | `✅ Loaded 9159 điều, 65 luật` · `✅ Reference Map: 2411 điều` · `✅ Qwen 2.5-3B loaded` · `🛡️ [RULE] Kích hoạt` |

Ghi chú: lần chạy đầu tiên với console mặc định cp1252 bị `UnicodeEncodeError` tại dòng `print("⏳ Loading resources...")`; đã khởi động lại với `PYTHONIOENCODING=utf-8`, server chạy ổn định.

## 2. Chụp ảnh & quay video — Playwright (thật, không mô phỏng)

Script: `scripts/record_demo.py` · Chromium headless · viewport 1280×720 · `ignore_https_errors=True` · `record_video_dir=demo/videos`.

Cách chờ câu trả lời: đếm số `[data-testid="stChatMessage"]` có caption chứa `⏱️` (time_caption) ≥ số câu đã hỏi và không còn spinner "Đang xử lý".

### Ảnh chụp thật từ app đang chạy

| # | File | Câu hỏi | Trạng thái |
|---|---|---|---|
| 1 | `demo/screenshots/01_trang_chu.png` | — (trang chủ) | THẬT |
| 2 | `demo/screenshots/04_hoi_dap_co_ban.png` | Di chúc là gì? | THẬT |
| 3 | `demo/screenshots/05_hoi_dap_phuc_tap.png` | Điều kiện để di chúc hợp pháp là gì? | THẬT |
| 4 | `demo/screenshots/06_thu_tuc.png` | Công chứng di chúc tại UBND xã thủ tục thế nào? | THẬT |
| 5 | `demo/screenshots/07_cross_law.png` | Công chứng di chúc theo BLDS và Luật Công chứng? | THẬT |
| 6 | `demo/screenshots/08_guardrail_ru.png` | Công thức nấu phở bò ngon nhất là gì? | THẬT |
| 7 | `demo/screenshots/09_guardrail_rp.png` | Tôi có thắng kiện không? | THẬT |
| 8 | `demo/screenshots/10_rule_an_treo.png` | Án treo là gì? | THẬT |
| 9 | `demo/screenshots/11_rule_tuoi.png` | Người 13 tuổi phạm tội giết người bị xử lý thế nào? | THẬT |
| 10 | `demo/screenshots/12_di_chuc_khuyet_tat.png` | Con tôi bị khuyết tật, không được ghi trong di chúc, có được hưởng không? | THẬT |
| 11 | `demo/screenshots/13_the_vi.png` | Cháu nội tôi có được thừa kế thay bố không? | THẬT |

Log chạy: `[DEMO] Xác nhận 10/10 câu trả lời có caption ⏱️`.

### Ảnh render bảng số liệu (PIL)

| # | File | Nguồn |
|---|---|---|
| 12 | `demo/screenshots/14_eval_summary.png` | `eval_summary.csv` (23 metric) — render bằng `scripts/record_demo.py --render-text` |
| 13 | `demo/screenshots/14b_eval_tier_scores.png` | `eval_tier_scores.csv` (5 tầng) — render bằng PIL |

## 3. Video

- Raw: `demo/videos/page@07ed41fca13c397acb0e9514a6afea46.webm` (11.773 KB, Playwright ghi trực tiếp phiên chạy).
- Chuyển đổi: `ffmpeg -i <webm> -c:v libx264 -crf 23 -pix_fmt yuv420p -movflags +faststart demo/demo.mp4`.
- Kết quả: `demo/demo.mp4` 2.603 KB, ~2 phút (toàn bộ 10 câu hỏi, gồm cả câu trả lời chậm nhất ~1m25s của fallback LLM).

## 4. Ảnh Docker (bổ sung sau khi build xong)

| # | File | Nguồn |
|---|---|---|
| 14 | `demo/screenshots/15_docker_ps.png` | render `docker compose ps` |
| 15 | `demo/screenshots/16_docker_build.png` | render log `docker build` |
| 16 | `demo/screenshots/17_docker_https.png` | Playwright mở `https://localhost` qua Caddy |

## 5. Tổng kết

- Ảnh thật từ app: **11/11** (01 + 10 câu hỏi). Không có ảnh mô phỏng.
- Ảnh render số liệu: 2 (từ CSV thật).
- Ảnh Docker: 3 (xem mục 4).
- Video: quay thật từ phiên Playwright, không phải slideshow.
