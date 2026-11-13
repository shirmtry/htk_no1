# Kịch bản demo — 10 câu hỏi

> Quay tự động bằng `scripts/record_demo.py` trên app thật (`http://localhost:8501`), Chromium viewport 1280×720. Câu hỏi nhập qua `st.chat_input` selector `[data-testid="stChatInput"] textarea`, gửi bằng Enter, chờ caption `⏱️` xuất hiện.

| Thứ tự | Câu hỏi | Mục đích | Ảnh |
|---|---|---|---|
| 1 | Di chúc là gì? | Hỏi đáp cơ bản, rule layer | `04_hoi_dap_co_ban.png` |
| 2 | Điều kiện để di chúc hợp pháp là gì? | Trả lời có trích dẫn Điều 630 | `05_hoi_dap_phuc_tap.png` |
| 3 | Công chứng di chúc tại UBND xã thủ tục thế nào? | Procedure addon, đánh số bước | `06_thu_tuc.png` |
| 4 | Công chứng di chúc theo BLDS và Luật Công chứng? | Cross-law retrieval | `07_cross_law.png` |
| 5 | Công thức nấu phở bò ngon nhất là gì? | Guardrail RU_TEXT | `08_guardrail_ru.png` |
| 6 | Tôi có thắng kiện không? | Guardrail RP_TEXT | `09_guardrail_rp.png` |
| 7 | Án treo là gì? | Rule layer Điều 65 BLHS | `10_rule_an_treo.png` |
| 8 | Người 13 tuổi phạm tội giết người bị xử lý thế nào? | Rule tuổi TNHS Điều 12 | `11_rule_tuoi.png` |
| 9 | Con tôi bị khuyết tật, không được ghi trong di chúc, có được hưởng không? | Điều 644 BLDS | `12_di_chuc_khuyet_tat.png` |
| 10 | Cháu nội tôi có được thừa kế thay bố không? | Điều 652 BLDS (thế vị) | `13_the_vi.png` |

Ngoài ra:

- `01_trang_chu.png`: chụp ngay sau khi load xong (0 câu hỏi).
- `14_eval_summary.png`, `14b_eval_tier_scores.png`: render từ CSV bằng PIL.
- `15_docker_ps.png`, `16_docker_build.png`, `17_docker_https.png`: bằng chứng Docker.
- `demo/demo.mp4`: toàn bộ phiên chạy 10 câu hỏi, quay thật từ Playwright.
