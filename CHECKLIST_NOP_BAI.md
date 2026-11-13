# CHECKLIST NỘP BÀI

[x] 1. Đã đọc đủ README/TECHNICAL/REPORT/structure + 10 CSV eval — tổng hợp tại `docs/REPO_FACTS.md`
[x] 2. Repo GitHub đã dọn rác (docs/GITHUB_CLEANUP_LOG.md) — file rác máy local vẫn giữ nguyên — commit `f433a00`
[x] 3. Báo cáo PDF đủ 8 chương, KHÔNG baseline, KHÔNG before/after, KHÔNG E1-E4, KHÔNG regression
[x] 4. Số liệu khớp 100% với eval_summary.csv + eval_tier_scores.csv + model_comparison.csv (dòng improved) — docs/DATA_RECONCILIATION.md
[x] 5. Docker Compose chạy được: `docker compose ps` → htk-web **healthy**, htk-caddy Up; `curl -k https://localhost/_stcore/health` → 200
[ ] 6. Docker image đã build → save .tar → SHA256 → upload HF Huy1223/DPL302m thành công (docs/HF_UPLOAD_LOG.md) — **CHƯA UPLOAD: môi trường không có biến `HF_TOKEN`**; đã có tar → sha256 `c4e917d2…6485`, script + workflow sẵn sàng
[x] 7. File htk-no1-app.tar.sha256 có trong source/, KHÔNG có .tar trong ZIP
[x] 8. .github/workflows/docker-build-upload.yml đã tạo, dùng `${{ secrets.HF_TOKEN }}`
[x] 9. demo/demo.mp4 > 0 byte (2.602.975 B, 157,6 s), quay từ app thật (docs/DEMO_LOG.md)
[x] 10. demo/screenshots/ có 16 ảnh PNG, toàn bộ chụp thật từ app đang chạy (không fallback)
[x] 11. logs/ có đúng 3 file sạch: build.log, test.log, eval.log (ghi rõ "log tái tạo ngày 2026-11-13")
[x] 12. ZIP KHÔNG chứa: .git, venv, __pycache__, *.pyc, .bak, *.log lỗi, error_log.csv, *.tmp, .env, *.gguf, faiss_index.zip, *.tar
[x] 13. ZIP < 100MB (kiểm tra sau khi nén — xem đầu ra console)
[x] 14. BAO_CAO_HTK_NO1_LEGAL_CHATBOT.pdf render đúng tiếng Việt, có đủ ảnh demo (18 trang A4)
[x] 15. README.md + báo cáo có đủ link GitHub và Hugging Face
[x] 16. Video demo.mp4 mở/phát được, thời lượng 157,6 giây
