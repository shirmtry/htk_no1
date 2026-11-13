# -*- coding: utf-8 -*-
"""record_demo.py — Tự động chụp ảnh UI + quay video demo chatbot pháp luật.

Chế độ:
    python -X utf8 scripts/record_demo.py [--url http://localhost:8501]
        → chụp 01_trang_chu + 10 câu hỏi (04..13) + quay video webm
    python -X utf8 scripts/record_demo.py --render-text <file.txt> <out.png> "<tiêu đề>"
        → render một khối text (vd: docker compose ps) thành ảnh PNG
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

BASE = Path(__file__).resolve().parent.parent
SHOTS = BASE / "demo" / "screenshots"
VIDEOS = BASE / "demo" / "videos"

# 10 câu hỏi demo theo yêu cầu (01_trang_chu chụp riêng)
QUESTIONS: list[tuple[str, str]] = [
    ("04_hoi_dap_co_ban", "Di chúc là gì?"),
    ("05_hoi_dap_phuc_tap", "Điều kiện để di chúc hợp pháp là gì?"),
    ("06_thu_tuc", "Công chứng di chúc tại UBND xã thủ tục thế nào?"),
    ("07_cross_law", "Công chứng di chúc theo BLDS và Luật Công chứng?"),
    ("08_guardrail_ru", "Công thức nấu phở bò ngon nhất là gì?"),
    ("09_guardrail_rp", "Tôi có thắng kiện không?"),
    ("10_rule_an_treo", "Án treo là gì?"),
    ("11_rule_tuoi", "Người 13 tuổi phạm tội giết người bị xử lý thế nào?"),
    ("12_di_chuc_khuyet_tat", "Con tôi bị khuyết tật, không được ghi trong di chúc, có được hưởng không?"),
    ("13_the_vi", "Cháu nội tôi có được thừa kế thay bố không?"),
]


def _font(size: int, bold: bool = False):
    """Font TrueType hỗ trợ tiếng Việt; fallback DejaVu của PIL."""
    candidates = (
        ["C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf"] if bold
        else ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf"]
    )
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def render_table(csv_path: Path, png_path: Path, title: str) -> None:
    """Render CSV thành bảng PNG (PASS xanh, FAIL đỏ) — cho 14/14b_eval_*.png."""
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, body = rows[0], rows[1:]
    ncol = len(header)
    fh, fb = _font(15, bold=True), _font(14)
    widths = [max(len(str(header[c])), *(len(str(r[c])) for r in body)) if body else
              len(str(header[c])) for c in range(ncol)]
    widths = [min(max(w, 8), 46) for w in widths]  # ký tự
    cw = [w * 8 + 24 for w in widths]
    row_h, margin, title_h = 30, 24, 64
    W = sum(cw) + margin * 2
    H = title_h + row_h * (len(body) + 1) + margin * 2 + 30
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((margin, 14), title, font=_font(20, bold=True), fill=(15, 23, 42))
    d.text((margin, 42), f"Nguồn: {csv_path.name}  ·  tái tạo 2026-11-13",
           font=_font(12), fill=(100, 116, 139))

    def draw_row(y: int, cells: list[str], is_header: bool, row: list[str] | None = None):
        x = margin
        for c in range(ncol):
            val = str(cells[c])
            if len(val) > widths[c]:
                val = val[: widths[c] - 1] + "…"
            if is_header:
                d.rectangle([x, y, x + cw[c], y + row_h], fill=(30, 58, 138))
                d.text((x + 8, y + 7), val, font=fh, fill="white")
            else:
                bg = (248, 250, 252) if row and body.index(row) % 2 else "white"
                color = (15, 23, 42)
                if row is not None and header[c].lower() == "status":
                    color = (22, 163, 74) if val.strip().upper() == "PASS" else (220, 38, 38)
                d.rectangle([x, y, x + cw[c], y + row_h], fill=bg, outline=(226, 232, 240))
                d.text((x + 8, y + 7), val, font=fb, fill=color)
            x += cw[c]

    y = title_h + margin
    draw_row(y, header, True)
    y += row_h
    for r in body:
        draw_row(y, [str(r[c]) if c < len(r) else "" for c in range(ncol)], False, r)
        y += row_h
    d.text((margin, H - 26), "Bảng trích nguyên văn từ CSV trong repo — không chỉnh sửa số liệu.",
           font=_font(12), fill=(100, 116, 139))
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)
    print(f"[RENDER] {png_path.name} ({W}x{H})")


def render_text_png(txt_path: Path, png_path: Path, title: str) -> None:
    """Render khối text (log/terminal) thành PNG — cho 15_docker_ps / 16_docker_build."""
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()[-60:] or [""]
    f = _font(14)
    char_w, row_h, margin, title_h = 8, 24, 20, 60
    maxlen = min(max(len(l) for l in lines), 160)
    W = max(maxlen * char_w + margin * 2, 640)
    H = title_h + row_h * len(lines) + margin * 2
    img = Image.new("RGB", (W, H), (15, 23, 42))  # nền tối kiểu terminal
    d = ImageDraw.Draw(img)
    d.text((margin, 12), title, font=_font(18, bold=True), fill=(148, 226, 213))
    d.text((margin, 36), "log tái tạo 2026-11-13 · nguồn: WSL2 Docker", font=_font(11),
           fill=(100, 116, 139))
    y = title_h + margin - 10
    for line in lines:
        d.text((margin, y), line[:160], font=f, fill=(226, 232, 240))
        y += row_h
    png_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(png_path)
    print(f"[RENDER] {png_path.name} ({W}x{H})")


def ask(page, question: str, expect_answers: int, timeout_s: int = 150) -> bool:
    """Gửi câu hỏi qua chat_input, chờ caption '⏱️' của assistant xuất hiện đủ."""
    ta = page.locator('[data-testid="stChatInput"] textarea').first
    ta.wait_for(state="visible", timeout=180_000)
    ta.click()
    ta.fill(question)
    ta.press("Enter")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        caps = page.locator('[data-testid="stChatMessage"]:has-text("⏱️")').count()
        busy = page.locator('text=Đang xử lý').count()
        if caps >= expect_answers and busy == 0:
            time.sleep(1.0)
            # cuộn khung chat xuống cuối để thấy câu trả lời mới nhất
            page.evaluate("""() => {
                document.querySelectorAll('div[data-testid="stVerticalBlock"]')
                    .forEach(el => { if (el.scrollHeight > el.clientHeight + 50) el.scrollTop = el.scrollHeight; });
                window.scrollTo(0, document.body.scrollHeight);
            }""")
            time.sleep(0.6)
            return True
        time.sleep(1.0)
    return False


def capture_ui(url: str) -> None:
    SHOTS.mkdir(parents=True, exist_ok=True)
    VIDEOS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 720},
            ignore_https_errors=True,
            record_video_dir=str(VIDEOS),
            record_video_size={"width": 1280, "height": 720},
        )
        page = ctx.new_page()
        print(f"[DEMO] Mở {url}")
        page.goto(url, wait_until="domcontentloaded", timeout=180_000)
        page.locator('[data-testid="stChatInput"] textarea').first.wait_for(
            state="visible", timeout=180_000)
        time.sleep(3)
        page.screenshot(path=str(SHOTS / "01_trang_chu.png"), full_page=True)
        print("[SHOT] 01_trang_chu")
        ok = 0
        for i, (name, q) in enumerate(QUESTIONS, 1):
            done = ask(page, q, i)
            page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)
            if done:
                ok += 1
                print(f"[SHOT] {name}: OK — {q[:60]}")
            else:
                print(f"[SHOT] {name}: TIMEOUT chờ caption, vẫn chụp màn hình hiện tại")
        page.close()
        ctx.close()  # đóng context → video webm được lưu
        browser.close()
    print(f"[DEMO] Xác nhận {ok}/{len(QUESTIONS)} câu trả lời có caption ⏱️")
    webms = sorted(VIDEOS.glob("*.webm"), key=lambda f: f.stat().st_mtime)
    if webms:
        print(f"[VIDEO] webm: {webms[-1]} ({webms[-1].stat().st_size // 1024} KB)")
    else:
        print("[VIDEO] Không có webm — sẽ dùng slideshow từ ảnh")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("DEMO_URL", "http://localhost:8501"))
    ap.add_argument("--render-text", nargs=3, metavar=("TXT", "PNG", "TITLE"))
    args = ap.parse_args()
    if args.render_text:
        txt, png, title = args.render_text
        render_text_png(Path(txt), Path(png), title)
        return
    capture_ui(args.url)


if __name__ == "__main__":
    main()
