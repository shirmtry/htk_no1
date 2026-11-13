# -*- coding: utf-8 -*-
"""upload_hf.py — Upload Docker image tar + sha256 lên Hugging Face Huy1223/DPL302m.

Yêu cầu: biến môi trường HF_TOKEN (hoặc ${{ secrets.HF_TOKEN }} trong GitHub Actions).
Chạy: python scripts/upload_hf.py
"""
import os

from huggingface_hub import HfApi

api = HfApi()
token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("ERROR: $HF_TOKEN không có trong môi trường.")

api.upload_file(
    path_or_fileobj="htk-no1-app.tar",
    path_in_repo="htk-no1-app.tar",
    repo_id="Huy1223/DPL302m",
    token=token,
    repo_type="model",
)
api.upload_file(
    path_or_fileobj="htk-no1-app.tar.sha256",
    path_in_repo="htk-no1-app.tar.sha256",
    repo_id="Huy1223/DPL302m",
    token=token,
    repo_type="model",
)
print("Uploaded to Hugging Face successfully.")
