import os

from huggingface_hub import HfApi

REPO_ID = "Huy1223/DPL302m"

token = os.environ.get("HF_TOKEN")
if not token:
    raise SystemExit("ERROR: $HF_TOKEN khong co trong moi truong.")

api = HfApi()
api.upload_file(
    path_or_fileobj="htk-no1-app.tar",
    path_in_repo="htk-no1-app.tar",
    repo_id=REPO_ID,
    token=token,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj="htk-no1-app.tar.sha256",
    path_in_repo="htk-no1-app.tar.sha256",
    repo_id=REPO_ID,
    token=token,
    repo_type="dataset",
)
print("Upload OK ->", REPO_ID)
