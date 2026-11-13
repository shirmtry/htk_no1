import os
import sys
from huggingface_hub import HfApi

def main():
    token = os.environ.get("HF_TOKEN")
    if not token or token.strip() == "":
        print("[!] LỖI: Biến môi trường HF_TOKEN đang bị rỗng hoặc không tồn tại!")
        sys.exit(1)

    api = HfApi()
    repo_id = "Huy1223/DPL302m"

    try:
        print("Đang upload htk-no1-app.tar...")
        api.upload_file(path_or_fileobj="htk-no1-app.tar", path_in_repo="htk-no1-app.tar", repo_id=repo_id, token=token, repo_type="dataset")
        print("Đang upload htk-no1-app.tar.sha256...")
        api.upload_file(path_or_fileobj="htk-no1-app.tar.sha256", path_in_repo="htk-no1-app.tar.sha256", repo_id=repo_id, token=token, repo_type="dataset")
        print("=> UPLOAD THÀNH CÔNG!")
    except Exception as e:
        print(f"Lỗi upload: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
