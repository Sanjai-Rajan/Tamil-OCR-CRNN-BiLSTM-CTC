import sys
import os
import requests
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from app.inference import OCRService

def main():
    img_path = root_dir / "sample_test.png"
    if not img_path.exists():
        print(f"Sample image not found: {img_path}")
        return

    # Direct inference
    print("Running direct inference...")
    ocr = OCRService()
    img = Image.open(img_path)
    direct_res = ocr.predict(img)
    print("Direct Prediction:")
    print(direct_res)

    # Frontend inference
    print("\nRunning frontend inference via HTTP...")
    url = "http://127.0.0.1:8000/api/ocr"
    try:
        with open(img_path, "rb") as f:
            files = {"file": ("sample_test.png", f, "image/png")}
            resp = requests.post(url, files=files)
        
        if resp.status_code == 200:
            frontend_res = resp.json()
            print("Frontend Prediction:")
            print(frontend_res)
            
            if direct_res["text"] == frontend_res["text"]:
                print("\nMATCH")
            else:
                print("\nDIFFERENT")
        else:
            print(f"Frontend failed with status {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Frontend request failed: {e}")

if __name__ == "__main__":
    main()
