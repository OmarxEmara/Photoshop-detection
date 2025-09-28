# EKYC – Egyptian ID (Front) | Step 1 Geometry + Local Path

This project tests **front-only** ID processing using **local image paths**:
- Detects the card in the photo
- Rectifies it to a canonical **856×540** canvas (Step 1)
- Runs YOLOv8 field detector on the rectified image
- (Optional) OCR: PaddleOCR for Arabic text; Tesseract for digits (NID)
- Prints JSON and saves artifacts (warped, overlay)

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
