"""
Diagnostic tool — overlays detected fields on the input image so you can
SEE exactly what OCR + line detection found, without running the full
voice/G-code pipeline.

Usage:
    python debug_form.py inputs/your_photo.jpg

Produces output/debug_overlay.png showing:
- Green boxes around detected labels (with the label text)
- Red horizontal lines where the line detector saw blanks
- Yellow boxes showing where the system will write each answer
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.form_scanner import (
    detect_horizontal_lines, detect_fields_ocr_px,
    scan_form, preprocess, find_form_bounding_box
)
from modules.config import OCRConfig

import pytesseract


def main(image_path: str, template_path: str = "templates/standard_form.json"):
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not load image {image_path}")
        sys.exit(1)

    cfg = OCRConfig()
    overlay = img.copy()

    # 1. Show the preprocessed binary
    pre = preprocess(img)
    Path("output").mkdir(exist_ok=True)
    cv2.imwrite("output/debug_preprocessed.png", pre)
    print(f"  Preprocessed binary -> output/debug_preprocessed.png")

    # 1.5 Show the form bounding box in MAGENTA
    fbb = find_form_bounding_box(img)
    cv2.rectangle(overlay, (fbb[0], fbb[1]), (fbb[2], fbb[3]), (255, 0, 255), 4)
    cv2.putText(overlay, "FORM AREA",
                (fbb[0], max(0, fbb[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    print(f"  Form bounding box: ({fbb[0]}, {fbb[1]}) to ({fbb[2]}, {fbb[3]})")

    # 2. Draw all detected horizontal lines in RED
    lines = detect_horizontal_lines(img, cfg)
    print(f"\n  Detected {len(lines)} horizontal line(s):")
    for i, (x1, y, x2) in enumerate(lines):
        cv2.line(overlay, (x1, y), (x2, y), (0, 0, 255), 3)
        cv2.putText(overlay, f"line{i}", (x1 + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3. Draw all OCR-detected words in BLUE
    data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT)
    print(f"\n  Tesseract words (conf >= 30):")
    for i, t in enumerate(data['text']):
        word = t.strip()
        if not word:
            continue
        try:
            conf = float(data['conf'][i])
        except ValueError:
            continue
        if conf < 30:
            continue
        x, y, w, h = (data['left'][i], data['top'][i],
                      data['width'][i], data['height'][i])
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 100, 0), 1)
        cv2.putText(overlay, f"{word}", (x, max(0, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 0), 1)
        print(f"    {word!r:<25} conf={conf:>4.0f}")

    # 4. Run pixel-space field detection and draw YELLOW boxes there
    px_fields = detect_fields_ocr_px(img, cfg)
    print(f"\n  Pixel-space accepted fields: {len(px_fields)}")
    for f in px_fields:
        x_px = int(f['x_px'])
        y_px = int(f['y_px'])
        w_px = int(f['w_px'])
        h_px = int(f['h_px'])
        top = max(0, y_px - h_px)
        cv2.rectangle(overlay, (x_px, top), (x_px + w_px, y_px), (0, 255, 255), 3)
        cv2.putText(overlay, f"{f['label']} [{f['source']}]",
                    (x_px, max(0, top - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        print(f"    {f['label']:<18} src={f['source']:<10} "
              f"px=({x_px},{y_px}) w={w_px}px")

    # 5. Now also run scan_form (which scales to target paper) for the final mm coords
    print(f"\n  Final mm coordinates on target A4 paper:")
    final_fields = scan_form(image_path)
    for f in final_fields:
        print(f"    {f['label']:<18} @ ({f['x_mm']:.1f}, {f['y_mm']:.1f}) mm  w={f['width_mm']:.1f}mm")

    out = "output/debug_overlay.png"
    cv2.imwrite(out, overlay)
    print(f"\n  Overlay -> {out}")
    print("\n  Color key:")
    print("    Magenta box  = form bounding box (auto-detected content area)")
    print("    Blue boxes   = words Tesseract saw")
    print("    Red lines    = horizontal blank lines detected")
    print("    Yellow boxes = where the arm WILL write (in image pixel space)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_form.py inputs/your_photo.jpg [templates/standard_form.json]")
        sys.exit(1)
    image = sys.argv[1]
    template = sys.argv[2] if len(sys.argv) > 2 else "templates/standard_form.json"
    main(image, template)