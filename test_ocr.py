"""
Standalone OCR test — no voice, no G-code.

Takes a form image, runs the form scanner, and prints exactly what fields
were detected and where their blanks are on the paper.

Usage:
    python test_ocr.py path/to/form.jpg
    python test_ocr.py path/to/form.jpg --template templates/standard_form.json
    python test_ocr.py path/to/form.jpg --debug      # save annotated preview image
"""

import argparse
import sys
from pathlib import Path

import cv2

from modules.form_scanner import scan_form, preprocess
from modules.config import OCRConfig


def annotate(image_path: str, fields: list, out_path: str):
    """Draw red boxes where blanks were detected and save a preview image."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [WARN] Could not load image for annotation.")
        return
    cfg = OCRConfig()
    for f in fields:
        x_px = int(f['x_mm'] * cfg.px_per_mm)
        y_px = int(f['y_mm'] * cfg.px_per_mm)
        w_px = int(f['width_mm'] * cfg.px_per_mm)
        h_px = int(f['height_mm'] * cfg.px_per_mm)
        # Draw box (blank starts at x_px, baseline at y_px, height upward)
        top = max(0, y_px - h_px)
        color = (0, 0, 255) if f['source'] == 'ocr' else (0, 165, 255)
        cv2.rectangle(img, (x_px, top), (x_px + w_px, y_px), color, 2)
        cv2.putText(img, f['label'], (x_px, top - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(out_path, img)
    print(f"  Annotated preview saved -> {out_path}")
    print(f"  (red box = OCR detected, orange box = template fallback)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to scanned form image")
    ap.add_argument("--template", default=None, help="Optional template JSON")
    ap.add_argument("--debug", action="store_true", help="Save annotated preview")
    args = ap.parse_args()

    if not Path(args.image).exists():
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    print("=" * 60)
    print("  OCR Test")
    print("=" * 60)
    print(f"  Image:    {args.image}")
    print(f"  Template: {args.template or '(none)'}")
    print()

    # Save preprocessed view if debugging
    if args.debug:
        img = cv2.imread(args.image)
        pre = preprocess(img)
        Path("output").mkdir(exist_ok=True)
        cv2.imwrite("output/preprocessed.png", pre)
        print("  Preprocessed image -> output/preprocessed.png")
        print()

    fields = scan_form(args.image, template_path=args.template)

    print()
    print("-" * 60)
    print(f"  RESULTS: {len(fields)} field(s) detected")
    print("-" * 60)
    if not fields:
        print("\n  No fields detected. Things to check:")
        print("    1. Is Tesseract installed? (pytesseract is just a wrapper)")
        print("       Windows: download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("    2. Is the image clear and high-contrast?")
        print("    3. Try --debug to inspect the preprocessed image")
        print("    4. Try providing --template templates/standard_form.json")
        sys.exit(1)

    for f in fields:
        src = f['source']
        conf = f.get('confidence', 0)
        print(f"\n  Label:    '{f['label']}'  (type: {f['type']})")
        print(f"  Source:   {src} (conf={conf:.0f})")
        print(f"  Position: x={f['x_mm']:.1f} mm, y={f['y_mm']:.1f} mm (paper frame)")
        print(f"  Size:     {f['width_mm']:.1f} mm wide × {f['height_mm']:.1f} mm tall")

    if args.debug:
        Path("output").mkdir(exist_ok=True)
        annotate(args.image, fields, "output/ocr_annotated.png")

    print("\n" + "=" * 60)
    print("  OCR test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
