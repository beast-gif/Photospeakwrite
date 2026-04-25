"""
Generate a sample blank form image for testing the pipeline.
Field positions match templates/standard_form.json.

Usage:
    python make_sample_form.py
"""

import cv2
import numpy as np
from pathlib import Path


def main():
    # A4 at 200 DPI: 210mm × 297mm = 1654 × 2339 px
    DPI = 200
    PX_PER_MM = DPI / 25.4
    W = int(210 * PX_PER_MM)
    H = int(297 * PX_PER_MM)
    img = np.full((H, W, 3), 255, dtype=np.uint8)

    def mm_to_px(mm): return int(mm * PX_PER_MM)

    # Title at top
    cv2.putText(img, "REGISTRATION FORM",
                (mm_to_px(60), mm_to_px(30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

    # Fields - x positions match standard_form.json
    fields = [
        ("name:",     60),
        ("place:",    75),
        ("phone no:", 90),
        ("age:",      105),
        ("gender:",   120),
    ]
    for label, y_mm in fields:
        # Label
        cv2.putText(img, label,
                    (mm_to_px(15), mm_to_px(y_mm)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Underline for the blank
        cv2.line(img,
                 (mm_to_px(35), mm_to_px(y_mm) + 2),
                 (mm_to_px(135), mm_to_px(y_mm) + 2),
                 (0, 0, 0), 1)

    out = Path("inputs/sample_form.png")
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), img)
    print(f"Sample form saved -> {out}")
    print(f"Now run:  python main.py")


if __name__ == "__main__":
    main()
