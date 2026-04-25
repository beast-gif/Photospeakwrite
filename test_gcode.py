"""
Test G-code generation in isolation. No OCR / STT required.
Simulates the output of the form scanner + voice layer with hardcoded data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.config import PaperConfig, PenConfig
from modules.gcode_generator import GCodeGenerator


def test_basic():
    paper = PaperConfig()
    pen = PenConfig()
    gen = GCodeGenerator(paper_cfg=paper, pen_cfg=pen)

    # Simulate scan_form() + voice answers
    fields_with_answers = [
        {"label": "name",     "x_mm": 35, "y_mm":  60, "width_mm": 100, "height_mm": 5,
         "answer": "AMRITH"},
        {"label": "place",    "x_mm": 35, "y_mm":  75, "width_mm": 100, "height_mm": 5,
         "answer": "BANGALORE"},
        {"label": "phone no", "x_mm": 35, "y_mm":  90, "width_mm": 100, "height_mm": 5,
         "answer": "9876543210"},
        {"label": "age",      "x_mm": 35, "y_mm": 105, "width_mm": 100, "height_mm": 5,
         "answer": "20"},
        {"label": "gender",   "x_mm": 35, "y_mm": 120, "width_mm": 100, "height_mm": 5,
         "answer": "MALE"},
    ]

    gen.begin()
    for f in fields_with_answers:
        gen.write_text(
            text=f["answer"],
            x_start_mm=f["x_mm"],
            y_baseline_mm=f["y_mm"],
            char_height_mm=f["height_mm"],
            max_width_mm=f["width_mm"],
        )
    gen.end()

    out_path = "output/test_form.gcode"
    Path("output").mkdir(exist_ok=True)
    gen.save(out_path)

    print(f"Generated {gen.line_count()} G-code lines -> {out_path}")
    print("\n--- First 25 lines ---")
    for line in gen.to_string().splitlines()[:25]:
        print(line)
    print("\n--- Last 10 lines ---")
    for line in gen.to_string().splitlines()[-10:]:
        print(line)

    # Sanity checks
    text = gen.to_string()
    assert "G21" in text, "Missing units"
    assert "G90" in text, "Missing absolute mode"
    assert "M2" in text, "Missing program end"
    assert "AMRITH" in text  # in the comment
    assert text.count("G1 X") > 50, "Too few writing moves; something is wrong"
    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    test_basic()
