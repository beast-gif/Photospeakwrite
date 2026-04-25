"""
PhotoSpeakWrite - Complete pipeline (single command).

Run this one file and it does everything:
  Step 1: Take/upload a form image
  Step 2: Scan and detect which fields need to be filled
  Step 3: Ask the user each field by voice
  Step 4: Capture the user's spoken answers
  Step 5: Generate G-code that writes the answers at the right coordinates

Usage:
    python main.py                              # auto-find form in inputs/
    python main.py inputs/my_form.jpg           # specify form
    python main.py inputs/my_form.jpg --dry-run # skip voice, use placeholders
"""

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installing as a package
sys.path.insert(0, str(Path(__file__).resolve().parent))

from modules.form_scanner import scan_form
from modules.voice_io import VoiceIO
from modules.gcode_generator import GCodeGenerator
from modules.config import PaperConfig, PenConfig


# Default folders
INPUTS_DIR = Path("inputs")
OUTPUT_DIR = Path("output")
TEMPLATE_DEFAULT = Path("templates/standard_form.json")


def find_form_image():
    """If user didn't pass an image, look for one in inputs/."""
    INPUTS_DIR.mkdir(exist_ok=True)
    candidates = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
        candidates.extend(INPUTS_DIR.glob(ext))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Multiple — pick most recent
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    print(f"  Multiple images in inputs/. Using newest: {candidates[0].name}")
    return candidates[0]


def step1_get_form(args):
    print()
    print("=" * 64)
    print("  STEP 1 / 5  ::  Form image")
    print("=" * 64)

    if args.image:
        path = Path(args.image)
    else:
        path = find_form_image()
        if path is None:
            print(f"\n  No image provided and inputs/ is empty.")
            print(f"  Drop a form image (jpg/png) into ./{INPUTS_DIR}/ and run again,")
            print(f"  or pass it directly:  python main.py path/to/form.jpg")
            sys.exit(1)

    if not path.exists():
        print(f"\n  ERROR: Image not found: {path}")
        sys.exit(1)

    print(f"  Using form image: {path}")
    return path


def step2_scan_form(image_path, template_path):
    print()
    print("=" * 64)
    print("  STEP 2 / 5  ::  Scanning form for fields to fill")
    print("=" * 64)

    template_str = str(template_path) if template_path and template_path.exists() else None
    if template_str:
        print(f"  Template fallback: {template_str}")

    try:
        fields = scan_form(str(image_path), template_path=template_str)
    except Exception as e:
        print(f"\n  ERROR during scan: {e}")
        if "tesseract" in str(e).lower():
            print("\n  Tesseract is not installed or not on PATH.")
            print("  Windows: https://github.com/UB-Mannheim/tesseract/wiki")
            print("  After installing, set this near the top of modules/form_scanner.py:")
            print("    pytesseract.pytesseract.tesseract_cmd = "
                  "r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        sys.exit(1)

    if not fields:
        print("\n  No fields detected.")
        print("  Try providing a template:  --template templates/standard_form.json")
        sys.exit(1)

    print(f"\n  Detected {len(fields)} field(s) to fill:")
    for i, f in enumerate(fields, 1):
        src = f['source']
        if src == 'ocr+line':
            src_marker = "[OCR+line]"   # most accurate: detected actual blank line
        elif src == 'ocr':
            src_marker = "[OCR]"        # label found, line inferred
        else:
            src_marker = "[template]"   # using hand-measured template coords
        print(f"    {i}. {f['label']:<16} {src_marker:<11} "
              f"@ paper (x={f['x_mm']:>5.1f}, y={f['y_mm']:>5.1f}) mm")
    return fields


def step3and4_collect_answers(fields, dry_run):
    print()
    print("=" * 64)
    print("  STEP 3 / 5  ::  Asking you to fill each field")
    print("  STEP 4 / 5  ::  Capturing your spoken answers")
    print("=" * 64)

    if dry_run:
        print("\n  DRY RUN - using placeholder answers (no microphone needed)")
        placeholders = {"name": "Amrith", "place": "Bangalore",
                        "phone no": "9876543210", "phone number": "9876543210",
                        "age": "20", "gender": "Male", "email": "amrith@example.com"}
        answers = {}
        for f in fields:
            answers[f['label']] = placeholders.get(f['label'].lower(), "TEST")
            print(f"    {f['label']:<14} -> {answers[f['label']]}")
        return answers

    print("\n  I will speak each question. Please answer clearly.")
    print("  Up to 3 attempts per field; if STT keeps failing it falls to typing.\n")

    vio = VoiceIO()
    answers = {}
    try:
        for i, f in enumerate(fields, 1):
            print(f"\n  --- Field {i}/{len(fields)}: {f['label']} ---")
            ans = vio.ask_and_capture(f['label'], expected_type=f.get('type', 'text'))
            answers[f['label']] = ans
            if ans:
                print(f"  [SAVED] {f['label']}: {ans}")
            else:
                print(f"  [SKIPPED] {f['label']} (no answer captured)")
    finally:
        vio.close()

    # Quick review with user
    print("\n  ---- Review captured answers ----")
    for label, ans in answers.items():
        marker = "OK " if ans else "-- "
        print(f"    {marker} {label:<14} : {ans!r}")

    if any(not v for v in answers.values()):
        print("\n  Some answers are empty. The arm will skip those fields.")

    try:
        confirm = input("\n  Proceed to generate G-code with these answers? [Y/n]: ").strip().lower()
    except EOFError:
        confirm = "y"
    if confirm and confirm[0] == "n":
        print("  Aborted. Re-run to try again.")
        sys.exit(0)

    return answers


def step5_generate_gcode(fields, answers, out_path):
    print()
    print("=" * 64)
    print("  STEP 5 / 5  ::  Generating G-code for robotic arm")
    print("=" * 64)

    paper = PaperConfig()
    pen = PenConfig()
    gen = GCodeGenerator(paper_cfg=paper, pen_cfg=pen)

    gen.begin()
    written = 0
    for f in fields:
        text = answers.get(f['label'], "")
        if not text:
            print(f"  Skipping '{f['label']}' (no answer)")
            continue
        gen.write_text(
            text=text,
            x_start_mm=f['x_mm'],
            y_baseline_mm=f['y_mm'],
            # Use a generous default char height; auto-shrink will reduce it
            # if the answer doesn't fit in the field's measured width.
            char_height_mm=4.0,
            max_width_mm=f.get('width_mm', 60.0),
        )
        written += 1
        print(f"  Wrote '{text}' at ({f['x_mm']:.1f}, {f['y_mm']:.1f}) mm")
    gen.end()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen.save(str(out_path))

    print(f"\n  G-code saved: {out_path}")
    print(f"  Lines: {gen.line_count()}  |  Fields written: {written}/{len(fields)}")
    return gen.line_count()


def save_answers(fields, answers, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        json.dump({"fields": fields, "answers": answers}, fp, indent=2)


def load_saved_answers(path):
    if not path.exists():
        print(f"  ERROR: No saved answers at {path}")
        sys.exit(1)
    with open(path) as fp:
        data = json.load(fp)
    return data["fields"], data["answers"]


def main():
    parser = argparse.ArgumentParser(
        description="PhotoSpeakWrite - scan form, ask user, generate G-code")
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to form image (or drop one in inputs/)")
    parser.add_argument("--template", default=str(TEMPLATE_DEFAULT),
                        help=f"Template JSON (default: {TEMPLATE_DEFAULT})")
    parser.add_argument("--out-gcode", default=str(OUTPUT_DIR / "form_output.gcode"))
    parser.add_argument("--out-answers", default=str(OUTPUT_DIR / "answers.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip voice; fill with placeholder answers")
    parser.add_argument("--use-saved-answers", action="store_true",
                        help="Skip steps 1-4; use answers.json from a previous run")
    args = parser.parse_args()

    print()
    print("#" * 64)
    print("#  PhotoSpeakWrite - Smart Robotic Writing Aid")
    print("#" * 64)

    if args.use_saved_answers:
        print("\n  Reusing saved answers from previous run...")
        fields, answers = load_saved_answers(Path(args.out_answers))
    else:
        # Step 1: Get the form image
        image_path = step1_get_form(args)

        # Step 2: Scan + detect fields
        template_path = Path(args.template) if args.template else None
        fields = step2_scan_form(image_path, template_path)

        # Steps 3 & 4: Voice prompt + answer capture
        answers = step3and4_collect_answers(fields, args.dry_run)

        # Persist answers in case anything goes wrong later
        save_answers(fields, answers, Path(args.out_answers))
        print(f"\n  Answers persisted to {args.out_answers}")

    # Step 5: G-code generation
    step5_generate_gcode(fields, answers, Path(args.out_gcode))

    # Friendly tail
    print()
    print("=" * 64)
    print("  ALL STEPS COMPLETE")
    print("=" * 64)
    print(f"  Answers JSON: {args.out_answers}")
    print(f"  G-code file:  {args.out_gcode}")
    print()
    print("  To preview what the arm will draw before sending it:")
    print(f"      python render_gcode.py {args.out_gcode} output/preview.png")
    print()


if __name__ == "__main__":
    main()