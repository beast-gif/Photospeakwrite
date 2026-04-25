"""
Standalone voice test — no OCR, no G-code.

Either:
  - asks a default set of questions (name, place, phone, age, gender), or
  - asks a custom list of fields you specify, or
  - asks fields detected by OCR from an image (combined OCR+STT test)

Saves all answers to output/answers.json so you can inspect what STT captured.

Usage:
    python test_stt.py
    python test_stt.py --fields name,place,age
    python test_stt.py --from-image my_form.jpg --template templates/standard_form.json
"""

import argparse
import json
import sys
from pathlib import Path

from modules.voice_io import VoiceIO
from modules.config import get_field_type


DEFAULT_FIELDS = [
    {"label": "name",     "type": "text"},
    {"label": "place",    "type": "text"},
    {"label": "phone no", "type": "digits"},
    {"label": "age",      "type": "digits"},
    {"label": "gender",   "type": "choice:male,female,other"},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default=None,
                    help="Comma-separated list of field names to ask, e.g. 'name,age'")
    ap.add_argument("--from-image", default=None,
                    help="Use OCR on this image to determine which questions to ask")
    ap.add_argument("--template", default=None,
                    help="Template JSON (used with --from-image)")
    ap.add_argument("--out", default="output/answers.json",
                    help="Where to save the captured answers")
    args = ap.parse_args()

    print("=" * 60)
    print("  Voice (TTS + STT) Test")
    print("=" * 60)

    # ---- Decide which fields to ask ----
    if args.from_image:
        from modules.form_scanner import scan_form
        print(f"\n  Running OCR on {args.from_image}...")
        detected = scan_form(args.from_image, template_path=args.template)
        if not detected:
            print("  No fields detected from image. Falling back to defaults.")
            fields = DEFAULT_FIELDS
        else:
            fields = [{"label": f['label'], "type": f['type']} for f in detected]
            print(f"  OCR found {len(fields)} field(s) to ask about.")
    elif args.fields:
        names = [n.strip() for n in args.fields.split(",") if n.strip()]
        fields = [{"label": n, "type": get_field_type(n)} for n in names]
    else:
        fields = DEFAULT_FIELDS

    print(f"\n  Will ask {len(fields)} question(s):")
    for f in fields:
        print(f"    - {f['label']} ({f['type']})")

    # ---- Ask them ----
    print("\n" + "-" * 60)
    print("  Starting voice interaction...")
    print("-" * 60)
    print("  Speak clearly when you see 'Listening...'")
    print("  You'll get up to 3 attempts per field.")
    print()

    vio = VoiceIO()
    answers = {}
    try:
        for i, f in enumerate(fields, 1):
            print(f"\n  [Question {i}/{len(fields)}]")
            ans = vio.ask_and_capture(f['label'], expected_type=f['type'])
            answers[f['label']] = ans
            print(f"  ==> Saved '{f['label']}' = '{ans}'")
    finally:
        vio.close()

    # ---- Save and print ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as fp:
        json.dump({
            "fields": fields,
            "answers": answers,
        }, fp, indent=2)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for label, ans in answers.items():
        marker = "✓" if ans else "✗ (empty)"
        print(f"  {marker} {label:<15} -> '{ans}'")
    print(f"\n  Answers saved to {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
