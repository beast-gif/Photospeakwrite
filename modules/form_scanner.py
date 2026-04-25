"""
Form scanner module.

Strategy:
  1. Preprocess the image (grayscale, deskew, denoise, threshold).
  2. Run Tesseract with bounding boxes to detect label text ("name:", "age:", etc.).
  3. For each detected label, the blank space is to the RIGHT of it on the same line.
  4. If OCR confidence is low or labels are missing, fall back to a template JSON
     that maps known labels -> (x_mm, y_mm, width_mm) coordinates.
  5. Convert all pixel coords to millimetres on the paper.
"""

import json
import re
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
    HAVE_TESSERACT = True
except ImportError:
    HAVE_TESSERACT = False

from .config import OCRConfig, get_field_type


# Labels we care about (lowercased, ":" stripped).
KNOWN_LABELS = {
    # Personal
    "name", "first name", "last name", "father's name", "mother's name",
    "age", "gender", "date", "dob", "date of birth",
    # Location
    "place", "city", "state", "country", "address", "pin code", "zip code",
    "postal code",
    # Contact
    "phone no", "phone number", "mobile", "mobile number", "email",
    # Identity / IDs
    "aadhar number", "aadhaar number", "aadhar", "aadhaar",
    "passport number", "passport",
    "pan number", "pan",
    "voter id", "license number", "driving license",
    # Education / work
    "occupation", "qualification", "company", "designation",
}

# Minimum OCR confidence to trust a detected label.
# Lower means more permissive (catches short labels like "DOB" but may
# also pick up garbage). 40 is a good balance for printed forms.
MIN_LABEL_CONFIDENCE = 40


def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Grayscale + lighting normalization + adaptive threshold.

    Tuned to be robust on:
    - Clean digital scans (works as before)
    - Camera photos of paper with uneven lighting / shadows
    - Slightly out-of-focus shots
    - Lighter or darker pen strokes
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Lighting normalization: divide by a heavy blur to flatten shadows.
    # Crucial for camera photos where one corner is darker than another.
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    # Avoid divide-by-zero
    blur = np.where(blur == 0, 1, blur)
    normalized = np.clip((gray.astype(np.float32) / blur) * 128, 0, 255).astype(np.uint8)

    # Bilateral filter: denoise but preserve edges (text strokes, lines)
    smoothed = cv2.bilateralFilter(normalized, d=7, sigmaColor=50, sigmaSpace=50)

    # Otsu threshold gives a clean binary; works well on the normalized image
    _, binary = cv2.threshold(smoothed, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Drop tiny specks
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cleaned


def deskew(gray: np.ndarray) -> np.ndarray:
    """Estimate skew via minAreaRect on dark pixels and rotate."""
    coords = np.column_stack(np.where(gray < 128))
    if coords.size == 0:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# OCR commonly misreads short labels. Map common misreads to the canonical label.
# Add to this map as you encounter new misreads in your real-world forms.
OCR_ALIASES = {
    "0ob": "dob",
    "d0b": "dob",
    "ddb": "dob",
    "do8": "dob",
    "dos": "dob",
    "doa": "dob",
    "dob,": "dob",
    "n4me": "name",
    "nane": "name",
    "narne": "name",
    "agc": "age",
    "agt": "age",
    "ace": "age",        # AGE often misread as ACE on photos
    "c1ty": "city",
    "ctty": "city",
    "cit": "city",       # CITY often gets cut to CIT:
    "cilty": "city",
    "phon3": "phone",
    "gendcr": "gender",
    "gende": "gender",
    "render": "gender",
    "aadhaar": "aadhar",
    "aadhar:": "aadhar",
}


def normalize_label(text: str) -> str:
    """Clean OCR text: lowercase, strip punctuation, collapse spaces, apply alias."""
    t = text.lower().strip()
    t = re.sub(r'[:\-_.;,!]+$', '', t)     # trailing punctuation (incl. semicolons)
    t = re.sub(r'\s+', ' ', t)
    t = t.strip()
    parts = [OCR_ALIASES.get(p, p) for p in t.split()]
    return " ".join(parts)


def detect_horizontal_lines(img_bgr: np.ndarray, ocr_cfg: OCRConfig) -> list:
    """
    Find horizontal lines (form blanks/underlines) in the image.

    Returns a list of (x1, y, x2) tuples in pixel coordinates.

    A real form blank line is THIN (≤1mm tall) and LONG (≥30mm).
    Bold label text rows can look like horizontal bars after morphology,
    so this function uses two strict tests:
      1. The component must have a very high width-to-height ratio (≥30:1)
      2. The component must be thinner than typical text (≤1mm)
    These together reliably reject text rows while keeping real underscores.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Lighting normalization (matches preprocess())
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    blur = np.where(blur == 0, 1, blur)
    normalized = np.clip((gray.astype(np.float32) / blur) * 128, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(normalized, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Suppress text by removing tall vertical strokes first.
    # Most letters have vertical strokes ≥2mm tall; we erode those away.
    text_height_px = max(2, int(ocr_cfg.px_per_mm * 1.5))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, text_height_px))
    text_suppressed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
    # text_suppressed still contains tall things — but we want to KEEP horizontal
    # thin lines, so subtract this from the original to remove text:
    text_only = cv2.dilate(text_suppressed, np.ones((3,3), np.uint8))
    lines_only = cv2.subtract(binary, text_only)

    # Now keep only long horizontal runs in lines_only.
    # Real form underlines are at least 30mm long.
    min_line_width_px = int(30 * ocr_cfg.px_per_mm)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_width_px, 1))
    horizontal = cv2.morphologyEx(lines_only, cv2.MORPH_OPEN, horiz_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        horizontal, connectivity=8)

    lines = []
    max_line_thickness_px = max(2, int(ocr_cfg.px_per_mm * 1.0))   # ≤1 mm
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        # Strict thickness test: real underlines are very thin
        if h > max_line_thickness_px:
            continue
        if w < min_line_width_px:
            continue
        # Strict aspect ratio: real lines are at least 30× wider than tall
        if w < h * 30:
            continue
        lines.append((int(x), int(y + h / 2), int(x + w)))

    lines.sort(key=lambda L: L[1])
    return lines


def find_line_for_label(lbl_left: int, lbl_right: int, lbl_bot: int,
                         lines: list, max_dy_px: int):
    """
    Given a label's bounding box, find the horizontal line that belongs to it.

    A real form line is on the SAME ROW as the label, starting just after
    the label's right edge. We do not consider lines below the label,
    because in real forms the text and the line are always aligned vertically.

    Returns (x_start, y_baseline, x_end) of the chosen line in pixels,
    or None if no matching line was found.
    """
    candidates = []
    for (x1, y, x2) in lines:
        # Vertical alignment: line's center must be within label's vertical span
        # (with some tolerance for descenders)
        if not (abs(y - lbl_bot) <= max_dy_px):
            continue
        # Horizontal alignment: line must START at or just after label's right edge.
        # If the line starts to the LEFT of the label, it belongs to a different field.
        if x1 < lbl_right - max_dy_px:
            continue
        # And the line must start within ~30mm right of the label
        # (otherwise it's too far away to be this label's blank)
        if x1 > lbl_right + max_dy_px * 6:
            continue
        candidates.append((x1, y, x2))

    if not candidates:
        return None
    # Pick the one whose start is closest to label's right edge
    candidates.sort(key=lambda L: abs(L[0] - lbl_right))
    return candidates[0]


def detect_fields_ocr(img_bgr: np.ndarray, ocr_cfg: OCRConfig) -> list:
    """
    Run Tesseract with bounding boxes, extract label-like words,
    detect actual horizontal blank lines, and pair them with labels.
    """
    if not HAVE_TESSERACT:
        print("  [WARN] pytesseract not available; OCR detection skipped.")
        return []

    pre = preprocess(img_bgr)
    h_img, w_img = pre.shape[:2]

    # Detect actual writing lines on the form (the underscores/horizontal
    # bars next to each label). We'll snap each label's writing position
    # to its nearest detected line — this is far more accurate than just
    # guessing "to the right of the label".
    blank_lines = detect_horizontal_lines(img_bgr, ocr_cfg)
    if blank_lines:
        print(f"  Detected {len(blank_lines)} horizontal blank line(s)")

    # image_to_data returns word-level boxes with confidence
    data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT)

    # Group words into lines so multi-word labels like "phone no" stay together
    lines = {}
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if not word:
            continue
        try:
            conf = float(data['conf'][i])
        except ValueError:
            conf = -1
        # Normally filter by confidence, but rescue conf=0 entries whose
        # normalized form matches a known label or alias — Tesseract gives
        # conf=0 when a region looks like text but recognition was uncertain,
        # which is common for short labels like "DOB:".
        if conf < MIN_LABEL_CONFIDENCE:
            normalized = normalize_label(word)
            if normalized not in KNOWN_LABELS and normalized not in OCR_ALIASES:
                continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        lines.setdefault(key, []).append({
            'text': word,
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i],
            'conf': conf,
        })

    fields = []
    for key, words in lines.items():
        words.sort(key=lambda w: w['left'])
        raw_line = " ".join(w['text'] for w in words).lower()

        for n in (3, 2, 1):
            if n > len(words):
                continue
            candidate = " ".join(w['text'] for w in words[:n])
            norm = normalize_label(candidate)
            if norm not in KNOWN_LABELS:
                continue

            after_label = raw_line[len(candidate):].strip()
            looks_like_label = (
                after_label.startswith((":", "-", "_"))
                or after_label == ""
                or n == len(words)
            )
            if not looks_like_label:
                continue

            # Bounding box of the label text
            lbl_words = words[:n]
            lbl_left = min(w['left'] for w in lbl_words)
            lbl_right = max(w['left'] + w['width'] for w in lbl_words)
            lbl_top = min(w['top'] for w in lbl_words)
            lbl_bot = max(w['top'] + w['height'] for w in lbl_words)
            lbl_height = lbl_bot - lbl_top

            # Try to find the actual writing line for this label
            max_dy_px = max(int(lbl_height * 1.5), int(ocr_cfg.px_per_mm * 5))
            matched_line = find_line_for_label(
                lbl_left, lbl_right, lbl_bot, blank_lines, max_dy_px)

            if matched_line is not None:
                # Use the detected blank line's actual coordinates
                line_x_start, line_y, line_x_end = matched_line
                # Start writing 1 mm into the line (small inset)
                inset_px = int(ocr_cfg.px_per_mm * 1)
                blank_left_px = line_x_start + inset_px
                blank_right_px = line_x_end
                # Baseline is the line itself; but humans write ABOVE the line,
                # so put baseline ~1 mm above the line so descenders sit on it
                baseline_offset_px = int(ocr_cfg.px_per_mm * 1)
                blank_baseline_px = line_y - baseline_offset_px
                # Use label height as a proxy for writing height
                blank_height_px = lbl_height
                source = 'ocr+line'
            else:
                # No underline found — write to the right of the label, on the
                # same row. This is the right approach for forms like
                # "NAME:    [empty space]" with no underline.
                gap_px = int(ocr_cfg.px_per_mm * 3)   # 3 mm gap after the colon
                blank_left_px = lbl_right + gap_px
                # Cap the writing width at 100mm — long enough for any
                # reasonable answer, short enough to leave margin for the
                # next column or page edge.
                max_blank_width_px = int(ocr_cfg.px_per_mm * 100)
                blank_right_px = min(
                    blank_left_px + max_blank_width_px,
                    w_img - int(ocr_cfg.px_per_mm * 10)
                )
                # Baseline is the label's baseline (its bottom edge)
                blank_baseline_px = lbl_bot
                blank_height_px = lbl_height
                source = 'ocr'

            blank_width_px = max(0, blank_right_px - blank_left_px)

            fields.append({
                'label': norm,
                'type': get_field_type(norm),
                'x_mm': ocr_cfg.px_to_mm(blank_left_px),
                'y_mm': ocr_cfg.px_to_mm(blank_baseline_px),
                'width_mm': ocr_cfg.px_to_mm(blank_width_px),
                'height_mm': ocr_cfg.px_to_mm(blank_height_px),
                'source': source,
                'confidence': sum(w['conf'] for w in lbl_words) / len(lbl_words),
            })
            break
    return fields


def detect_fields_ocr_px(img_bgr: np.ndarray, ocr_cfg: OCRConfig) -> list:
    """
    Same as detect_fields_ocr but returns pixel coordinates.

    Each field dict has: label, type, x_px, y_px, w_px, h_px, source, confidence.
    The caller is responsible for converting pixels to physical mm
    using whatever calibration is appropriate for the target paper.
    """
    if not HAVE_TESSERACT:
        print("  [WARN] pytesseract not available; OCR detection skipped.")
        return []

    pre = preprocess(img_bgr)
    h_img, w_img = pre.shape[:2]

    blank_lines = detect_horizontal_lines(img_bgr, ocr_cfg)
    if blank_lines:
        print(f"  Detected {len(blank_lines)} horizontal blank line(s)")
    else:
        print(f"  No horizontal blank lines detected (form may have only labels)")

    data = pytesseract.image_to_data(pre, output_type=pytesseract.Output.DICT)

    grouped = {}
    for i in range(len(data['text'])):
        word = data['text'][i].strip()
        if not word:
            continue
        try:
            conf = float(data['conf'][i])
        except ValueError:
            conf = -1
        if conf < MIN_LABEL_CONFIDENCE:
            normalized = normalize_label(word)
            if normalized not in KNOWN_LABELS and normalized not in OCR_ALIASES:
                continue
        key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
        grouped.setdefault(key, []).append({
            'text': word,
            'left': data['left'][i],
            'top': data['top'][i],
            'width': data['width'][i],
            'height': data['height'][i],
            'conf': conf,
        })

    fields = []
    for key, words in grouped.items():
        words.sort(key=lambda w: w['left'])
        raw_line = " ".join(w['text'] for w in words).lower()

        for n in (3, 2, 1):
            if n > len(words):
                continue
            candidate = " ".join(w['text'] for w in words[:n])
            norm = normalize_label(candidate)
            if norm not in KNOWN_LABELS:
                continue

            after_label = raw_line[len(candidate):].strip()
            looks_like_label = (
                after_label.startswith((":", "-", "_", ";"))   # ; common OCR misread
                or after_label == ""
                or n == len(words)
            )
            if not looks_like_label:
                continue

            lbl_words = words[:n]
            lbl_left = min(w['left'] for w in lbl_words)
            lbl_right = max(w['left'] + w['width'] for w in lbl_words)
            lbl_top = min(w['top'] for w in lbl_words)
            lbl_bot = max(w['top'] + w['height'] for w in lbl_words)
            lbl_height = lbl_bot - lbl_top

            # Use a px-based tolerance for line matching (5% of label height)
            max_dy_px = max(int(lbl_height * 1.5), int(ocr_cfg.px_per_mm * 5))
            matched_line = find_line_for_label(
                lbl_left, lbl_right, lbl_bot, blank_lines, max_dy_px)

            if matched_line is not None:
                line_x_start, line_y, line_x_end = matched_line
                inset_px = int(ocr_cfg.px_per_mm * 1)
                blank_left = line_x_start + inset_px
                blank_right = line_x_end
                baseline_offset = int(ocr_cfg.px_per_mm * 1)
                blank_baseline = line_y - baseline_offset
                blank_height = lbl_height
                source = 'ocr+line'
            else:
                # Write right of the label, on the same row
                gap_px = max(int(lbl_height * 0.3), 5)   # small gap after colon
                blank_left = lbl_right + gap_px
                # Width: stop before the next text on this row, or the image edge
                next_text_left = w_img
                for w_dict in words[n:]:
                    if w_dict['left'] > blank_left:
                        next_text_left = min(next_text_left, w_dict['left'])
                blank_right = min(next_text_left - gap_px,
                                  w_img - int(ocr_cfg.px_per_mm * 5))
                blank_baseline = lbl_bot
                blank_height = lbl_height
                source = 'ocr'

            blank_width = max(0, blank_right - blank_left)

            fields.append({
                'label': norm,
                'type': get_field_type(norm),
                'x_px': blank_left,
                'y_px': blank_baseline,
                'w_px': blank_width,
                'h_px': blank_height,
                'source': source,
                'confidence': sum(w['conf'] for w in lbl_words) / len(lbl_words),
            })
            break
    return fields


def load_template(template_path: str) -> list:
    """Load a fallback template defining known field positions."""
    p = Path(template_path)
    if not p.exists():
        return []
    with open(p) as fp:
        data = json.load(fp)
    fields = []
    for entry in data.get('fields', []):
        label = normalize_label(entry['label'])
        fields.append({
            'label': label,
            'type': get_field_type(label),
            'x_mm': float(entry['x_mm']),
            'y_mm': float(entry['y_mm']),
            'width_mm': float(entry.get('width_mm', 60.0)),
            'height_mm': float(entry.get('height_mm', 5.0)),
            'source': 'template',
            'confidence': 100.0,
        })
    return fields


def merge_fields(ocr_fields: list, template_fields: list) -> list:
    """
    Coordinates come from the image being processed — NOT from a template.
    Template coordinates would only be valid if every form had the exact
    same physical layout, which is never true in practice.

    Logic:
    1. Deduplicate OCR detections by label (keep topmost — handles cases
       where a label word appears in a form title).
    2. Always use OCR-derived coordinates, whether from line detection
       (source='ocr+line') or label-position fallback (source='ocr').
    3. The template (if provided) is only used to know which labels are
       valid field names — it does NOT contribute coordinates.
    """
    # Step 1: dedupe by label, keep topmost occurrence
    ocr_by_label = {}
    for f in sorted(ocr_fields, key=lambda f: f['y_mm']):
        if f['label'] not in ocr_by_label:
            ocr_by_label[f['label']] = f
    deduped = list(ocr_by_label.values())
    deduped.sort(key=lambda f: f['y_mm'])
    return deduped


def find_form_bounding_box(img_bgr: np.ndarray) -> tuple:
    """
    Find the bounding box that contains all the form content (text + lines).

    Returns (left, top, right, bottom) in pixel coordinates, defining
    the area of the image that contains actual form content. This works
    regardless of whether the image is a screenshot, photo, or scan,
    or whether the surrounding margins are different sizes.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (51, 51), 0)
    blur = np.where(blur == 0, 1, blur)
    normalized = np.clip((gray.astype(np.float32) / blur) * 128, 0, 255).astype(np.uint8)
    _, binary = cv2.threshold(normalized, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Drop tiny specks (noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(cleaned)
    if coords is None:
        h, w = img_bgr.shape[:2]
        return (0, 0, w, h)
    x, y, w, h = cv2.boundingRect(coords)
    return (x, y, x + w, y + h)


def scan_form(image_path: str, template_path: str = None,
              target_paper_width_mm: float = 210.0,
              target_paper_height_mm: float = 297.0) -> list:
    """
    Main entry point: returns a list of field dicts with coordinates in
    mm on the TARGET paper (the paper the robotic arm will write on).

    Strategy:
    1. Find the form's content bounding box on the image (handles photos,
       screenshots, partial views — doesn't matter).
    2. Detect each field's pixel position within that bounding box.
    3. Express each field as a fraction (0-1) of the form bounding box.
    4. Scale those fractions to the target paper's dimensions.

    This decouples the image (any size, any aspect ratio) from the
    physical output paper (always known: e.g., A4 or A5).

    Args:
      image_path: path to the form image (any size/format)
      template_path: optional path to template JSON (currently unused for coords)
      target_paper_width_mm: physical paper width (default 210mm = A4)
      target_paper_height_mm: physical paper height (default 297mm = A4)
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h_img, w_img = img.shape[:2]

    # Step 1: find the form bounding box on the image
    fbb_left, fbb_top, fbb_right, fbb_bot = find_form_bounding_box(img)
    fbb_w = fbb_right - fbb_left
    fbb_h = fbb_bot - fbb_top

    print(f"  Image: {w_img}x{h_img} px")
    print(f"  Form content area: ({fbb_left}, {fbb_top}) to ({fbb_right}, {fbb_bot})")
    print(f"  Form area size: {fbb_w}x{fbb_h} px")
    print(f"  Target paper: {target_paper_width_mm}x{target_paper_height_mm} mm")

    # Step 2: determine which mm-margins to leave around the form on the
    # target paper. We mirror the image's own margins proportionally:
    # if the form occupies 80% of the image, it'll occupy 80% of the
    # target paper, centered.
    image_margin_left_frac = fbb_left / w_img
    image_margin_right_frac = (w_img - fbb_right) / w_img
    image_margin_top_frac = fbb_top / h_img
    image_margin_bot_frac = (h_img - fbb_bot) / h_img
    # Target paper margins (mm): mirror the image's relative margins,
    # but clamp to a sensible minimum (10mm) so writing isn't on the very edge.
    target_margin_left_mm = max(10.0, image_margin_left_frac * target_paper_width_mm)
    target_margin_top_mm = max(10.0, image_margin_top_frac * target_paper_height_mm)
    target_form_width_mm = target_paper_width_mm - target_margin_left_mm \
        - max(10.0, image_margin_right_frac * target_paper_width_mm)
    target_form_height_mm = target_paper_height_mm - target_margin_top_mm \
        - max(10.0, image_margin_bot_frac * target_paper_height_mm)

    # Step 3: set up an OCRConfig whose px-to-mm conversion treats the
    # form bounding box as the reference frame.
    ocr_cfg = OCRConfig()
    ocr_cfg.dpi = 200   # placeholder (we'll override conversion below)

    # Run field detection in pixel space, then convert to target paper mm.
    ocr_fields_px = detect_fields_ocr_px(img, ocr_cfg)
    print(f"  OCR detected {len(ocr_fields_px)} field(s).")

    # Step 4: convert each field's pixel coords -> fraction of form bbox -> mm on target paper
    ocr_fields = []
    for f in ocr_fields_px:
        # Position within the form bounding box, as a fraction
        x_frac = (f['x_px'] - fbb_left) / max(1, fbb_w)
        y_frac = (f['y_px'] - fbb_top) / max(1, fbb_h)
        w_frac = f['w_px'] / max(1, fbb_w)
        h_frac = f['h_px'] / max(1, fbb_h)

        # Scale to mm on target paper
        x_mm = target_margin_left_mm + x_frac * target_form_width_mm
        y_mm = target_margin_top_mm + y_frac * target_form_height_mm
        width_mm = w_frac * target_form_width_mm
        height_mm = h_frac * target_form_height_mm

        # Clamp width to a reasonable maximum
        width_mm = min(width_mm, 100.0)
        height_mm = max(height_mm, 4.0)

        ocr_fields.append({
            'label': f['label'],
            'type': f['type'],
            'x_mm': x_mm,
            'y_mm': y_mm,
            'width_mm': width_mm,
            'height_mm': height_mm,
            'source': f['source'],
            'confidence': f['confidence'],
        })

    return merge_fields(ocr_fields, [])