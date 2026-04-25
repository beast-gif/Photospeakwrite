"""
G-code generator for the PhotoSpeakWrite robotic arm.

Produces standard CNC-style G-code:
  G21       millimetres
  G90       absolute coordinates
  G0 X.. Y.. F..   rapid travel (pen up assumed)
  G1 X.. Y.. F..   linear move (pen down for writing)
  G0 Z<travel>     pen up
  G1 Z<write>      pen down
  M5/M3            optional spindle off/on (commented; some pen plotters use this for servo)

The generator:
  - Iterates each character of the answer
  - Looks up its Hershey strokes
  - Scales font-units -> mm using char_height_mm / CAP_HEIGHT
  - Translates strokes to (x_start_mm + cursor_x, y_baseline_mm) in PAPER frame
  - Converts PAPER frame to ROBOT frame using PaperConfig.origin_*
  - Emits pen-up between strokes/glyphs and pen-down within a stroke
  - Wraps to next line if max_width_mm exceeded
"""

from typing import List
from .config import PaperConfig, PenConfig
from .hershey_font import get_glyph, CAP_HEIGHT


class GCodeGenerator:
    def __init__(self, paper_cfg: PaperConfig, pen_cfg: PenConfig):
        self.paper = paper_cfg
        self.pen = pen_cfg
        self._lines: List[str] = []
        self._pen_down = False

    # ---------- frame conversion ----------
    def _paper_to_robot(self, x_paper_mm: float, y_paper_mm: float):
        """
        Paper frame: origin at top-left of paper, X right, Y DOWN (image-style).
        Robot frame (this generator): X = origin_x + x_paper, Y = origin_y - y_paper.
        We flip Y so visually "down on the form" maps to decreasing Y in robot space,
        which matches how most pen-plotter setups are calibrated. Tweak if your
        robot's Y convention is reversed.
        """
        rx = self.paper.origin_x_mm + x_paper_mm
        ry = self.paper.origin_y_mm - y_paper_mm
        return rx, ry

    # ---------- low-level emit ----------
    def _emit(self, line: str):
        self._lines.append(line)

    def _pen_up(self):
        if self._pen_down:
            self._emit(f"G0 Z{self.pen.travel_z_mm:.3f} F{self.pen.z_feed:.0f}  ; pen up")
            self._pen_down = False

    def _pen_down_to_surface(self):
        if not self._pen_down:
            self._emit(f"G1 Z{self.pen.write_z_mm:.3f} F{self.pen.z_feed:.0f}  ; pen down")
            self._pen_down = True

    def _move_xy(self, x_mm: float, y_mm: float, write: bool):
        rx, ry = self._paper_to_robot(x_mm, y_mm)
        if write:
            self._emit(f"G1 X{rx:.3f} Y{ry:.3f} F{self.pen.write_feed:.0f}")
        else:
            self._emit(f"G0 X{rx:.3f} Y{ry:.3f} F{self.pen.travel_feed:.0f}")

    # ---------- public API ----------
    def begin(self):
        """Header: units, absolute mode, pen up at safe height."""
        self._emit("; PhotoSpeakWrite generated G-code")
        self._emit("; Units: millimetres, absolute positioning")
        self._emit("G21      ; mm")
        self._emit("G90      ; absolute")
        self._emit(f"G0 Z{self.pen.travel_z_mm:.3f} F{self.pen.z_feed:.0f}  ; raise pen")
        self._pen_down = False

    def end(self):
        """Footer: pen up, return home, stop."""
        self._pen_up()
        self._emit(f"G0 X0 Y0 F{self.pen.travel_feed:.0f}  ; return home")
        self._emit("M2       ; program end")

    def write_text(self, text: str, x_start_mm: float, y_baseline_mm: float,
                   char_height_mm: float = 4.0, max_width_mm: float = 60.0,
                   auto_shrink: bool = True, min_char_height_mm: float = 2.5):
        """
        Render `text` starting at (x_start_mm, y_baseline_mm) on the paper.

        If auto_shrink=True (default), the character height is reduced as
        needed to make the entire string fit within max_width_mm — the arm
        will write the whole answer on one line, just smaller. This is much
        more legible than wrapping mid-word, which is what humans expect
        when filling form fields.

        If shrinking would force chars below min_char_height_mm, falls back
        to wrapping.
        """
        if not text:
            return

        # First pass: compute total width at the requested char_height
        def total_width_at(h):
            scale = h / CAP_HEIGHT
            w = 0.0
            for ch in text:
                g = get_glyph(ch)
                w += (g["right"] - g["left"]) * scale + self.pen.char_spacing_mm
            return w - self.pen.char_spacing_mm   # no trailing space

        original_h = char_height_mm
        if auto_shrink:
            full_w = total_width_at(char_height_mm)
            if full_w > max_width_mm:
                # Scale char height down proportionally
                shrink_factor = max_width_mm / full_w
                char_height_mm = max(min_char_height_mm,
                                     char_height_mm * shrink_factor)
                if char_height_mm < original_h:
                    self._emit(
                        f"; (auto-shrink: '{text}' would be {full_w:.1f}mm "
                        f"@ {original_h}mm height; using {char_height_mm:.2f}mm)"
                    )

        scale = char_height_mm / CAP_HEIGHT
        cursor_x = 0.0
        cursor_y = 0.0

        self._emit(f"; ----- writing '{text}' at "
                   f"({x_start_mm:.2f}, {y_baseline_mm:.2f}) mm "
                   f"(h={char_height_mm:.2f}mm) -----")

        for ch in text:
            glyph = get_glyph(ch)
            glyph_w = (glyph["right"] - glyph["left"]) * scale
            advance = glyph_w + self.pen.char_spacing_mm

            # Wrap only if even after shrinking we don't fit
            if cursor_x + glyph_w > max_width_mm and cursor_x > 0:
                cursor_x = 0.0
                cursor_y += self.pen.line_spacing_mm
                self._pen_up()

            # Draw each stroke of the glyph
            for stroke in glyph["strokes"]:
                if not stroke:
                    continue
                # First point: travel with pen up
                fx, fy = stroke[0]
                px = x_start_mm + cursor_x + (fx - glyph["left"]) * scale
                py = y_baseline_mm + cursor_y - fy * scale

                self._pen_up()
                self._move_xy(px, py, write=False)
                self._pen_down_to_surface()

                # Subsequent points: write
                for (sx, sy) in stroke[1:]:
                    px = x_start_mm + cursor_x + (sx - glyph["left"]) * scale
                    py = y_baseline_mm + cursor_y - sy * scale
                    self._move_xy(px, py, write=True)

            self._pen_up()
            cursor_x += advance

        # Pen safely up after each field
        self._pen_up()

    # ---------- output ----------
    def to_string(self) -> str:
        return "\n".join(self._lines) + "\n"

    def save(self, path: str):
        with open(path, "w") as fp:
            fp.write(self.to_string())

    def line_count(self) -> int:
        return len(self._lines)