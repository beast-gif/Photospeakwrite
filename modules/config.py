"""
Central configuration. Tweak these to match your physical setup.
All dimensions in millimetres unless noted.
"""

from dataclasses import dataclass


@dataclass
class PaperConfig:
    # A4 portrait by default
    width_mm: float = 210.0
    height_mm: float = 297.0
    # Origin offset: where (0,0) of paper sits in the robot's base frame.
    # Calibrate this by jogging the arm to the top-left corner of the paper.
    origin_x_mm: float = 0.0
    origin_y_mm: float = 0.0
    # Paper Z height in robot frame (the surface where the pen tip lands).
    surface_z_mm: float = 0.0


@dataclass
class PenConfig:
    # Z lift above paper for travel moves (pen up)
    travel_z_mm: float = 5.0
    # Z dip below the surface for firm contact (set 0 if your pen is spring-loaded)
    write_z_mm: float = 0.0
    # Feedrates (mm/min)
    travel_feed: float = 3000.0   # G0 / fast moves with pen up
    write_feed: float = 1500.0    # G1 / writing moves with pen down
    z_feed: float = 1000.0        # pen up/down moves
    # Glyph stroke spacing
    char_spacing_mm: float = 1.5  # gap between characters
    line_spacing_mm: float = 8.0  # gap between lines if text wraps


@dataclass
class OCRConfig:
    # Pixel-to-mm scale for the form image. If unknown, set via DPI.
    # 200 DPI -> 1 mm = 200/25.4 ≈ 7.874 px
    dpi: float = 200.0

    @property
    def px_per_mm(self) -> float:
        return self.dpi / 25.4

    def px_to_mm(self, px: float) -> float:
        return px / self.px_per_mm


# Field labels -> expected input type (used by STT for constrained recognition)
FIELD_TYPE_HINTS = {
    # Personal
    "name": "text",
    "first name": "text",
    "last name": "text",
    "father's name": "text",
    "mother's name": "text",
    "age": "digits",
    "gender": "choice:male,female,other",
    "date": "date",
    "dob": "date",
    "date of birth": "date",
    # Location
    "place": "text",
    "city": "text",
    "state": "text",
    "country": "text",
    "address": "text",
    "pin code": "digits",
    "zip code": "digits",
    "postal code": "digits",
    # Contact
    "phone no": "digits",
    "phone number": "digits",
    "mobile": "digits",
    "mobile number": "digits",
    "email": "email",
    # Identity / IDs (all numeric)
    "aadhar number": "digits",
    "aadhaar number": "digits",
    "aadhar": "digits",
    "aadhaar": "digits",
    "passport number": "text",        # alphanumeric
    "passport": "text",
    "pan number": "text",             # alphanumeric (e.g. ABCDE1234F)
    "pan": "text",
    "voter id": "text",
    "license number": "text",
    "driving license": "text",
    # Education / work
    "occupation": "text",
    "qualification": "text",
    "company": "text",
    "designation": "text",
}


def get_field_type(label: str) -> str:
    return FIELD_TYPE_HINTS.get(label.strip().lower(), "text")