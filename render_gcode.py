"""
Parse the generated G-code and render it as an image.
This simulates what the robotic arm will actually draw on paper —
useful for verifying the output BEFORE running the real arm.
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_gcode(path: str):
    """Return list of (x, y, pen_down) tuples representing the path."""
    moves = []
    x, y, z = 0.0, 0.0, 5.0
    pen_down = False
    pat = re.compile(r"([XYZ])(-?\d+\.?\d*)")

    with open(path) as fp:
        for line in fp:
            line = line.split(";", 1)[0].strip()  # strip comments
            if not line:
                continue
            cmd = line.split()[0] if line.split() else ""
            if cmd not in ("G0", "G1"):
                continue
            for axis, val in pat.findall(line):
                v = float(val)
                if axis == "X": x = v
                elif axis == "Y": y = v
                elif axis == "Z": z = v
            pen_down = z <= 0.5   # pen is down when at or below paper
            moves.append((x, y, pen_down))
    return moves


def render(moves, out_png: str):
    fig, ax = plt.subplots(figsize=(10, 14))
    # draw segments where pen is down
    last = None
    for (x, y, pd) in moves:
        if last is not None and pd and last[2]:
            ax.plot([last[0], x], [last[1], y], "k-", linewidth=1.2)
        last = (x, y, pd)

    ax.set_aspect("equal")
    ax.set_title("Simulated paper output (what the arm will draw)")
    ax.set_xlabel("X (mm, robot frame)")
    ax.set_ylabel("Y (mm, robot frame)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    plt.close()
    print(f"Saved render -> {out_png}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/test_form.gcode"
    out = sys.argv[2] if len(sys.argv) > 2 else "output/test_form_render.png"
    moves = parse_gcode(path)
    print(f"Parsed {len(moves)} moves from {path}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    render(moves, out)
