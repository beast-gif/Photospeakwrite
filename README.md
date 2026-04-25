# PhotoSpeakWrite — Raspberry Pi 4B Implementation

End-to-end pipeline for the assistive form-writing robotic arm:

```
scan form -> detect blank fields (OCR + template) ->
ask user via voice -> capture answers (STT) ->
generate G-code -> stream to robotic arm
```

## Project layout

```
photospeakwrite/
├── main.py                    # Pipeline orchestrator
├── test_gcode.py              # Standalone G-code generator test (no OCR/STT)
├── render_gcode.py            # Visualize G-code output as PNG
├── requirements.txt
├── modules/
│   ├── config.py              # Paper, pen, OCR settings
│   ├── form_scanner.py        # OCR + template fallback
│   ├── voice_io.py            # TTS prompts + STT capture
│   ├── hershey_font.py        # Single-stroke font (uppercase + digits)
│   └── gcode_generator.py     # G0/G1 generator with pen up/down
├── templates/
│   └── standard_form.json     # Fallback field coordinates
└── output/
    ├── form_output.gcode
    └── answers.json
```

## Setup on Raspberry Pi 4B

```bash
# System packages
sudo apt update
sudo apt install -y python3-pip python3-venv \
                    tesseract-ocr libtesseract-dev \
                    espeak portaudio19-dev libatlas-base-dev

# Python env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Vosk offline model (~40 MB)
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

## Calibration (do this once)

The whole system depends on knowing where the paper sits in the robot's coordinate frame. Edit `modules/config.py`:

| Setting | What to do |
|---|---|
| `PaperConfig.origin_x_mm`, `origin_y_mm` | Jog the arm to the **top-left corner of the paper** with the pen tip touching. Read those (X,Y) off the controller. Put them here. |
| `PaperConfig.surface_z_mm` | The Z value where the pen just touches the paper. |
| `PenConfig.travel_z_mm` | A safe Z above paper for moves (~5 mm above surface_z). |
| `PenConfig.write_z_mm` | The Z to drop to while writing (= surface_z, or slightly below for spring-loaded pen holder). |
| `PenConfig.write_feed`, `travel_feed` | Tune for your arm's stable speed. Start slow (1500 / 3000 mm/min). |
| `OCRConfig.dpi` | DPI of your scanned form image. 200 DPI is the default. |

**Y-axis convention.** `gcode_generator._paper_to_robot()` flips Y so paper-down maps to robot-Y-down. If your arm uses the opposite convention, change the sign there.

## Running

### Dry run (no microphone needed) — produces full G-code with placeholder answers

```bash
python main.py --image my_form.jpg \
               --template templates/standard_form.json \
               --dry-run
```

### Full pipeline (with voice)

```bash
python main.py --image my_form.jpg \
               --template templates/standard_form.json \
               --out-gcode output/filled_form.gcode
```

The system will:
1. Detect each blank field (e.g., "name:", "phone no:", "age:")
2. Speak: "Please say your name."
3. Listen, transcribe, validate, reprompt up to 3× if confidence is low
4. Write the G-code at the correct (X, Y) for that field's blank
5. Repeat for all fields, then save the combined G-code

### Verify before running on the real arm

```bash
python render_gcode.py output/filled_form.gcode output/preview.png
```

This parses the G-code and renders a PNG of exactly what the pen will draw — same coordinates, same stroke order. **Do this every time** before sending G-code to the arm; it's free and catches calibration mistakes before they break a pen or scratch the paper.

## Streaming G-code to the arm

The generator emits **standard G-code** so any of these will work:

- **GRBL controller** (Arduino-based, common for cheap pen plotters): use `grbl-streamer` or `bCNC`.
- **3D printer firmware** (Marlin/Klipper) on a converted 3D printer: send via OctoPrint or `pyserial`.
- **RoboDK arm** (your sim): RoboDK has a native G-code post-processor; `Program -> Add G-code Program` reads the file and converts MoveL commands. See `RoboDK -> File -> Open -> *.gcode`.
- **Custom 4-DOF arm**: write a small `pyserial` streamer that reads each line and sends it to your microcontroller.

A minimal streamer for GRBL/Marlin:

```python
import serial, time
ser = serial.Serial('/dev/ttyUSB0', 115200)
time.sleep(2)
with open('output/filled_form.gcode') as fp:
    for line in fp:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        ser.write((line + '\n').encode())
        # wait for "ok" before sending next line
        while True:
            r = ser.readline().decode().strip()
            if r.startswith('ok') or r.startswith('error'):
                break
```

## How the modules fit together

| Module | Inputs | Outputs |
|---|---|---|
| `form_scanner.scan_form` | image path, optional template | list of `{label, x_mm, y_mm, width_mm, height_mm, type}` |
| `voice_io.VoiceIO.ask_and_capture` | label, expected_type | string answer |
| `gcode_generator.GCodeGenerator.write_text` | text, x_mm, y_mm, char_height | appends G-code lines |

Each layer is independently swappable: replace Tesseract with a YOLO field detector, swap Vosk for Whisper, or change Hershey font for a Bezier glyph library — the contracts between layers stay the same.

## Known limitations (matches what's discussed in the paper's conclusion)

- Hershey font is single-stroke and uppercase-leaning; lowercase looks blocky. For more realistic handwriting, swap `hershey_font.py` for a Bezier-curve glyph library.
- The OCR-detected blank assumes the field is to the right of the label and runs to the right margin. Wide multi-column forms need the template fallback.
- No closed-loop feedback — if the arm misses a stroke (slipping pen, paper shift), there is no correction. A camera-in-the-loop system would solve this; future work.
