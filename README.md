# PhotoSpeakWrite

**A multimodal assistive robotic writing system: it reads a paper form, asks the user for their answers by voice, and writes them on the form with a robotic arm.**

PhotoSpeakWrite is for people with motor impairments who can't fill in handwritten forms themselves. A camera image of the form goes in. The system finds the blank fields with OCR, reads each question aloud, transcribes the spoken answer, and generates G-code so a 4-DOF robotic arm writes each answer in the right place.

> 📄 This repository is the software implementation behind our published Q1 IEEE journal paper, *PhotoSpeakWrite*.

```
 ┌────────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────┐   ┌───────────┐
 │ Form image │ → │ OCR + field  │ → │ TTS asks  │ → │ STT gets   │ → │ G-code →  │
 │ (photo /   │   │ detection    │   │ question  │   │ answer +   │   │ robotic   │
 │ screenshot)│   │ (Tesseract)  │   │ (pyttsx3) │   │ validation │   │ arm writes│
 └────────────┘   └──────────────┘   └───────────┘   └────────────┘   └───────────┘
```

Runs fully **offline** on a **Raspberry Pi 4B** (Tesseract + Vosk + eSpeak), and also on Windows for development.

---

## Features

- **Form understanding**
  - OpenCV preprocessing: bilateral denoise, adaptive threshold, deskew.
  - Tesseract OCR with word-level bounding boxes to find labels such as `Name:`, `Age:` and `Phone No:`.
  - **OCR alias map** for common misreads (`dos → dob`, `cit → city`, `ace → age`, `gender; → gender`).
  - **Horizontal-line detection**, so the pen writes on the actual underline instead of guessing from the label's position.
  - **Form bounding-box auto-calibration.** Coordinates are stored as fractions of the detected form region, so the same form works whether the input is a tight screenshot or a phone photo with background around it.
  - Fields are ordered top to bottom (`y_mm`). Arm coordinates are absolute, so the writing order doesn't affect accuracy.
- **Voice interaction**
  - TTS through `pyttsx3` (eSpeak on Pi). The engine is reinitialised on every call, which fixes the Windows `runAndWait()` hang.
  - STT fallback chain: **Vosk (offline)** → **SpeechRecognition / Google (online)** → **keyboard input**.
  - Type-aware post-processing: spoken digits become numbers ("one two three four" → `1234`), gender is limited to Male/Female/Other, and emails handle "at"/"dot".
  - Up to 3 re-prompts per field when confidence is low.
- **Writing**
  - Built-in **Hershey single-stroke font** (A–Z, 0–9, common punctuation, `@`).
  - Standard G-code: `G21`/`G90`, `G0` travel, `G1` writing, and Z-axis pen lifts.
  - **Auto-shrink**: long answers are scaled down to fit inside the field width instead of wrapping onto the next line.
  - Paper-to-robot frame conversion happens in one function, so calibration only needs changing in one place.
- **Tooling**
  - `debug_form.py` draws detected labels, lines and write positions on the form image.
  - `render_gcode.py` renders G-code to a PNG so you can check exactly what the pen will draw.
  - `make_sample_form.py` generates a clean test form.

---

## Project layout

```
PhotoSpeakWrite/
├── main.py                  # Five-step pipeline orchestrator
├── debug_form.py            # Diagnostic overlay of detected fields
├── render_gcode.py          # G-code → PNG preview
├── make_sample_form.py      # Generates a sample test form
├── test_gcode.py            # G-code generator test (no OCR/STT needed)
├── requirements.txt
├── modules/
│   ├── config.py            # PaperConfig, PenConfig, FIELD_TYPE_HINTS, KNOWN_LABELS
│   ├── form_scanner.py      # Preprocessing, OCR, alias map, line + form-bbox detection
│   ├── voice_io.py          # TTS + Vosk / SpeechRecognition / keyboard STT
│   ├── hershey_font.py      # Single-stroke glyph library
│   └── gcode_generator.py   # G0/G1 + Z-lift generator with auto-shrink
├── models/
│   └── vosk-model-small-en-us-0.15/   # Not committed (download it, see below)
└── output/                  # Generated G-code, answers.json, previews (git-ignored)
```

---

## Installation

### Raspberry Pi 4B (Raspberry Pi OS Bookworm)

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv \
    tesseract-ocr libtesseract-dev \
    espeak espeak-ng \
    portaudio19-dev libasound2-dev \
    libgl1 libglib2.0-0
```

> ⚠️ Leave out `libatlas-base-dev`. Bookworm doesn't have it, and the install fails if you include it.

```bash
cd ~/Desktop/rasp            # project folder on the Pi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Moving the project from Windows:** copy the folder over by USB, but **leave out `.venv/` and `output/`**. A Windows venv contains `.exe` binaries that won't run on the Pi, so build a fresh one there.

### Windows (development)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) and add it to `PATH`. If you set `pytesseract.pytesseract.tesseract_cmd` to a Windows path, remove that line before deploying to the Pi.

### Vosk speech model (~40 MB)

```bash
mkdir -p models && cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip && rm vosk-model-small-en-us-0.15.zip
cd ..
```

If the Pi has no internet access, download the zip on another machine, copy it over by USB and unzip it into `models/`. The code expects `models/vosk-model-small-en-us-0.15/`.

> For better recognition of Indian names, `vosk-model-small-en-in-0.4` is a similar-sized alternative. Update `VOSK_MODEL_PATH` in `voice_io.py` if you switch.

---

## Audio setup on the Pi

These were the settings needed to get a USB mic and headphones working on the Pi 4B.

**1. Sample rate.** The USB mic only supports 48 kHz. In `modules/voice_io.py`:

```python
SAMPLE_RATE = 48000      # was 16000; 16 kHz fails on most USB mics
```

**2. Input device.** Set `sounddevice` to use the USB mic explicitly:

```python
_sd_default.default.device = (1, None)   # (input_index, output_index)
```

Find your index with `python -c "import sounddevice as sd; print(sd.query_devices())"`.

**3. Output device (TTS through headphones).** Create `~/.asoundrc`:

```
pcm.!default {
    type plug
    slave.pcm "hw:2,0"
}
ctl.!default {
    type hw
    card 2
}
```

Check your card number with `aplay -l`, and test it with `espeak "hello"`.

**Quick STT check:** say "one two three four". The log should show `1234` with `conf=1.00`.

---

## Calibration (once per physical setup)

Everything depends on knowing where the paper sits in the arm's coordinate frame. Edit `modules/config.py`:

| Setting | How to set it |
|---|---|
| `PaperConfig.origin_x_mm`, `origin_y_mm` | Jog the arm so the pen tip touches the **top-left corner of the paper**, then copy the X/Y from the controller. |
| `PaperConfig.surface_z_mm` | The Z where the pen just touches the paper. |
| `PenConfig.travel_z_mm` | Safe pen-up height, about 5 mm above the surface. |
| `PenConfig.write_z_mm` | Pen-down Z: the same as the surface, or slightly below for a spring-loaded holder. |
| `PenConfig.write_feed` / `travel_feed` | Start slow (1500 / 3000 mm/min) and increase once the strokes look stable. |

**Y-axis convention:** `GCodeGenerator._paper_to_robot()` flips Y so that moving down the page maps to −Y on the robot. If your arm uses the opposite convention, flip the sign there.

---

## Usage

### 1. Check field detection

```bash
python debug_form.py
```

Writes an overlay image showing each detected label, its underline and the computed write position. Fix detection problems here before involving the arm.

### 2. Dry run (no microphone; uses placeholder answers)

```bash
python main.py --image inputs/form.jpg --dry-run
```

### 3. Full run with voice

```bash
python main.py --image inputs/form.jpg --out-gcode output/filled_form.gcode
```

For each field, the system will:
1. Say "Please say your name."
2. Listen, transcribe and clean up the answer for the field type.
3. Read the answer back, or re-prompt (up to 3 times) if confidence is low.
4. Save all answers to `output/answers.json` and the combined G-code to `output/filled_form.gcode`.

### 4. Preview before writing

```bash
python render_gcode.py output/filled_form.gcode output/preview.png
```

**Do this every time** before streaming to the arm. It costs nothing and catches calibration mistakes before they ruin a form or break a pen.

### 5. Stream to the arm

The output is standard G-code, so it works with GRBL, Marlin/Klipper (OctoPrint), RoboDK (`File → Open → *.gcode`), or a custom microcontroller. Minimal `pyserial` streamer:

```python
import serial, time
ser = serial.Serial('/dev/ttyUSB0', 115200); time.sleep(2)
for line in open('output/filled_form.gcode'):
    line = line.split(';')[0].strip()
    if not line:
        continue
    ser.write((line + '\n').encode())
    while not ser.readline().decode().strip().startswith(('ok', 'error')):
        pass
```

---

## Module contracts

| Module | Input | Output |
|---|---|---|
| `form_scanner.scan_form` | image path | `[{label, type, x_mm, y_mm, width_mm, height_mm, source, confidence}]` |
| `voice_io.VoiceIO.ask_and_capture` | label, expected type | cleaned answer string |
| `gcode_generator.GCodeGenerator.write_text` | text, x, y, char height, max width | G-code lines (auto-shrunk to fit) |

Each layer can be swapped on its own. For example, you could use a YOLO field detector instead of Tesseract, Whisper instead of Vosk, or a Bézier glyph set instead of Hershey, without changing the other layers.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `apt` fails on `libatlas-base-dev` | Remove it from the install line (it isn't available on Bookworm). |
| `Invalid sample rate` / no audio from mic | Set `SAMPLE_RATE = 48000` and pin the input device index. |
| TTS prints but there's no sound | Point `~/.asoundrc` at the right `hw:X,0` (check with `aplay -l`). |
| TTS hangs after the first prompt (Windows) | Already handled: the engine is reinitialised per call. Make sure you're on the latest `voice_io.py`. |
| A label is missed (e.g. "DOB" read as "dos") | Add the misread to the OCR alias map in `form_scanner.py`. |
| Text lands above or beside the line | Run `debug_form.py` and check that the underline and form bounding box were detected. |
| Vosk model not found | The path must be exactly `models/vosk-model-small-en-us-0.15/`, relative to where you run `main.py`. |

---

## Limitations and future work

- Hershey glyphs are single-stroke and look blocky for lowercase letters. A Bézier-curve glyph set would look more natural.
- Writing is open-loop: if the pen slips or the paper shifts, nothing corrects it. A camera in the loop would fix this.
- Field detection assumes the blank is to the right of its label. Multi-column or tabular forms need more layout logic.

---

## Authors

Built by **Sai Amrith Reddy** ([GitHub](https://github.com/beast-gif) · [Portfolio](https://amrithdev.vercel.app)) and co-authors, Electronics and Computer Engineering, Amrita Vishwa Vidyapeetham, Bengaluru.

If you use this work, please cite the PhotoSpeakWrite paper (IEEE).
