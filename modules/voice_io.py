"""
Voice I/O for Raspberry Pi 4B.

- TTS: pyttsx3 (offline, works on Pi via espeak)
- STT: Vosk (offline, lightweight) preferred; falls back to SpeechRecognition+Google.

Each field gets up to 3 listening attempts before accepting whatever was heard.
"""

import json
import queue
import re
import sys
import time

# --- Optional imports, gracefully degrade ---
try:
    import pyttsx3
    HAVE_TTS = True
except (ImportError, OSError):
    HAVE_TTS = False

try:
    import sounddevice as sd
    import vosk
    HAVE_VOSK = True
except (ImportError, OSError):
    HAVE_VOSK = False

try:
    import speech_recognition as sr
    HAVE_SR = True
except (ImportError, OSError):
    HAVE_SR = False


VOSK_MODEL_PATH = "models/vosk-model-small-en-us-0.15"   # ~40 MB, Pi-friendly
SAMPLE_RATE = 16000
MAX_ATTEMPTS = 3


class VoiceIO:
    def __init__(self):
        # ---- TTS ----
        # We reinitialize per call (Windows pyttsx3 bug workaround), so just
        # do a quick check that init works.
        if HAVE_TTS:
            try:
                e = pyttsx3.init()
                e.stop()
                del e
                print("  TTS backend: pyttsx3 (reinitialized per call)")
            except Exception as e:
                print(f"  [WARN] TTS init test failed: {e}")

        # ---- STT init ----
        self.stt_backend = None
        self.vosk_model = None
        self.recognizer = None

        if HAVE_VOSK:
            import os
            if os.path.isdir(VOSK_MODEL_PATH):
                try:
                    self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
                    self.stt_backend = "vosk"
                    print("  STT backend: Vosk (offline)")
                except Exception as e:
                    print(f"  [WARN] Vosk model load failed: {e}")
            else:
                print(f"  [INFO] Vosk model not found at '{VOSK_MODEL_PATH}'.")
                print(f"         Download it (40 MB) from:")
                print(f"         https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
                print(f"         Unzip into ./models/  so you have ./{VOSK_MODEL_PATH}/")

        if self.stt_backend is None and HAVE_SR:
            self.recognizer = sr.Recognizer()
            self.stt_backend = "sr"
            print("  STT backend: SpeechRecognition (online, needs internet)")

        if self.stt_backend is None:
            print("  [WARN] No STT backend available. Will use stdin fallback.")
            self.stt_backend = "stdin"

    # ---------- TTS ----------
    def speak(self, text: str):
        """
        Speak text aloud.

        NOTE: pyttsx3 has a long-standing bug on Windows where the engine's
        run loop gets stuck after the first runAndWait(), causing subsequent
        speak() calls to hang silently. The fix is to reinitialize the engine
        per call. This is slightly slower but rock-solid.
        """
        print(f"  [TTS] {text}")
        if not HAVE_TTS:
            return
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"  [WARN] TTS speak failed: {e}")

    # ---------- STT ----------
    def _listen_vosk(self, timeout_s: float = 6.0) -> tuple:
        """Return (text, confidence_estimate_0_to_1)."""
        q = queue.Queue()

        def callback(indata, frames, time_info, status):
            q.put(bytes(indata))

        rec = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
        rec.SetWords(True)

        text = ""
        avg_conf = 0.0
        start = time.time()
        with sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=8000,
                                dtype='int16', channels=1, callback=callback):
            while time.time() - start < timeout_s:
                try:
                    data = q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text = res.get("text", "")
                    words = res.get("result", [])
                    if words:
                        avg_conf = sum(w.get("conf", 0) for w in words) / len(words)
                    if text:
                        break
            else:
                # timeout: take whatever partial we have
                res = json.loads(rec.FinalResult())
                text = res.get("text", "")
                words = res.get("result", [])
                if words:
                    avg_conf = sum(w.get("conf", 0) for w in words) / len(words)
        return text.strip(), avg_conf

    def _listen_sr(self, timeout_s: float = 8.0) -> tuple:
        """
        Listen via SpeechRecognition + Google API.
        timeout_s = max seconds to wait for speech to START
        phrase_time_limit caps how long a single utterance can be.
        """
        try:
            mic = sr.Microphone(sample_rate=SAMPLE_RATE)
        except Exception as e:
            print(f"  [WARN] Could not open microphone: {e}")
            return "", 0.0

        with mic as source:
            # Quick ambient calibration on first use; shorter on subsequent calls
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            self.recognizer.pause_threshold = 0.8   # seconds of silence = end of phrase
            print(f"  [Speak now...]")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout_s,
                    phrase_time_limit=8.0,
                )
            except sr.WaitTimeoutError:
                print(f"  [No speech detected within {timeout_s}s]")
                return "", 0.0
            except Exception as e:
                print(f"  [WARN] Listen failed: {e}")
                return "", 0.0

        print("  [Got audio, transcribing...]")
        try:
            text = self.recognizer.recognize_google(audio)
            return text.strip(), 0.8
        except sr.UnknownValueError:
            print("  [Could not understand audio]")
            return "", 0.0
        except sr.RequestError as e:
            print(f"  [WARN] Google API error: {e}")
            return "", 0.0

    def _listen_stdin(self) -> tuple:
        """Manual fallback when no STT is available."""
        try:
            text = input("  [stdin] Type the answer: ").strip()
            return text, 1.0 if text else 0.0
        except EOFError:
            return "", 0.0

    def listen(self) -> tuple:
        if self.stt_backend == "vosk":
            return self._listen_vosk()
        elif self.stt_backend == "sr":
            return self._listen_sr()
        else:
            return self._listen_stdin()

    # ---------- Per-field interaction ----------
    def ask_and_capture(self, label: str, expected_type: str = "text") -> str:
        prompt = self._build_prompt(label)
        for attempt in range(1, MAX_ATTEMPTS + 1):
            self.speak(prompt if attempt == 1
                       else f"Sorry, I didn't catch that. {prompt}")
            print(f"  [{attempt}/{MAX_ATTEMPTS}] Listening for '{label}'...")
            text, conf = self.listen()

            if not text:
                print(f"  [Heard nothing on attempt {attempt}]")
                continue

            cleaned = self._postprocess(text, expected_type)
            print(f"  [Heard: '{text}' -> cleaned: '{cleaned}', conf={conf:.2f}]")

            if cleaned and conf >= 0.4:
                self.speak(f"I heard {cleaned}.")
                return cleaned
            elif cleaned and attempt == MAX_ATTEMPTS:
                # last attempt: accept whatever we have
                return cleaned

        # All attempts failed
        return ""

    def _build_prompt(self, label: str) -> str:
        templates = {
            "name": "Please say your name.",
            "first name": "Please say your first name.",
            "last name": "Please say your last name.",
            "father's name": "Please say your father's name.",
            "mother's name": "Please say your mother's name.",
            "place": "Please say your place or city.",
            "city": "Please say your city.",
            "state": "Please say your state.",
            "country": "Please say your country.",
            "address": "Please say your address.",
            "pin code": "Please say your pin code, digit by digit.",
            "zip code": "Please say your zip code, digit by digit.",
            "postal code": "Please say your postal code, digit by digit.",
            "phone no": "Please say your phone number, digit by digit.",
            "phone number": "Please say your phone number, digit by digit.",
            "mobile": "Please say your mobile number.",
            "mobile number": "Please say your mobile number.",
            "age": "Please say your age.",
            "gender": "Please say your gender. Male, female, or other.",
            "email": "Please say your email address.",
            "date": "Please say the date.",
            "dob": "Please say your date of birth.",
            "date of birth": "Please say your date of birth.",
            "aadhar number": "Please say your Aadhar number, digit by digit.",
            "aadhaar number": "Please say your Aadhaar number, digit by digit.",
            "aadhar": "Please say your Aadhar number, digit by digit.",
            "aadhaar": "Please say your Aadhaar number, digit by digit.",
            "passport number": "Please say your passport number.",
            "passport": "Please say your passport number.",
            "pan number": "Please say your PAN number, character by character.",
            "pan": "Please say your PAN number, character by character.",
            "voter id": "Please say your voter ID.",
            "license number": "Please say your license number.",
            "driving license": "Please say your driving license number.",
            "occupation": "Please say your occupation.",
            "qualification": "Please say your highest qualification.",
            "company": "Please say your company name.",
            "designation": "Please say your designation.",
        }
        return templates.get(label.lower(), f"Please say your {label}.")

    def _postprocess(self, text: str, expected_type: str) -> str:
        """Constrain output to expected type (digits, choice, etc.)."""
        text = text.strip()

        if expected_type == "digits":
            # Convert spelled-out numbers into digits where possible
            mapping = {
                "zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3",
                "four": "4", "five": "5", "six": "6", "seven": "7",
                "eight": "8", "nine": "9", "ten": "10",
            }
            tokens = text.lower().split()
            result = []
            for t in tokens:
                if t in mapping:
                    result.append(mapping[t])
                elif t.isdigit():
                    result.append(t)
            joined = "".join(result) if result else re.sub(r'\D', '', text)
            return joined

        if expected_type.startswith("choice:"):
            options = expected_type.split(":", 1)[1].split(",")
            tl = text.lower()
            for opt in options:
                if opt.strip().lower() in tl:
                    return opt.strip().capitalize()
            return text.capitalize()

        if expected_type == "email":
            t = text.lower().replace(" at ", "@").replace(" dot ", ".").replace(" ", "")
            return t

        # Default text: title case, single spaces
        return " ".join(text.split()).title()

    def close(self):
        # Nothing to close — TTS engine is reinitialized per call,
        # mic resources are released by the `with` block in _listen_sr.
        pass