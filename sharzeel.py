"""
╔══════════════════════════════════════════════════════════════════╗
║          ✈  INDORE AIRPORT ASSISTANT ROBOT  v3.1                ║
║          Built by Acropolis College, Indore                      ║
╠══════════════════════════════════════════════════════════════════╣
║  STT   → Groq Whisper (whisper-large-v3)                        ║
║  LLM   → Groq LLaMA 3.3 70B                                     ║
║  TTS   → Microsoft Edge TTS (edge-tts, 100% free)               ║
║  UI    → Full-screen new.jpg (Tkinter + Pillow)                 ║
║  GPIO  → Raspberry Pi LED status indicator                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import asyncio
import tempfile
import queue
import time
import threading
import re
import logging
from enum import Enum
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import soundfile as sf
from groq import Groq
import edge_tts
import pygame
from dotenv import load_dotenv

try:
    from PIL import Image, ImageTk
    import tkinter as tk
    HAS_DISPLAY = True
except ImportError:
    HAS_DISPLAY = False
    print("⚠️  Pillow / tkinter not found — image display disabled.")

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("ℹ️  RPi.GPIO not found — running without GPIO (dev mode).")

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("AirportBot")

# ─────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    sys.exit("❌  GROQ_API_KEY not set. Add it to your .env file.")

STT_MODEL        = "whisper-large-v3"
CHAT_MODEL       = "llama-3.3-70b-versatile"

TTS_VOICE_EN     = "en-IN-NeerjaNeural"
TTS_VOICE_HI     = "hi-IN-SwaraNeural"

SAMPLE_RATE      = 16_000
CHANNELS         = 1
MAX_TOKENS       = 350
CHAT_TEMPERATURE = 0.7

ENERGY_THRESHOLD     = 0.010
SILENCE_AFTER_SPEECH = 0.8
PRE_ROLL_CHUNKS      = 6
MIN_SPEECH_SECS      = 0.5
CHUNK_SECS           = 0.1

IDLE_TIMEOUT         = 10.0
IDLE_POLL_TIMEOUT    = 60.0

GREEN_LED_PIN = 18
STT_MODEL       = "whisper-large-v3"        # used for real conversation (accurate)
STT_MODEL_FAST  = "whisper-large-v3-turbo"  # used only for wake word (3x faster)

WAKE_WORDS = ["hello", "hey", "hello acrobot", "hey acrobot", "acrobot", "airport"]

# Place new.jpg in the same folder as this script
IMAGE_PATH = "/home/admin123/Downloads/new.jpeg"

# ─────────────────────────────────────────────────────────────────
#  SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────

SYSTEM_EN = """\
You are AirBot, the official AI assistant robot at Devi Ahilyabai Holkar Airport, Indore.
You were built by the students of Acropolis College of Engineering and Technology, Indore.
the contact number of indore airport's terminal manager is nine four two five zero five seven seven one six


Your responsibilities:
- Welcome travelers warmly and professionally.
- Provide accurate guidance on flights, check-in counters, gates, baggage, lounges, and airport facilities.
- Answer questions about Indore: Rajwada Palace, 56 Dukan, Sarafa Bazaar, Lal Bagh Palace,
  Chokhi Dhani, Patalpani Waterfall, and Tincha Falls.
- Share practical travel tips.

Rules:
- Keep replies under 3 sentences — this is a kiosk environment.
- No bullet points, no markdown, no emoji in spoken replies.
- Never make up flight data — direct the traveler to the airline counter if unsure.
- Never discuss anything unrelated to travel, Indore, or the airport.
"""

SYSTEM_HI = """\
Aap AirBot hain — Devi Ahilyabai Holkar Airport, Indore ke official AI assistant robot.
Aapko Acropolis College of Engineering and Technology, Indore ke students ne banaya hai.
indore airport ke terminal manager ka number nine four two five zero five seven seven one six

Aapki zimmedariyan:
- Yatriyon ka aadar se swagat karein.
- Flights, check-in, gates, baggage, lounges aur airport suvidhaon ki jaankari dein.
- Indore ke prasidh sthalon ke baare mein bataayein: Rajwada, 56 Dukan, Sarafa,
  Lal Bagh Palace, Chokhi Dhani, Patalpani, Tincha Falls.

Niyam:
- Roman/Latin script mein jawab dein — Devanagari script bilkul mat use karein.
- 3 se zyada sentence mat bolein.
- Koi bullet points, markdown, ya emojis nahi.
- Agar flight data clear na ho toh airline counter pe bhejein.
"""

# ─────────────────────────────────────────────────────────────────
#  STATIC QA PAIRS
# ─────────────────────────────────────────────────────────────────

QA_EN = {
    "wifi":           "Free Wi-Fi is available throughout the airport. Connect to AAI_FREE_WIFI.",
    "washroom":       "Restrooms are located on every floor near the departure gates.",
    "atm":            "ATMs are available in the departure and arrival areas.",
    "food":           "Restaurants and food courts are on the first floor of the terminal.",
    "taxi":           "Pre-paid taxi counters are just outside the arrival exit.",
    "parking":        "Multi-level parking is available right in front of the terminal.",
    "lounge":         "The VIP lounge is on the second floor. Ask airline staff for access cards.",
    "baggage":        "Baggage claim is on the ground floor of the arrival hall.",
    "lost and found": "Please visit the Airport Authority of India helpdesk near gate 3.",
    "emergency":      "For any emergency, dial 112 or contact the nearest CISF officer.",
}

QA_HI = {
    "wifi":     "Airport mein free Wi-Fi available hai. AAI_FREE_WIFI se connect karein.",
    "washroom": "Washrooms har floor par departure gate ke paas hain.",
    "khana":    "Restaurant aur food court terminal ki pehli manzil par hain.",
    "taxi":     "Pre-paid taxi counter arrival exit ke bahaar hai.",
    "parking":  "Multi-level parking terminal ke saamne hai.",
}

# ─────────────────────────────────────────────────────────────────
#  STATE
# ─────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE      = "idle"
    LISTENING = "listening"
    SPEAKING  = "speaking"
    THINKING  = "thinking"

# ─────────────────────────────────────────────────────────────────
#  GPIO
# ─────────────────────────────────────────────────────────────────

def gpio_setup():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GREEN_LED_PIN, GPIO.OUT)
    GPIO.output(GREEN_LED_PIN, GPIO.LOW)

def gpio_set(on: bool):
    if GPIO_AVAILABLE:
        GPIO.output(GREEN_LED_PIN, GPIO.HIGH if on else GPIO.LOW)

def gpio_cleanup():
    if GPIO_AVAILABLE:
        GPIO.cleanup()

# ─────────────────────────────────────────────────────────────────
#  GROQ CLIENT
# ─────────────────────────────────────────────────────────────────

client   = Groq(api_key=GROQ_API_KEY)
_history = {"en": [], "hi": []}

# ─────────────────────────────────────────────────────────────────
#  VAD RECORDING
# ─────────────────────────────────────────────────────────────────

def capture_speech(timeout: float) -> Optional[np.ndarray]:
    audio_q   = queue.Queue()
    blocksize = int(SAMPLE_RATE * CHUNK_SECS)

    def callback(indata, frames, time_info, status):
        audio_q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        dtype="float32", blocksize=blocksize, callback=callback,
    )
    stream.start()
    gpio_set(True)

    speech_buffer: list            = []
    pre_buffer:    list            = []
    recording                      = False
    silence_start: Optional[float] = None
    idle_clock                     = time.time()

    try:
        while True:
            try:
                chunk = audio_q.get(timeout=0.5)
            except queue.Empty:
                if not recording and time.time() - idle_clock >= timeout:
                    return None
                continue

            rms = float(np.sqrt(np.mean(chunk ** 2)))

            if rms >= ENERGY_THRESHOLD:
                idle_clock = time.time()
                silence_start = None
                if not recording:
                    recording = True
                    speech_buffer = list(pre_buffer)
                speech_buffer.append(chunk)

            elif recording:
                speech_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start >= SILENCE_AFTER_SPEECH:
                    break

            else:
                pre_buffer.append(chunk)
                if len(pre_buffer) > PRE_ROLL_CHUNKS:
                    pre_buffer.pop(0)
                if time.time() - idle_clock >= timeout:
                    return None

    finally:
        stream.stop()
        stream.close()
        gpio_set(False)

    if not speech_buffer:
        return None

    audio = np.concatenate(speech_buffer, axis=0)
    return audio if len(audio) >= SAMPLE_RATE * MIN_SPEECH_SECS else None

# ─────────────────────────────────────────────────────────────────
#  TRANSCRIBE
# ─────────────────────────────────────────────────────────────────

def transcribe(audio: np.ndarray) -> Tuple[str, str]:
    """Full quality transcription — used during conversation."""
    return _transcribe_with_model(audio, STT_MODEL)

def transcribe_fast(audio: np.ndarray) -> Tuple[str, str]:
    """Faster transcription — used only for wake word detection."""
    return _transcribe_with_model(audio, STT_MODEL_FAST)

def _transcribe_with_model(audio: np.ndarray, model: str) -> Tuple[str, str]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    sf.write(tmp_path, audio, SAMPLE_RATE)

    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model=model, file=f, response_format="verbose_json",
            )
    finally:
        os.unlink(tmp_path)

    text = (result.text or "").strip()
    lang = (result.language or "en").strip().lower()

    if lang in ("ur", "ur-PK"):
        lang = "hi"
    if lang not in ("hi", "en"):
        lang = "en"

    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            lang = "hi"; break
        if 0x0600 <= cp <= 0x06FF:
            lang = "hi"; break

    return text, lang

# ─────────────────────────────────────────────────────────────────
#  WAKE WORD
# ─────────────────────────────────────────────────────────────────

def is_wake_word(text: str) -> bool:
    lower = text.lower().strip()
    return any(w in lower for w in WAKE_WORDS)

# ─────────────────────────────────────────────────────────────────
#  STATIC QA
# ─────────────────────────────────────────────────────────────────

def static_answer(text: str, lang: str) -> Optional[str]:
    lower = text.lower()
    table = QA_HI if lang == "hi" else QA_EN
    for key, answer in table.items():
        if key in lower:
            return answer
    return None

# ─────────────────────────────────────────────────────────────────
#  LLM REPLY
# ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"[*_`~^#\[\]{}<>•]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def get_ai_reply(user_text: str, lang: str) -> str:
    quick = static_answer(user_text, lang)
    if quick:
        return quick

    system       = SYSTEM_HI if lang == "hi" else SYSTEM_EN
    lang_history = _history[lang]
    lang_history.append({"role": "user", "content": user_text})

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": system}, *lang_history[-20:]],
            max_tokens=MAX_TOKENS,
            temperature=CHAT_TEMPERATURE,
        )
        reply = clean_text(response.choices[0].message.content)
    except Exception as exc:
        log.error("LLM error: %s", exc)
        reply = (
            "I'm sorry, I'm having trouble connecting. Please visit the information desk."
            if lang == "en" else
            "Maafi chahta hoon, connection mein problem hai. Information desk par jaayein."
        )

    lang_history.append({"role": "assistant", "content": reply})
    return reply

# ─────────────────────────────────────────────────────────────────
#  TTS
# ─────────────────────────────────────────────────────────────────

pygame.mixer.init()

def pick_voice(lang: str, text: str = "") -> str:
    if lang == "hi":
        return TTS_VOICE_HI
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F or 0x0600 <= cp <= 0x06FF:
            return TTS_VOICE_HI
    return TTS_VOICE_EN


async def _tts_save(text: str, path: str, voice: str):
    await edge_tts.Communicate(text, voice=voice).save(path)


def speak(text: str, lang: str = "en"):
    if not text.strip():
        return
    voice = pick_voice(lang, text)
    log.info("TTS [%s] %s", lang.upper(), text[:80])

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    asyncio.run(_tts_save(text, tmp_path, voice))
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.music.unload()
    try:
        os.unlink(tmp_path)
    except OSError:
        pass

# ─────────────────────────────────────────────────────────────────
#  FULL-SCREEN IMAGE DISPLAY
# ─────────────────────────────────────────────────────────────────

def show_image_fullscreen():
    if not HAS_DISPLAY:
        return
    if not os.path.exists(IMAGE_PATH):
        log.warning("Image not found: %s", IMAGE_PATH)
        return

    TICKER_ITEMS = [
        "✈  Welcome to Devi Ahilyabai Holkar Airport, Indore",
        "🛄  Baggage claim on ground floor, arrival hall",
        "🍽  Restaurants & food court — First Floor",
        "🚕  Pre-paid taxi counter — just outside arrivals exit",
        "📶  Free Wi-Fi: connect to  AAI_FREE_WIFI",
        "🏨  Hotel shuttle pick-up at Gate 2 every 30 minutes",
        "🅿  Multi-level parking in front of the terminal",
        "💱  Currency exchange at Terminal 1 ground floor",
        "🩺  Medical centre near Gate 5, Ground Floor",
        "❓  Information desk open 24 × 7 — near main entrance",
    ]

    root = tk.Tk()
    root.title("Indore Airport Assistant")
    root.configure(bg="black")

    # ── KEY FIX: maximise first, update, THEN read screen size ──
    root.attributes("-fullscreen", True)
    root.overrideredirect(True)
    root.update_idletasks()   # force Tkinter to fully render the window
    root.update()             # second pass — ensures geometry is finalised

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    root.geometry(f"{sw}x{sh}+0+0")  # explicitly set size as a fallback

    root.bind("<Escape>", lambda e: root.destroy())

    # ── Full-screen background image ────────────────────────────
    img   = Image.open(IMAGE_PATH).resize((sw, sh), Image.LANCZOS)
    photo = ImageTk.PhotoImage(img)

    label = tk.Label(root, image=photo, bg="black", borderwidth=0)
    label.image = photo
    label.place(x=0, y=0, width=sw, height=sh)  # use explicit w/h, not relwidth

    # ── Ticker pill at bottom-centre ────────────────────────────
    ticker_idx = [0]

    pill_w = int(sw * 0.70)
    pill_h = 52
    pill_x = (sw - pill_w) // 2
    pill_y = sh - 70

    canvas = tk.Canvas(
        root, width=pill_w, height=pill_h,
        bg="black", highlightthickness=0,
    )
    canvas.place(x=pill_x, y=pill_y)

    r = 26
    canvas.create_arc(0,          0,          r*2,      r*2,      start=90,  extent=90, fill="#001428", outline="#00C6FF", width=1)
    canvas.create_arc(pill_w-r*2, 0,          pill_w,   r*2,      start=0,   extent=90, fill="#001428", outline="#00C6FF", width=1)
    canvas.create_arc(0,          pill_h-r*2, r*2,      pill_h,   start=180, extent=90, fill="#001428", outline="#00C6FF", width=1)
    canvas.create_arc(pill_w-r*2, pill_h-r*2, pill_w,   pill_h,   start=270, extent=90, fill="#001428", outline="#00C6FF", width=1)
    canvas.create_rectangle(r,  0,      pill_w-r, pill_h, fill="#001428", outline="")
    canvas.create_rectangle(0,  r,      pill_w,   pill_h-r, fill="#001428", outline="")
    canvas.create_line(r,       0,      pill_w-r, 0,      fill="#00C6FF", width=1)
    canvas.create_line(r,       pill_h, pill_w-r, pill_h, fill="#00C6FF", width=1)
    canvas.create_line(0,       r,      0,        pill_h-r, fill="#00C6FF", width=1)
    canvas.create_line(pill_w,  r,      pill_w,   pill_h-r, fill="#00C6FF", width=1)

    dot = canvas.create_oval(14, pill_h//2-5, 24, pill_h//2+5, fill="#00C6FF", outline="")

    text_id = canvas.create_text(
        pill_w // 2 + 10, pill_h // 2,
        text=TICKER_ITEMS[0],
        fill="#E0F7FF",
        font=("Segoe UI", 15),
        anchor="center",
    )

    def rotate_ticker():
        ticker_idx[0] = (ticker_idx[0] + 1) % len(TICKER_ITEMS)
        canvas.itemconfig(text_id, text=TICKER_ITEMS[ticker_idx[0]])
        root.after(4000, rotate_ticker)

    dot_state = [True]
    def pulse_dot():
        canvas.itemconfig(dot, fill="#00C6FF" if dot_state[0] else "#003355")
        dot_state[0] = not dot_state[0]
        root.after(700, pulse_dot)

    root.after(4000, rotate_ticker)
    root.after(700,  pulse_dot)
    root.mainloop()
# ─────────────────────────────────────────────────────────────────
#  CHATBOT STATE MACHINE  (background thread)
# ─────────────────────────────────────────────────────────────────

def chatbot_loop():
    state = State.LISTENING
    reply = ""
    lang  = "hi"

    greeting = (
        "Namaste! mein AirBot hoon — Indore Airport ka AI assistant. "
        "Aap apna sawaal poochhiye."
    )
    speak(greeting, "hi")

    while True:

        if state == State.IDLE:
            log.info("IDLE — waiting for wake word")
            audio = capture_speech(timeout=IDLE_POLL_TIMEOUT)
            if audio is None:
                continue
            wake_text, _ = transcribe_fast(audio)          # ← faster model for wake word
            log.info("Heard (idle): %r", wake_text)
            if is_wake_word(wake_text):
                state = State.LISTENING
                wakeup = (
                    "Haan, boliye."                        # ← short = less TTS delay
                    if lang == "hi" else
                    "Yes, how can I help?"
                )
                speak(wakeup, lang)
            continue

        if state == State.LISTENING:
            log.info("LISTENING (timeout=%.0fs)", IDLE_TIMEOUT)
            audio = capture_speech(timeout=IDLE_TIMEOUT)
            if audio is None:
                state = State.IDLE
                idle_msg = (
                    "Idle mode mein ja raha hoon. Hello kahiye jab zaroorat ho."
                    if lang == "hi" else
                    "Going idle. Say Hello when you need help."
                )
                speak(idle_msg, lang)
                continue

            log.info("Transcribing...")
            user_text, lang = transcribe(audio)            # ← full quality for conversation
            if not user_text:
                log.warning("Empty transcript — re-listening")
                continue

            log.info("User [%s]: %s", lang.upper(), user_text)
            log.info("Getting AI reply...")
            reply = get_ai_reply(user_text, lang)
            log.info("Bot  [%s]: %s", lang.upper(), reply)
            state = State.SPEAKING
            continue

        if state == State.SPEAKING:
            speak(reply, lang)
            state = State.LISTENING
            continue
# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def main():
    gpio_setup()
    log.info("Starting Indore Airport Robot v3.1")

    # Chatbot runs in background — audio/TTS is all handled here
    bot_thread = threading.Thread(
        target=chatbot_loop, daemon=True, name="ChatbotLoop"
    )
    bot_thread.start()

    # Image display runs on main thread (Tkinter requirement)
    # This call blocks until the window is closed
    show_image_fullscreen()

    gpio_cleanup()
    log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
