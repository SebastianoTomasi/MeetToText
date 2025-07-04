#!/usr/bin/env python3
"""
Minimal transcript script using a bundled FFmpeg via pip. No external FFmpeg install needed.

Call `python main.py` after `pip install openai python-dotenv python-docx imageio-ffmpeg`
and set your OpenAI key in a `.env` file.
"""

import sys
import tempfile
from pathlib import Path
import subprocess

from dotenv import load_dotenv
from openai import OpenAI
from docx import Document
from imageio_ffmpeg import get_ffmpeg_exe

# Target audio sample rate
AUDIO_RATE = 16_000  # Hz


def extract_audio(src: Path) -> Path:
    """
    Convert any media to 16-kHz mono WAV using a bundled ffmpeg binary.
    """
    ffmpeg_exe = get_ffmpeg_exe()
    tmpdir = Path(tempfile.mkdtemp(prefix="audio_"))
    wav = tmpdir / "audio.wav"
    cmd = [
        ffmpeg_exe,
        "-loglevel", "error",
        "-y", "-i", str(src),
        "-ar", str(AUDIO_RATE),
        "-ac", "1",
        str(wav),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(f"❌ FFmpeg processing failed: {e}")
    return wav


def transcribe(wav: Path, model: str = "gpt-4o-transcribe") -> str:
    """
    Send audio to OpenAI STT and return the transcript text.
    """
    client = OpenAI()  # reads OPENAI_API_KEY from env
    with wav.open("rb") as f:
        resp = client.audio.transcriptions.create(
            model=model,
            file=f,
            response_format="json",
        )
    return resp.text


def save_docx(text: str, dst: Path) -> None:
    """Write the transcript into a .docx file."""
    doc = Document()
    doc.add_heading("Meeting Transcript", level=1)
    doc.add_paragraph(text)
    doc.save(dst)


def transcribe_file(input_path: str, output_path: str = "transcript.docx") -> None:
    """
    High-level helper: extract audio, transcribe, and save to DOCX.
    """
    load_dotenv()  # loads OPENAI_API_KEY
    src = Path(input_path)
    if not src.exists():
        sys.exit(f"❌ File not found: {src}")

    wav = extract_audio(src)
    text = transcribe(wav)
    save_docx(text, Path(output_path))
    print(f"✅ Saved transcript → {Path(output_path).absolute()}")


if __name__ == "__main__":
    # Edit this to your file, then run `python main.py`
    transcribe_file("videoplayback.m4a")
