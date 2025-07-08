#!/usr/bin/env python3
"""
Meeting-to-DOCX pipeline – “chunked” edition + paragraph timestamps
July 2025 (rev E-folder-model)

Change log (rev E-folder-model)
===============================
• **Updated folder structure**: the original media file is copied to
  `./data/audio_files/` for archival, and the generated DOCX transcript is
  saved to `./data/transcripts/`.
• **Summary model switched**: summarisation now defaults to `o4-mini`, per
  OpenAI’s docs: https://platform.openai.com/docs/models/o4-mini
• **Chat completion params updated**: using `max_completion_tokens` instead
  of `max_tokens` for `o4-mini` compatibility.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from dotenv import load_dotenv
from docx import Document
from imageio_ffmpeg import get_ffmpeg_exe
from openai import OpenAI, APIConnectionError, InternalServerError
import httpx

AUDIO_RATE = 16_000  # Hz
MAX_UPLOAD_MB = 10  # Hard-limit per OpenAI docs (≈ 25 MB per request)
_BYTES_PER_SEC = AUDIO_RATE * 2  # 16-bit mono → 2 bytes per sample
CHUNK_SECONDS = max(1, (MAX_UPLOAD_MB * 1024 ** 2) // _BYTES_PER_SEC)  # ≈ 787 s

# -----------------------------------------------------------------------------
# Models & formats
# -----------------------------------------------------------------------------
SEGMENT_MODELS = {
    "whisper-1",
    "gpt-4o-mini-transcribe",
    "gpt-4o-transcribe",
}

def _supports_segments(model: str) -> bool:
    return model in SEGMENT_MODELS


def _response_format_for(model: str) -> str:
    return "verbose_json" if _supports_segments(model) else "json"

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def format_timestamp(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02}:{m:02}:{s:02}"

# -----------------------------------------------------------------------------
# Paragraph grouping
# -----------------------------------------------------------------------------

def paragraphs_from_segments(
    segs: List[Dict[str, Any]],
    *,
    time_offset: float,
    min_chars: int = 300,
) -> List[str]:
    lines: List[str] = []
    buf: List[str] = []
    para_start: Optional[float] = None

    def flush():
        if buf and para_start is not None:
            ts = format_timestamp(para_start)
            lines.append(f"[{ts}] {' '.join(buf).strip()}")
            buf.clear()

    for seg in segs:
        if para_start is None:
            para_start = time_offset + seg["start"]
        buf.append(seg["text"].strip())
        ends_sentence = buf[-1].endswith((".", "!", "?"))
        if ends_sentence and sum(map(len, buf)) >= min_chars:
            flush()
            para_start = None

    flush()
    return lines

# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------
from subprocess import run

def extract_audio(src: Path, wav_out: Path) -> Path:
    ffmpeg = get_ffmpeg_exe()
    cmd = [ffmpeg, "-loglevel", "error", "-y", "-i", str(src), "-ar", str(AUDIO_RATE), "-ac", "1", str(wav_out)]
    run(cmd, check=True, close_fds=sys.platform != "win32")
    return wav_out

def chunk_audio(wav: Path, chunk_dir: Path) -> List[Path]:
    limit_bytes = MAX_UPLOAD_MB * 1024 ** 2
    if wav.stat().st_size <= limit_bytes:
        return [wav]
    chunk_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    out_pattern = chunk_dir / "chunk_%04d.wav"
    cmd = [ffmpeg, "-loglevel", "error", "-y", "-i", str(wav), "-f", "segment", "-segment_time", str(CHUNK_SECONDS), "-c", "copy", str(out_pattern)]
    run(cmd, check=True, close_fds=sys.platform != "win32")
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("Chunking failed: no output segments created.")
    return chunks

# -----------------------------------------------------------------------------
# Retry helper
# -----------------------------------------------------------------------------

def _with_retries(fn, *, max_retries: int, backoff: float = 2.0, exc_types: tuple[Type[BaseException], ...]):
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except exc_types as e:
            if attempt == max_retries:
                raise
            wait = backoff ** (attempt - 1)
            print(f"⚠️ {e.__class__.__name__}: {e} – retry {attempt}/{max_retries - 1} in {wait:.1f}s …")
            time.sleep(wait)

# -----------------------------------------------------------------------------
# OpenAI STT
# -----------------------------------------------------------------------------

def transcribe_chunk(
    wav: Path,
    *,
    model: str,
    timeout_s: float,
    max_retries: int,
) -> Dict[str, Any]:
    client = OpenAI(timeout=timeout_s)
    fmt = _response_format_for(model)
    print(f"⏳ Transcribing {wav.name} with {model} … (format: {fmt})")

    def _call():
        with wav.open("rb") as fh:
            return client.audio.transcriptions.create(
                model=model,
                file=fh,
                response_format=fmt,
                timestamp_granularities=["segment"] if fmt == "verbose_json" else None,
            )

    resp = _with_retries(
        _call, max_retries=max_retries, exc_types=(APIConnectionError, InternalServerError, httpx.ReadError)
    )
    data = resp.model_dump()
    return {"text": data.get("text", ""), "segments": data.get("segments", [])}

# -----------------------------------------------------------------------------
# Summarisation
# -----------------------------------------------------------------------------

def summarise(text: str, *, model: str = "o4-mini-2025-04-16", timeout_s: float = 60.0) -> str:
    client = OpenAI(timeout=timeout_s)
    system_msg = (
        "Agisci come un esperto facilitatore di riunioni e project-manager. "
        "Il tuo compito è distillare trascrizioni in sintesi utilizzabili da dirigenti italiani. "
        "Il risultato deve essere neutro, preciso e professionale. "
        "Pensa passo-passo internamente ma NON rivelare il ragionamento."
    )
    instructions = (
        "Lavora in due fasi:\n"
        "1. ANALISI (nascosta): individua temi, decisioni, metriche, rischi, prossimi passi.\n"
        "2. OUTPUT (visibile): restituisci ESCLUSIVAMENTE le due sezioni seguenti, in markdown.\n\n"
        "### 📝 Riepilogo Esecutivo\n"
        "• 5-7 bullet (≤ 20 parole) con verbi all'infinito.\n\n"
        "### ✅ Azioni e Responsabili\n"
        "• Formato: **<Responsabile> → <Azione> (scadenza)**\n"
        "• Se la scadenza non è indicata, scrivi “data da definire”."
    )
    user_msg = f"transcript:\n {text}"
    temper = 1 if model== "o4-mini-2025-04-16" else 0.0
    print("⏳ Generazione del riepilogo …")
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instructions},
            {"role": "user", "content": user_msg},
        ],
        temperature=temper,
        max_completion_tokens=10000,
    )
    return res.choices[0].message.content.strip()

# -----------------------------------------------------------------------------
# DOCX writer
# -----------------------------------------------------------------------------

def save_docx(transcript: str, summary: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    doc.add_heading("Executive Summary & Action Items", level=1)
    for line in summary.splitlines():
        doc.add_paragraph(line)
    doc.add_page_break()
    doc.add_heading("Meeting Transcript", level=1)
    for line in transcript.splitlines():
        if line.strip():
            doc.add_paragraph(line)
    
    doc.save(dst)

# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def transcribe_file(
    input_path: str,
    *,
    stt_model: str,
    timeout_s: float,
    max_retries: int,
) -> None:
    load_dotenv()
    src = Path(input_path)
    if not src.exists():
        sys.exit(f"❌ File not found: {src}")

    # Setup folders
    transcripts_dir = Path("./data/transcripts")
    audio_dir = Path("./data/audio_files")
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Derive paths
    output_docx_path = transcripts_dir / src.with_suffix(".docx").name
    archive_path = audio_dir / src.name
    if src.resolve() != archive_path.resolve():
        try:
            shutil.copy2(src, archive_path)
        except Exception as e:
            print(f"⚠️  Unable to archive original media: {e}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav_path = td_path / "full.wav"
        extract_audio(src, wav_path)
        chunk_dir = td_path / "chunks"
        chunks = chunk_audio(wav_path, chunk_dir)

        plain_parts: List[str] = []
        timestamped_lines: List[str] = []
        time_offset = 0.0

        for idx, cp in enumerate(chunks, 1):
            print(f"— Chunk {idx}/{len(chunks)} ({cp.stat().st_size / 1_048_576:.1f} MB)")
            stt_resp = transcribe_chunk(cp, model=stt_model, timeout_s=timeout_s, max_retries=max_retries)
            plain_parts.append(stt_resp["text"])
            segments = stt_resp.get("segments", [])
            if segments:
                timestamped_lines.extend(paragraphs_from_segments(segments, time_offset=time_offset))
                time_offset += segments[-1].get("end", 0.0)
            else:
                ts = format_timestamp(time_offset)
                timestamped_lines.append(f"[{ts}] {stt_resp['text'].strip()}")
                time_offset += cp.stat().st_size / _BYTES_PER_SEC

        full_transcript = "\n".join(plain_parts)
        final_transcript = "\n".join(timestamped_lines)

    summary_text = summarise(full_transcript, timeout_s=timeout_s)
    save_docx(final_transcript, summary_text, output_docx_path)

    print(f"\n📄 Transcript saved: {output_docx_path.absolute()}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Transcribe a media file → DOCX (chunk-safe + paragraph timestamps + retries)"
    )
    p.add_argument("input_file", help="Video or audio file (any format ffmpeg supports)")
    p.add_argument(
        "--model", default="whisper-1",
        help="STT model for transcription (default: whisper-1)"
    )
    p.add_argument(
        "--timeout", type=float, default=60,
        help="Per-request timeout in seconds"
    )
    p.add_argument(
        "--max-retries", type=int, default=4,
        help="Number of retries on transient errors"
    )
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    transcribe_file(
        input_path=args.input_file,
        stt_model=args.model,
        timeout_s=args.timeout,
        max_retries=args.max_retries,
    )
