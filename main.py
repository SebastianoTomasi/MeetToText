#!/usr/bin/env python3
"""
Meeting-to-DOCX pipeline – “chunked” edition
July 2025

• Automatically splits long recordings so every request stays below OpenAI’s
  current 25 MB file‑size cap (≈ 13 min at 16 kHz mono WAV). ​​​
• Removes the final export of the temporary WAV file (no clutter on disk).
• Keeps the previous debug prints and overall CLI.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from docx import Document
from imageio_ffmpeg import get_ffmpeg_exe
from openai import OpenAI

AUDIO_RATE = 16_000  # Hz
MAX_UPLOAD_MB = 25   # Hard‑limit per OpenAI docs (≈ 25 MB per request)
_BYTES_PER_SEC = AUDIO_RATE * 2  # 16‑bit mono → 2 bytes per sample
CHUNK_SECONDS = max(1, (MAX_UPLOAD_MB * 1024 ** 2) // _BYTES_PER_SEC)  # ≈ 787 s

# -----------------------------------------------------------------------------
# Audio helpers
# -----------------------------------------------------------------------------

def extract_audio(src: Path, wav_out: Path) -> Path:
    """Convert *src* to 16 kHz mono WAV stored at *wav_out*."""
    ffmpeg = get_ffmpeg_exe()
    cmd: List[str] = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-ar",
        str(AUDIO_RATE),
        "-ac",
        "1",
        str(wav_out),
    ]
    from subprocess import run  # local import keeps top‑level clean

    run(cmd, check=True, close_fds=sys.platform != "win32")
    return wav_out


def chunk_audio(wav: Path, chunk_dir: Path) -> List[Path]:
    """Split *wav* into ≤ 25 MB (~13 min) chunks inside *chunk_dir*.

    If the source file is already under the limit it is returned as‑is.
    """
    limit_bytes = MAX_UPLOAD_MB * 1024 ** 2

    if wav.stat().st_size <= limit_bytes:
        return [wav]

    chunk_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = get_ffmpeg_exe()
    out_pattern = chunk_dir / "chunk_%04d.wav"

    cmd: List[str] = [
        ffmpeg,
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(wav),
        "-f",
        "segment",
        "-segment_time",
        str(CHUNK_SECONDS),
        "-c",
        "copy",
        str(out_pattern),
    ]
    from subprocess import run

    run(cmd, check=True, close_fds=sys.platform != "win32")

    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    if not chunks:
        raise RuntimeError("Chunking failed: no output segments created.")
    return chunks


# -----------------------------------------------------------------------------
# OpenAI STT
# -----------------------------------------------------------------------------

def _response_format_for(model: str) -> str:
    return "json" if "-transcribe" in model else "verbose_json"


def transcribe_chunk(wav: Path, *, model: str = "gpt-4o-transcribe") -> Dict[str, Any]:
    client = OpenAI()
    fmt = _response_format_for(model)
    print(f"⏳ Transcribing {wav.name} with {model} …")
    with wav.open("rb") as fh:
        resp = client.audio.transcriptions.create(
            model=model,
            file=fh,
            response_format=fmt,
            timestamp_granularities=["segment"] if fmt == "verbose_json" else None,
        )
    data = resp.model_dump()
    return {
        "text": data.get("text", ""),
        "segments": data.get("segments", []),
    }


# -----------------------------------------------------------------------------
# Summarisation (unchanged)
# -----------------------------------------------------------------------------

def summarise(text: str, *, model: str = "gpt-4.1-mini") -> str:
    """
    Produce un riepilogo esecutivo e un elenco di azioni da una trascrizione.

    Tecniche prompt:
    • role‑priming (facilitatore / PM)               • chain‑of‑thought nascosto
    • istruzioni step‑wise ANALISI/OUTPUT           • struttura markdown con emoji
    • requisiti su lunghezza, formato e scadenze    • temperature bassa per coerenza
    """
    client = OpenAI()

    # 1️⃣ Contesto e tono
    system_msg = (
        "Agisci come un esperto facilitatore di riunioni e project‑manager. "
        "Il tuo compito è distillare trascrizioni in sintesi utilizzabili da dirigenti italiani. "
        "Il risultato deve essere neutro, preciso e professionale. "
        "Pensa passo‑passo internamente ma NON rivelare il ragionamento."
    )

    # 2️⃣ Istruzioni dettagliate
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

    # 3️⃣ Transcript delimitato per parsing più robusto
    user_msg = f"""transcript:\n {text}"""

    print("⏳ Generazione del riepilogo …")
    res = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instructions},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
        top_p=0.95,
        max_tokens=900,
        presence_penalty=0.2,
        frequency_penalty=0.2,
    )
    return res.choices[0].message.content.strip()


# -----------------------------------------------------------------------------
# DOCX output
# -----------------------------------------------------------------------------

def save_docx(transcript: str, summary: str, dst: Path) -> None:
    doc = Document()
    doc.add_heading("Meeting Transcript", level=1)
    for line in transcript.splitlines():
        if line.strip():
            doc.add_paragraph(line)

    doc.add_page_break()
    doc.add_heading("Executive Summary & Action Items", level=1)
    for line in summary.splitlines():
        doc.add_paragraph(line)

    doc.save(dst)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def transcribe_file(input_path: str, output_docx: str = "transcript.docx") -> None:
    load_dotenv()

    src = Path(input_path)
    if not src.exists():
        sys.exit(f"❌ File not found: {src}")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        wav_path = td_path / "full.wav"

        # 1. Extract & chunk
        extract_audio(src, wav_path)
        chunk_dir = td_path / "chunks"
        chunks = chunk_audio(wav_path, chunk_dir)

        # 2. STT for each chunk
        transcript_parts: List[str] = []
        for idx, cp in enumerate(chunks, 1):
            size_mb = cp.stat().st_size / 1_048_576
            print(f"— Chunk {idx}/{len(chunks)} ({size_mb:.1f} MB)")
            stt_resp = transcribe_chunk(cp)
            transcript_parts.append(stt_resp["text"])

        transcript_text = "\n".join(transcript_parts)

    # 3. Summarisation
    summary_text = summarise(transcript_text)

    # 4. DOCX
    save_docx(transcript_text, summary_text, Path(output_docx))

    # 5. Console output
    print("\n—— FULL TRANSCRIPT ——\n")
    print(transcript_text or "[empty]")
    print("\n—— SUMMARY ——\n")
    print(summary_text or "[empty]")

    print(f"\n📄 DOCX saved  → {Path(output_docx).absolute()}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Transcribe a media file → DOCX (chunk‑safe)")
    p.add_argument("input_file", help="Video or audio file (any format ffmpeg supports)")
    p.add_argument("output_docx", nargs="?", default="transcript.docx")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infile = args.input_file
    out_docx = args.output_docx or f"transcript_{Path(infile).stem}.docx"
    transcribe_file(infile, out_docx)
