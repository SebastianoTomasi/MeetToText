
# MeetToText: Audio/Video to DOCX Transcript and Summary

MeetToText converts audio or video into DOCX transcripts and optional summaries.
It chunks large audio, retries transient API failures, and stores outputs in
`data/transcripts/`.

## Features

- Audio/video to DOCX transcripts with timestamps
- Three tasks: `full`, `transcript`, `summarize`
- Three summary styles: `meeting`, `lecture`, `qa`
- Automatic chunking for large audio inputs
- STT default: `gpt-4o-transcribe` (with `gpt-4o-mini-transcribe` and `whisper-1` supported)
- Summary default: `gpt-5-mini` (optional higher-quality `gpt-5.2`)
- Archives source media to `data/audio_files/` when needed
- Configurable timeout and retry behavior

## Folder Structure

```text
MeetToText/
|-- main.py
|-- pyproject.toml
|-- poetry.lock
|-- README.md
`-- data/
    |-- audio_files/
    `-- transcripts/
```

## Requirements

- Python 3.13+
- [ffmpeg](https://ffmpeg.org/) available in `PATH`
- OpenAI API key in `.env`:

```env
OPENAI_API_KEY=your-api-key-here
```

## Installation

1. Clone:

```powershell
git clone https://github.com/SebastianoTomasi/MeetToText.git
cd MeetToText
```

2. Install dependencies:

```powershell
poetry install --no-root
```

Or:

```powershell
pip install -r requirements.txt
```

## Usage

From the project root:

```powershell
poetry run python main.py <input_file> [--task {full|transcript|summarize}] [--model MODEL] [--summarize-model MODEL] [--format {meeting|lecture|qa}] [--timeout SECONDS] [--max-retries N]
```

You can also run with plain `python` if that interpreter has project deps installed.

## Arguments

- `input_file`: media file for `full`/`transcript`, or `.docx` file for `summarize`
- `--task`: `full` (default), `transcript`, or `summarize`
- `--model`: speech-to-text model (default: `gpt-4o-transcribe`)
- `--summarize-model`: summary model (default: `gpt-5-mini`)
- `--format`: summary format `meeting` (default), `lecture`, or `qa`
- `--timeout`: per-request timeout in seconds (default: `60`)
- `--max-retries`: retries for transient failures (default: `4`)

## Model Notes

- `gpt-4o-transcribe` and `gpt-4o-mini-transcribe` run in JSON mode. The script
  keeps coarse timestamps (one per chunk).
- `whisper-1` supports segment timestamps, so transcript timestamps can be more
  granular.
- `gpt-5.2` can be used as a higher-quality summary model:

```powershell
poetry run python main.py input.mp3 --summarize-model gpt-5.2
```

## Examples

Full pipeline with defaults:

```powershell
poetry run python main.py .\data\audio_files\work_meeting.wav --format meeting
```

Transcript only:

```powershell
poetry run python main.py .\data\audio_files\work_meeting.wav --task transcript
```

Add summary to an existing DOCX in place:

```powershell
poetry run python main.py .\data\transcripts\work_meeting.docx --task summarize --format meeting
```

Use Whisper for finer timestamps:

```powershell
poetry run python main.py input.mp3 --model whisper-1
```

## Output

- `full`: creates `data/transcripts/<input_basename>.docx` with Summary + Transcript
- `transcript`: creates `data/transcripts/<input_basename>.docx` with Transcript only
- `summarize`: updates the provided DOCX in place by prepending a Summary section
- Source media is copied to `data/audio_files/` unless it is already there

## Troubleshooting

- If you see missing module errors, run with `poetry run python ...` to ensure
  the correct virtual environment is used.
- If `ffmpeg` is missing, install it and confirm `ffmpeg -version` works.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- [OpenAI](https://platform.openai.com/) for transcription and summarization APIs
- [python-docx](https://python-docx.readthedocs.io/)
- [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg)
