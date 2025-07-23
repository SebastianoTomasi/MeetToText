# MeetToText

**MeetToText** is a command-line tool that transcribes meetings or interviews from audio/video files into timestamped DOCX documents, with automatic executive summaries and action items. It is designed for Italian-speaking professionals who need accurate, chunk-safe transcription and concise meeting summaries.

## Features

- **Transcribes audio/video files** (any format supported by ffmpeg) to DOCX with paragraph-level timestamps.
- **Chunked processing**: Handles large files by splitting them into manageable segments for OpenAI's API.
- **Automatic executive summary** and action items, generated using OpenAI's `o4-mini` model.
- **Robust error handling**: Retries on transient API errors.
- **Archival**: Copies original media to an archive folder for safekeeping.
- **Customizable**: Choose STT model, timeout, and retry count via CLI.

## Folder Structure

```
MeetToText/
├── main.py
├── pyproject.toml
├── poetry.lock
├── README.md
└── data/
    ├── audio_files/        # Archived original media files
    ├── audio_test/         # (Optional) Test audio files
    └── transcripts/        # Generated DOCX transcripts
```

## Requirements

- Python 3.8+
- [ffmpeg](https://ffmpeg.org/) (for audio extraction)
- OpenAI API key (set in a `.env` file)
- Dependencies (install via Poetry or pip):
  - `openai`, `python-dotenv`, `python-docx`, `imageio-ffmpeg`, `httpx`

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/SebastianoTomasi/MeetToText.git
   cd MeetToText
   ```
2. **Install dependencies:**
   - With Poetry:
     ```sh
     poetry install
     ```
   - Or with pip:
     ```sh
     pip install -r requirements.txt
     ```
3. **Set up your OpenAI API key:**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```
4. **Ensure ffmpeg is installed and in your PATH.**


## Usage

From the project root, run:

```sh
python main.py <input_file> [--model whisper-1] [--format meeting] [--timeout 60] [--max-retries 4]
```

- `<input_file>`: Path to your audio or video file (e.g., `.m4a`, `.wav`, `.mkv`, etc.)
- `--model`: (Optional) STT model to use (default: `whisper-1`)
- `--format`: (Optional) Type of summary to generate (default: `meeting`). Options:
  - `meeting`: Executive summary & action items (for meetings)
  - `lecture`: Clean, well-written rewrite of a talk or lecture
  - `qa`: Extracts Q&A pairs and final suggestions from Q&A sessions
- `--timeout`: (Optional) Per-request timeout in seconds (default: 60)
- `--max-retries`: (Optional) Number of retries on transient errors (default: 4)

### Example

```sh
python main.py data/audio_files/intervista_santroni07072025.m4a --format lecture
```

The transcript and summary will be saved as a DOCX file in `data/transcripts/`.

## Output

- **DOCX file** with two sections:
  1. **Executive Summary & Action Items** (in Italian, formatted for managers)
  2. **Meeting Transcript** (with paragraph timestamps)

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Uses [OpenAI](https://platform.openai.com/) for transcription and summarization.
- Built with [python-docx](https://python-docx.readthedocs.io/), [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), and [ffmpeg](https://ffmpeg.org/).
