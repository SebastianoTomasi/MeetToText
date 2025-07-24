
# MeetToText: Audio/Video to DOCX Transcript & Summary

MeetToText converts audio or video files into timestamped DOCX transcripts, with optional structured summaries. It supports chunked processing for large files, archives original media, and offers multiple summary formats for meetings, lectures, and Q&A sessions.


## Features

- Converts audio/video files to DOCX transcripts with paragraph-level timestamps
- Adds structured summaries in different formats (meeting, lecture, Q&A)
- Archives original media files in `data/audio_files/`
- Chunked processing for large files
- Uses OpenAI Whisper or GPT-4o models for transcription
- Robust error handling with configurable retries and timeouts


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
   ```powershell
   git clone https://github.com/SebastianoTomasi/MeetToText.git
   cd MeetToText
   ```
2. **Install dependencies:**
   - With Poetry:
     ```powershell
     poetry install
     ```
   - Or with pip:
     ```powershell
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

```powershell
python main.py <input_file> [--task {full|transcript|summarize}] [--model {whisper-1|gpt-4o-transcribe}] [--format {meeting|lecture|qa}] [--summarize-model {o4-mini|o3}] [--timeout SECONDS] [--max-retries N]
```

### Arguments

- `input_file`: Path to your audio/video file (e.g., `.m4a`, `.wav`, `.mkv`, etc.) or DOCX file (for summarize task)
- `--task`:
    - `full` (default): transcript + summary
    - `transcript`: transcript only
    - `summarize`: add summary to DOCX (input must be a DOCX file)
- `--model`: STT model for audio tasks (default: whisper-1; supports gpt-4o-transcribe)
- `--summarize-model`: model for summarization (default o4-mini; supports o3)  
- `--format`: Summary style (choices: meeting, lecture, qa)
    - `meeting`: Executive summary & action items (for meetings)
    - `lecture`: Clean, well-structured lecture/talk text
    - `qa`: Extracts Q&A pairs and suggestions
- `--timeout`: Per-request timeout in seconds (default: 60)
- `--max-retries`: Retries on transient errors (default: 4)

### Example

```powershell
python main.py data/audio_files/intervista_santroni07072025.m4a --task full --model whisper-1 --format lecture
```

The transcript and summary will be saved as a DOCX file in `data/transcripts/`.


## Output

- **DOCX file** in `data/transcripts/` with:
  1. **Summary** (format depends on --format: meeting, lecture, or qa)
  2. **Transcript** (with paragraph timestamps)
- **Original media** archived in `data/audio_files/`


## License

MIT License. See [LICENSE](LICENSE) for details.


## Acknowledgments

- Uses [OpenAI](https://platform.openai.com/) for transcription and summarization.
- Built with [python-docx](https://python-docx.readthedocs.io/), [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg), and [ffmpeg](https://ffmpeg.org/).
