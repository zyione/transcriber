# Transcriber

Batch-transcribe audio and video into timestamped text using the Whisper
`medium` model through faster-whisper. The script currently uses Filipino
(`language="tl"`) for Taglish/Filipino recordings and skips silent sections.

## Install on Windows

Install Python, then open PowerShell in this repository. Create an isolated
environment and install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install a CUDA-enabled PyTorch build into the same
environment using the command from the [official PyTorch installer](https://pytorch.org/get-started/locally/).
Replace its `pip` or `pip3` prefix with `.\.venv\Scripts\python.exe -m pip`.
Also follow the [faster-whisper GPU requirements](https://github.com/SYSTRAN/faster-whisper#requirements)
for the CUDA/cuDNN libraries used by its CTranslate2 backend. A CUDA-enabled
PyTorch install alone does not provide all of those runtime requirements.

Check whether PyTorch sees your GPU:

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

The script selects CUDA with float16 when PyTorch detects a GPU; otherwise it
uses CPU with int8. CPU transcription can take longer. If pip cannot find a
compatible dependency for your Python version, use a Python version supported
by both PyTorch and faster-whisper and recreate the environment.

## Run

1. Place recordings in `input/` (create the folder if necessary).
2. Run:

   ```powershell
   .\.venv\Scripts\python.exe transcribe.py
   ```

3. Find timestamped transcripts in `output/<recording>-converted.txt`.
   Successfully processed recordings move to `history-input/`.

The first run downloads the model and needs internet access. Files with an
existing transcript are skipped; move or remove that transcript to process
the recording again. Failed recordings stay in `input/` for retrying.

Supported formats: MP3, MP4, M4A, WAV, AAC, OGG, FLAC, WMA, OPUS, MKV, MOV,
WEBM, and M4V. PyAV provides the FFmpeg libraries used to decode media;
`imageio-ffmpeg` also supplies the script's fallback FFmpeg executable.

## Existing installations / metadata_errors fix

If you already run the script with `py transcribe.py`, update that same
environment from the repository folder:

```powershell
py -m pip install -r requirements.txt
py transcribe.py
```

The requirements pin `faster-whisper==1.2.1` and `av==18.1.0`. With PyAV 19.0.2,
this faster-whisper version fails during audio decoding with:

```text
open() got an unexpected keyword argument 'metadata_errors'
```

PyAV 18.1.0 was verified to decode the previously failing M4A recording.
Keep these versions together when updating dependencies. The script does not
need `openai-whisper`, `torchvision`, or `torchaudio`.
