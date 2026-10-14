import os
import shutil
from datetime import datetime

import torch
from faster_whisper import WhisperModel
from tqdm import tqdm


def ensure_ffmpeg():
    """Ensure ffmpeg executable is available on PATH."""
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir = os.path.dirname(ffmpeg_exe)
        standard_ffmpeg = os.path.join(ffmpeg_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        if not os.path.exists(standard_ffmpeg):
            shutil.copyfile(ffmpeg_exe, standard_ffmpeg)
        if ffmpeg_dir not in os.environ.get("PATH", ""):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def format_timestamp(seconds: float) -> str:
    """Convert seconds to the timestamp format HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def transcribe_batch():
    ensure_ffmpeg()

    # Use all available CPU cores when running without a GPU
    torch.set_num_threads(os.cpu_count())

    # Base directories relative to script location
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input")
    output_dir = os.path.join(base_dir, "output")
    history_dir = os.path.join(base_dir, "history-input")

    # Create directories if they do not exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(history_dir, exist_ok=True)

    # Supported audio/video extensions
    valid_extensions = (
        ".mp3", ".mp4", ".m4a", ".wav", ".aac",
        ".ogg", ".flac", ".wma", ".opus", ".mkv",
        ".mov", ".webm", ".m4v"
    )
    files_to_process = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(valid_extensions)
    ]

    if not files_to_process:
        print("\n==================================")
        print(f"[INFO] No supported media files found in: {input_dir}")
        print(f"[INFO] Supported formats: {', '.join(valid_extensions)}")
        print(f"[INFO] Please place your media files in the 'input' folder and run the script again.")
        print("==================================\n")
        return

    # System/GPU check
    print("\n========== SYSTEM CHECK ==========")
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        print("GPU in use:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU (Install CUDA to enable GPU!)")
    print("==================================\n")

    # Load Whisper model once for batch processing (faster-whisper / CTranslate2 backend)
    print("[INFO] Loading Whisper model 'medium'...")
    device = "cuda" if cuda_available else "cpu"
    compute_type = "float16" if cuda_available else "int8"
    model = WhisperModel("medium", device=device, compute_type=compute_type)
    print("[INFO] Model loaded successfully.\n")

    print(f"[INFO] Found {len(files_to_process)} file(s) to process.\n")

    for idx, file_name in enumerate(files_to_process, 1):
        input_file_path = os.path.join(input_dir, file_name)
        base_name, _ = os.path.splitext(file_name)
        output_file_name = f"{base_name}-converted.txt"
        output_file_path = os.path.join(output_dir, output_file_name)

        # Skip files that were already transcribed (e.g. resuming after an interruption)
        if os.path.exists(output_file_path):
            print(f"[{idx}/{len(files_to_process)}] [SKIP] Already transcribed: {file_name}")
            continue

        print(f"[{idx}/{len(files_to_process)}] Processing: {file_name}")
        start_time = datetime.now()

        try:
            # Force Taglish/Filipino transcription
            segments_gen, info = model.transcribe(
                input_file_path,
                language="tl",
                task="transcribe",
                vad_filter=True,  # skip silent/non-speech stretches: faster + fewer hallucinations
            )

            duration = info.duration or 0
            tmp_output_path = output_file_path + ".tmp"

            with open(tmp_output_path, "w", encoding="utf-8") as f, \
                 tqdm(total=round(duration, 1), unit="s", desc="  Transcribing", leave=False) as pbar:
                last_end = 0.0
                for segment in segments_gen:
                    start_ts = format_timestamp(segment.start)
                    end_ts = format_timestamp(segment.end)
                    text = segment.text.strip()
                    f.write(f"[{start_ts} --> {end_ts}] {text}\n")

                    # Advance the progress bar by however much audio this segment covered
                    pbar.update(round(segment.end - last_end, 1))
                    last_end = segment.end

                # Close out the bar in case VAD trimmed trailing silence
                if duration and last_end < duration:
                    pbar.update(round(duration - last_end, 1))

            # Atomic write: only replace the real output once writing succeeded fully
            os.replace(tmp_output_path, output_file_path)

            # Move processed input to history-input
            history_file_path = os.path.join(history_dir, file_name)
            if os.path.exists(history_file_path):
                # If a file with the same name exists, append a timestamp
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                name_part, ext_part = os.path.splitext(file_name)
                history_file_path = os.path.join(history_dir, f"{name_part}_{timestamp_str}{ext_part}")

            shutil.move(input_file_path, history_file_path)

            end_time = datetime.now()
            print("\n---------------- FINISHED FILE ----------------")
            print(f"Transcript saved to : {output_file_path}")
            print(f"Moved input file to : {history_file_path}")
            print(f"Time taken          : {end_time - start_time}")
            print("----------------------------------------------\n")

        except Exception as e:
            # Clean up any partial tmp file left behind by a failed run
            tmp_output_path = output_file_path + ".tmp"
            if os.path.exists(tmp_output_path):
                os.remove(tmp_output_path)
            print(f"[ERROR] Failed to process {file_name}: {e}\n")

    print("\n========== ALL FILES PROCESSED ==========\n")


if __name__ == "__main__":
    transcribe_batch()