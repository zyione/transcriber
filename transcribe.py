import os
import shutil
from datetime import datetime
import torch
import whisper


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
    valid_extensions = (".mp3", ".mp4")
    files_to_process = [
        f for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(valid_extensions)
    ]

    if not files_to_process:
        print("\n==================================")
        print(f"[INFO] No .mp3 or .mp4 files found in: {input_dir}")
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

    # Load Whisper model once for batch processing
    print("[INFO] Loading Whisper model 'medium'...")
    device = "cuda" if cuda_available else "cpu"
    model = whisper.load_model("medium", device=device)
    print("[INFO] Model loaded successfully.\n")

    print(f"[INFO] Found {len(files_to_process)} file(s) to process.\n")

    for idx, file_name in enumerate(files_to_process, 1):
        input_file_path = os.path.join(input_dir, file_name)
        base_name, _ = os.path.splitext(file_name)
        output_file_name = f"{base_name}-converted.txt"
        output_file_path = os.path.join(output_dir, output_file_name)

        print(f"[{idx}/{len(files_to_process)}] Processing: {file_name}")
        start_time = datetime.now()

        try:
            # Force Taglish/Filipino transcription
            result = model.transcribe(
                input_file_path,
                verbose=True,
                language="tl",
                task="transcribe"
            )

            # Write transcript with timestamps
            with open(output_file_path, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    start_ts = format_timestamp(segment["start"])
                    end_ts = format_timestamp(segment["end"])
                    text = segment["text"].strip()
                    f.write(f"[{start_ts} --> {end_ts}] {text}\n")

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
            print(f"[ERROR] Failed to process {file_name}: {e}\n")

    print("\n========== ALL FILES PROCESSED ==========\n")


if __name__ == "__main__":
    transcribe_batch()
