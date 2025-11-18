import torch
import whisper
import os
from datetime import datetime


def format_timestamp(seconds: float):
    """Convert seconds to the timestamp format HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def transcribe_video(video_file):
    # Check if file exists
    if not os.path.isfile(video_file):
        print(f"[ERROR] File not found: {video_file}")
        return

    # Output filename
    base_name = os.path.splitext(video_file)[0]
    output_file = base_name + ".txt"

    # System/GPU check
    print("\n========== SYSTEM CHECK ==========")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU in use:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU (Install CUDA to enable GPU!)")
    print("==================================\n")

    # Load Whisper model (medium recommended)
    print("[INFO] Loading Whisper model 'medium'...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("medium", device=device)
    print("[INFO] Model loaded successfully.\n")

    print(f"[INFO] Starting transcription for: {video_file}")
    start_time = datetime.now()

    # Force Taglish/Filipino (best for Taglish)
    result = model.transcribe(
        video_file,
        verbose=True,
        language="tl",
        task="transcribe"
    )

    # Write transcript with timestamps
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            start_ts = format_timestamp(segment["start"])
            end_ts = format_timestamp(segment["end"])
            text = segment["text"].strip()

            f.write(f"[{start_ts} --> {end_ts}] {text}\n")

    end_time = datetime.now()

    print("\n========== FINISHED ==========")
    print(f"Output saved to: {output_file}")
    print(f"Time taken: {end_time - start_time}")
    if torch.cuda.is_available():
        print("GPU WAS USED: Yes ✔")
    else:
        print("GPU WAS USED: No ❌")
    print("================================\n")


if __name__ == "__main__":
    # Change your video file here
    video_filename = "max2.mp4"
    transcribe_video(video_filename)
