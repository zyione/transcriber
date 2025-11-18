pip install torch torchvision torchaudio

pip install openai-whisper


if not using GPU
    pip uninstall torch torchvision torchaudio -y

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    or 

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

