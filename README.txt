See README.md for full Windows installation, GPU setup, and usage instructions.

For an existing installation, run these commands from the repository folder:

py -m pip install -r requirements.txt
py transcribe.py

requirements.txt pins compatible faster-whisper and PyAV versions to prevent
the "metadata_errors" audio decoding error.
