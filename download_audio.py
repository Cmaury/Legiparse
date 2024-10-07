import os
import shutil
import subprocess
import urllib.parse

def download_audio(input_source, output_filename="audio.mp3"):
    """
    Downloads audio from a playlist URL or extracts audio from a local MP4 file using FFmpeg.

    Args:
        input_source (str): URL to the playlist or path to a local MP4 file.
        output_filename (str, optional): Name of the output audio file. Defaults to "audio.mp3".

    Raises:
        EnvironmentError: If FFmpeg is not installed or not available in PATH.
        ValueError: If the input_source is invalid or the file does not exist.
    """
    # Check if FFmpeg is installed
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg is not installed or not available in PATH.")

    # Parse the input_source to determine if it's a URL or a local file
    parsed_url = urllib.parse.urlparse(input_source)
    is_url = parsed_url.scheme in ('http', 'https', 'ftp')

    if is_url:
        # Input is a URL (e.g., playlist)
        command = [
            "ffmpeg",
            "-i", input_source,    # Input playlist URL
            "-vn",                 # Disable video recording (only extract audio)
            "-acodec", "mp3",      # Use the MP3 audio codec
            "-y",                  # Overwrite output file without asking
            output_filename        # Output file name
        ]
        action = "Downloading audio from URL"
    else:
        # Input is assumed to be a local file path
        if not os.path.isfile(input_source):
            raise ValueError(f"The specified file does not exist: {input_source}")
        if not input_source.lower().endswith('.mp4'):
            raise ValueError("The specified file is not an MP4 file.")

        command = [
            "ffmpeg",
            "-i", input_source,    # Input MP4 file
            "-vn",                 # Disable video recording (only extract audio)
            "-acodec", "mp3",      # Use the MP3 audio codec
            "-y",                  # Overwrite output file without asking
            output_filename        # Output file name
        ]
        action = "Extracting audio from local MP4 file"

    try:
        print(f"{action}...")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio processed successfully as '{output_filename}'.")
    except subprocess.CalledProcessError as e:
        # Capture and log FFmpeg's stderr output for debugging
        error_message = e.stderr.decode('utf-8') if e.stderr else 'No error message provided.'
        print(f"Error processing audio: {error_message}")