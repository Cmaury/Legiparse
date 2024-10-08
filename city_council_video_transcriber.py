import subprocess
import threading
import time
import logging
import json
import re
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base, Meeting, Media  # Importing Meeting and Media models
import database  # Ensure this module provides DATABASE_URL
import requests
from bs4 import BeautifulSoup
import urllib.parse
import signal
import sys
import traceback
from pyannote.audio import Pipeline
import librosa
import whisper
import torch
import textwrap
import tempfile
import soundfile as sf
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import shutil

from agenda_processor import get_meeting_details

# -------------------- Setup Logging --------------------
# Remove any existing handlers
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup Rotating File Handler
from logging.handlers import RotatingFileHandler

rotating_handler = RotatingFileHandler(
    'city_council_video_transcriber.log', maxBytes=5*1024*1024, backupCount=5
)  # 5 MB per file, 5 backups
rotating_handler.setLevel(logging.DEBUG)
rotating_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
rotating_handler.setFormatter(rotating_formatter)
logging.getLogger('').addHandler(rotating_handler)

# Setup Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Changed to DEBUG for more detailed console output
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger('').addHandler(console_handler)

logging.info("Logging is set up successfully.")

# -------------------- Signal Handlers --------------------
def signal_handler(sig, frame):
    logging.info("Shutdown signal received. Terminating script.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals

# -------------------- Load Configuration --------------------
def load_config(config_file='config.json'):
    logging.info("Entering load_config function.")
    try:
        logging.debug(f"Attempting to open configuration file: {config_file}")
        with open(config_file, 'r') as file:
            config = json.load(file)
        logging.info("Configuration loaded successfully.")
        logging.debug(f"Configuration contents: {config}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file '{config_file}' not found.")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in configuration file: {e}")
        logging.debug(traceback.format_exc())
        raise
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        logging.debug(traceback.format_exc())
        raise
    finally:
        logging.info("Exiting load_config function.")

# -------------------- Database Setup --------------------
def setup_database(db_uri):
    logging.info("Entering setup_database function.")
    logging.debug(f"Database URI: {db_uri}")  # Be cautious with sensitive info
    try:
        logging.debug("Creating SQLAlchemy engine.")
        engine = create_engine(db_uri, echo=False)
        logging.debug("Engine created successfully.")
        
        logging.debug("Creating sessionmaker.")
        Session = sessionmaker(bind=engine)
        logging.debug("Sessionmaker created successfully.")
        
        logging.debug("Instantiating session.")
        session = Session()
        logging.info("Database connection established successfully.")
        return session
    except Exception as e:
        logging.error(f"Database connection failed: {e}")
        logging.debug(traceback.format_exc())
        raise
    finally:
        logging.info("Exiting setup_database function.")

# -------------------- Ensure Directories Exist --------------------
def ensure_directory(directory_path):
    logging.info(f"Ensuring directory exists: {directory_path}")
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logging.info(f"Created directory: {directory_path}")
        else:
            logging.info(f"Directory already exists: {directory_path}")
    except Exception as e:
        logging.error(f"Failed to ensure directory '{directory_path}': {e}")
        logging.debug(traceback.format_exc())
        raise

# -------------------- Transcript Generation --------------------
def generate_transcript(
    audio_file: str,
    auth_token: str,
    whisper_model_path: str,
    whisper_model_size: str = "large",
    language: str = "en"
) -> str:
    """
    Generates a speaker diarization and transcription transcript from an audio file.

    Parameters:
    - audio_file (str): Path to the input audio file (e.g., "audio.mp3").
    - auth_token (str): Hugging Face authentication token for accessing models.
    - whisper_model_size (str, optional): Size of the Whisper model to load (e.g., "tiny", "base", "small", "medium", "large"). Defaults to "large".
    - language (str, optional): Language code for transcription (e.g., "en" for English). Defaults to "en".

    Returns:
    - final_transcript (str): The formatted transcript with speaker identification.
    """
    logging.info("Starting transcript generation.")
    try:
        logging.info("Loading models...")
        # Step 1: Load models
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        logging.debug("Speaker diarization pipeline loaded.")
        
        whisper_model = whisper.load_model(whisper_model_path)
        logging.debug(f"Whisper model '{whisper_model_size}' loaded.")
        
        logging.info("Performing speaker diarization...")
        # Step 2: Perform speaker diarization
        diarization = diarization_pipeline(audio_file)
        logging.debug("Speaker diarization completed.")
        
        # Step 3: Extract and consolidate speaker segments
        def consolidate_segments(segments):
            """
            Merge consecutive segments from the same speaker.
            """
            if not segments:
                return []
            
            consolidated = [segments[0]]
            
            for current in segments[1:]:
                last = consolidated[-1]
                if current['speaker'] == last['speaker']:
                    # Extend the last segment's end time
                    last['end'] = current['end']
                else:
                    consolidated.append(current)
            
            return consolidated

        # Extract speaker segments
        speaker_segments = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segment = {
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            }
            speaker_segments.append(segment)
            logging.debug(f"Extracted segment: Speaker={speaker}, Start={turn.start}, End={turn.end}")

        logging.info("Consolidating speaker segments...")
        # Consolidate consecutive segments by the same speaker
        consolidated_segments = consolidate_segments(speaker_segments)
        logging.debug(f"Consolidated segments count: {len(consolidated_segments)}")
        
        logging.info(f"Total consolidated segments: {len(consolidated_segments)}")
        
        # Step 4: Transcribe each consolidated segment
        def load_audio_segment(file_path, start_time, end_time):
            audio, sr = librosa.load(file_path, sr=16000)
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            audio_segment = audio[start_sample:end_sample]
            logging.debug(f"Loaded audio segment: Start={start_time}, End={end_time}")
            return audio_segment

        transcriptions = []
        
        logging.info("Transcribing segments...")
        for idx, segment in enumerate(consolidated_segments, 1):
            start_time = segment['start']
            end_time = segment['end']
            speaker = segment['speaker']
            logging.debug(f"Transcribing segment {idx}: Speaker={speaker}, Start={start_time}, End={end_time}")
            audio_segment = load_audio_segment(audio_file, start_time, end_time)
            
            # Save the audio segment to a temporary file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
                sf.write(tmp.name, audio_segment, 16000)
                logging.debug(f"Saved temporary audio segment to {tmp.name}")
                result = whisper_model.transcribe(tmp.name, fp16=False, language=language)  # Specify language if known
                logging.debug(f"Whisper transcription result: {result}")
            
            transcript = result['text'].strip()
            
            if transcript:  # Only add if there's transcribed text
                transcriptions.append({
                    "speaker": speaker,
                    "text": transcript
                })
                logging.debug(f"Transcript for segment {idx}: {transcript}")
            else:
                logging.debug(f"No transcript generated for segment {idx}.")

            # Optional: Print progress every 10 segments
            if idx % 10 == 0 or idx == len(consolidated_segments):
                logging.info(f"Transcribed {idx}/{len(consolidated_segments)} segments...")

        # Step 5: Format the transcript
        logging.info("Formatting the transcript...")
        def format_transcript(transcriptions):
            formatted_output = ""
            for entry in transcriptions:
                speaker = entry['speaker']
                text = entry['text']
                speaker_line = f"{speaker}:"
                text_lines = textwrap.wrap(text, width=70)
                centered_text = "\n".join([line.center(70) for line in text_lines])
                formatted_output += f"{speaker_line}\n{centered_text}\n\n"
                logging.debug(f"Formatted transcript for speaker {speaker}.")
            return formatted_output

        final_transcript = format_transcript(transcriptions)
        
        logging.info("Transcript generation completed successfully.")
        return final_transcript
    except Exception as e:
        logging.error(f"Failed to generate transcript: {e}")
        logging.debug(traceback.format_exc())
        return ""

# -------------------- Fetch Playlist URL --------------------
def fetch_playlist_url(source_url, chromedriver_path, headless=True):
    """
    Fetches the .m3u8 playlist URL from the source webpage using Selenium.

    Parameters:
    - source_url (str): The URL of the webpage to scrape.
    - chromedriver_path (str): Path to the ChromeDriver executable.
    - headless (bool, optional): Whether to run Chrome in headless mode. Defaults to True.

    Returns:
    - m3u8_url (str or None): The found .m3u8 URL or None if not found.
    """
    logging.info(f"Fetching playlist URL from {source_url}")
    try:
        if not source_url:
            logging.warning("Empty source URL provided. Cannot fetch playlist URL.")
            return None

        # Set up Chrome options
        chrome_options = Options()
        chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
        chrome_options.add_argument("--no-sandbox")  # Add this if you're running into sandbox issues
        if headless:
            chrome_options.add_argument("--headless")  # Run in headless mode

        # Enable performance logging
        chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        # Create a new Chrome WebDriver instance with performance logging enabled
        logging.debug("Initializing Chrome WebDriver.")
        driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)
        logging.debug("Chrome WebDriver initialized successfully.")

        # Navigate to the source URL
        logging.debug(f"Navigating to URL: {source_url}")
        driver.get(source_url)
        logging.info("Navigated to URL.")

        # Wait for a few seconds to ensure the page is fully loaded
        logging.debug("Waiting for the page to load.")
        time.sleep(5)  # Consider replacing with WebDriverWait for better reliability

        # Capture the network logs
        logging.debug("Capturing network logs.")
        logs = driver.get_log("performance")
        logging.debug(f"Captured {len(logs)} performance log entries.")

        # Parse the logs to find the .m3u8 URL
        m3u8_url = None
        logging.debug("Parsing network logs to find .m3u8 URL.")
        for log in logs:
            try:
                log_json = json.loads(log["message"])
                message = log_json.get("message", {})
                method = message.get("method", "")
                if "Network.responseReceived" in method:
                    response = message.get("params", {}).get("response", {})
                    url = response.get("url", "")
                    if ".m3u8" in url:
                        m3u8_url = url
                        logging.info(f"Found m3u8 URL: {m3u8_url}")
                        break  # Stop once we find the .m3u8 URL
            except json.JSONDecodeError as e:
                logging.debug(f"JSON decode error while parsing performance log: {e}")
                continue
            except Exception as e:
                logging.debug(f"Unexpected error while parsing performance log: {e}")
                continue

        # Close the browser
        logging.debug("Closing Chrome WebDriver.")
        driver.quit()
        logging.debug("Chrome WebDriver closed.")

        if m3u8_url:
            logging.info(f"Found m3u8 URL: {m3u8_url}")
            return m3u8_url
        else:
            logging.warning("No m3u8 URL found.")
            return None
    except Exception as e:
        logging.error(f"Error fetching playlist URL: {e}")
        logging.debug(traceback.format_exc())
        return None

# -------------------- Download Audio --------------------
def download_audio(input_source, output_filename="audio.mp3"):
    """
    Downloads audio from a playlist URL or extracts audio from a local MP4 file using FFmpeg.

    Args:
        input_source (str): URL to the playlist or path to a local MP4 file.
        output_filename (str, optional): Name of the output audio file. Defaults to "audio.mp3".

    Returns:
        bool: True if download/extraction was successful, False otherwise.
    """
    logging.info(f"Starting audio download/extraction for source: {input_source}")
    try:
        # Check if FFmpeg is installed
        if not shutil.which("ffmpeg"):
            logging.error("FFmpeg is not installed or not available in PATH.")
            return False

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
                logging.error(f"The specified file does not exist: {input_source}")
                return False
            if not input_source.lower().endswith('.mp4'):
                logging.error("The specified file is not an MP4 file.")
                return False

            command = [
                "ffmpeg",
                "-i", input_source,    # Input MP4 file
                "-vn",                 # Disable video recording (only extract audio)
                "-acodec", "mp3",      # Use the MP3 audio codec
                "-y",                  # Overwrite output file without asking
                output_filename        # Output file name
            ]
            action = "Extracting audio from local MP4 file"

        logging.info(f"{action}...")
        logging.debug(f"Running FFmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Audio processed successfully as '{output_filename}'.")
        return True
    except subprocess.CalledProcessError as e:
        # Capture and log FFmpeg's stderr output for debugging
        error_message = e.stderr.decode('utf-8') if e.stderr else 'No error message provided.'
        logging.error(f"Error processing audio: {error_message}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during audio processing: {e}")
        logging.debug(traceback.format_exc())
        return False

# -------------------- Download and Transcribe --------------------
def download_and_transcribe(source_url, config, auth_token, chromedriver_path, output_dir):
    """
    Downloads audio from a video URL, transcribes it with speaker identification, and saves the transcript to the database.

    Parameters:
    - source_url (str): The video URL to process.
    - config (dict): Configuration dictionary.
    - auth_token (str): Hugging Face authentication token.
    - chromedriver_path (str): Path to the ChromeDriver executable.
    - output_dir (str): Directory to save downloaded audio files.

    Returns:
    - transcript (str): The generated transcript.
    """
    try:
        logging.info(f"Processing source URL: {source_url}")

        if not source_url:
            logging.warning("Empty source URL provided. Skipping transcription.")
            return ""

        # Fetch the .m3u8 playlist URL
        logging.debug("Fetching playlist URL.")
        playlist_url = fetch_playlist_url(source_url, chromedriver_path)

        if not playlist_url:
            logging.warning(f"Could not find playlist URL for {source_url}. Skipping.")
            return ""

        # Define the audio file path
        timestamp = int(time.time())
        audio_filename = os.path.join(output_dir, f"audio_{timestamp}.mp3")
        logging.debug(f"Audio filename set to: {audio_filename}")

        # Download the audio from the playlist
        logging.debug("Downloading audio.")
        success = download_audio(playlist_url, audio_filename)
        if not success:
            logging.warning(f"Audio download failed for {playlist_url}. Skipping transcription.")
            return ""

        # Generate transcript
        logging.debug("Generating transcript.")
        transcript = generate_transcript(
            audio_file=audio_filename,
            auth_token=auth_token,
            whisper_model_path= config['transcript'].get('whisper_model_path'),
            whisper_model_size=config['transcript'].get('whisper_model_size', 'large'),
            language=config['transcript'].get('language', 'en')
        )

        if not transcript:
            logging.warning(f"Transcript generation failed for {source_url}.")
            return ""

        # Optionally, delete the audio file after transcription to save space
        try:
            os.remove(audio_filename)
            logging.info(f"Deleted temporary audio file: {audio_filename}")
        except Exception as e:
            logging.warning(f"Could not delete temporary audio file '{audio_filename}': {e}")

        return transcript
    except Exception as e:
        logging.error(f"Failed to download and transcribe {source_url}: {e}")
        logging.debug(traceback.format_exc())
        return ""

# -------------------- Process Local Video Files --------------------
def process_local_videos(session, video_folder, auth_token, config):
    """
    Processes all untranscribed video files in the specified local folder.

    Parameters:
    - session (Session): SQLAlchemy session object.
    - video_folder (str): Path to the folder containing video files.
    - auth_token (str): Hugging Face authentication token.
    - config (dict): Configuration dictionary.
    """
    logging.info(f"Starting to process local video files in '{video_folder}'")
    try:
        for filename in os.listdir(video_folder):
            if not filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                logging.debug(f"Skipping non-video file: {filename}")
                continue  # Skip non-video files

            filepath = os.path.join(video_folder, filename)
            logging.debug(f"Found video file: {filepath}")

            # Extract meetingid from filename assuming format 'meeting_<id>.mp4'
            match = re.match(r'meeting_(\d+)\.(mp4|mkv|avi|mov)$', filename, re.IGNORECASE)
            if not match:
                logging.warning(f"Filename '{filename}' does not match expected pattern 'meeting_<id>.<ext>'. Skipping.")
                continue

            meeting_id = int(match.group(1))
            logging.debug(f"Extracted meeting_id: {meeting_id}")

            # Find the Meeting entry
            meeting = session.query(Meeting).filter(Meeting.meetingid == meeting_id).first()
            if not meeting:
                logging.warning(f"No Meeting found with meetingid={meeting_id}. Skipping.")
                continue

            if meeting.transcript != 'none':
                logging.info(f"Transcript already exists for Meeting ID {meeting_id}. Skipping.")
                continue  # Already processed

            # Check if a Media entry exists for this video file
            media_entry = session.query(Media).filter(
                Media.meetingid == meeting_id,
                Media.url == filepath
            ).first()

            if not media_entry:
                # Create a new Media entry
                media_entry = Media(
                    meetingid=meeting_id,
                    mediatype='video',
                    url=filepath,
                    description=f"Local video file {filename}"
                )
                session.add(media_entry)
                session.commit()
                logging.info(f"Added Media entry for '{filename}' to the database.")

            # Transcribe the video file
            logging.info(f"Transcribing local video file for Meeting ID {meeting_id}: {filename}")
            transcript = generate_transcript(
                audio_file=filepath,
                auth_token=auth_token,
                whisper_model_path= config['transcript'].get('whisper_model_path'),                
                whisper_model_size=config['transcript'].get('whisper_model_size', 'large'),
                language=config['transcript'].get('language', 'en')
            )

            if transcript:
                # Update the Meeting entry with the transcript
                meeting.transcript = transcript
                meeting.processed_at = datetime.utcnow()
                session.commit()
                logging.info(f"Transcript saved for Meeting ID {meeting_id}.")
            else:
                logging.warning(f"No transcript generated for Meeting ID {meeting_id}.")

    except Exception as e:
        logging.error(f"Error processing local video files: {e}")
        logging.debug(traceback.format_exc())

# -------------------- Process Database Meetings --------------------
def process_db_meetings(session, auth_token, config):
    """
    Processes all meetings in the database that do not have a transcript.

    Parameters:
    - session (Session): SQLAlchemy session object.
    - auth_token (str): Hugging Face authentication token.
    - config (dict): Configuration dictionary.
    """
    logging.info("Starting to process untranscribed meetings from the database.")
    try:
        # Fetch all meetings where transcript is 'none' and videourl or meetingurl is present
        untranscribed_meetings = session.query(Meeting).filter(
            Meeting.transcript == 'none',
            (Meeting.videourl != None) | (Meeting.meetingurl != None)
        ).all()

        logging.info(f"Found {len(untranscribed_meetings)} untranscribed meetings.")

        chromedriver_path = config['video'].get('chromedriver_path')
        if not chromedriver_path:
            logging.error("ChromeDriver path not specified in configuration.")
            return

        output_dir = config['video'].get('output_directory')
        if not output_dir:
            logging.error("Output directory for audio files not specified in configuration.")
            return

        ensure_directory(output_dir)

        for meeting in untranscribed_meetings:
            meeting_id = meeting.meetingid
            source_url = meeting.videourl if meeting.videourl else None
            logging.info(f"Processing Meeting ID {meeting_id} with URL: {source_url}")

            if not source_url:
                logging.warning(f"Meeting ID {meeting_id} has no videourl or meetingurl. Skipping.")
                continue

            # Validate URL format
            parsed_url = urllib.parse.urlparse(source_url)
            if not parsed_url.scheme in ('http', 'https', 'ftp'):
                logging.warning(f"Meeting ID {meeting_id} has an invalid video URL: {source_url}. Skipping.")
                continue

            # Download and transcribe
            logging.debug("Starting download and transcription process.")
            transcript = download_and_transcribe(
                source_url=source_url,
                config=config,
                auth_token=auth_token,
                chromedriver_path=chromedriver_path,
                output_dir=output_dir
            )

            if transcript:
                # Save the transcript to the database
                meeting.transcript = transcript
                meeting.processed_at = datetime.utcnow()

                # Optionally, add a Media entry if desired
                media_entry = session.query(Media).filter(
                    Media.meetingid == meeting.meetingid,
                    Media.mediatype == 'video',
                    Media.url == source_url
                ).first()

                if not media_entry:
                    media_entry = Media(
                        meetingid=meeting.meetingid,
                        mediatype='video',
                        url=source_url,
                        description=f"Remote video URL {source_url}"
                    )
                    session.add(media_entry)

                session.commit()
                logging.info(f"Transcript saved for Meeting ID {meeting.meetingid}.")
            else:
                logging.warning(f"Transcript generation failed for Meeting ID {meeting.meetingid}.")

    except Exception as e:
        logging.error(f"Error processing database meetings: {e}")
        logging.debug(traceback.format_exc())

# -------------------- Main Function --------------------
def main():
    logging.info("Entering main function.")
    try:
        # Load configuration
        config = load_config()

        # Setup database
        session = setup_database(database.DATABASE_URL)

        # Extract video configuration
        video_config = config.get('video', {})
        if not video_config:
            logging.error("No 'video' configuration found in config file.")
            raise ValueError("Missing 'video' configuration.")

        video_output_dir = video_config.get('output_directory')
        video_folder = video_config.get('video_folder')  # Folder containing existing video files
        chromedriver_path = video_config.get('chromedriver_path')

        if not all([video_output_dir, chromedriver_path, video_folder]):
            logging.error("Incomplete 'video' configuration. 'output_directory', 'video_folder', and 'chromedriver_path' are required.")
            raise ValueError("Incomplete 'video' configuration.")

        logging.debug(f"Video Output Directory: {video_output_dir}")
        logging.debug(f"Video Folder: {video_folder}")
        logging.debug(f"ChromeDriver Path: {chromedriver_path}")

        # Ensure directories exist
        ensure_directory(video_output_dir)
        ensure_directory(video_folder)

        # Process existing local video files
        logging.info("Starting to process local video files.")
        process_local_videos(
            session=session,
            video_folder=video_folder,
            auth_token=config['transcript'].get('pynote_auth'),
            config=config
        )

        # Process untranscribed meetings from the database
        logging.info("Starting to process untranscribed meetings from the database.")
        process_db_meetings(
            session=session,
            auth_token=config['transcript'].get('pynote_auth'),
            config=config
        )

        logging.info("All processing completed successfully.")
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")
        logging.debug(traceback.format_exc())
    finally:
        logging.info("Exiting main function.")
        logging.info("Script has been terminated.")

if __name__ == "__main__":
    main()