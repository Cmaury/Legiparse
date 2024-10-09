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
from transformers import WhisperProcessor, WhisperForConditionalGeneration
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
rotating_handler.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
rotating_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
rotating_handler.setFormatter(rotating_formatter)
logging.getLogger('').addHandler(rotating_handler)

# Setup Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # INFO level for console
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console_handler.setFormatter(console_formatter)
logging.getLogger('').addHandler(console_handler)

# Set the root logger level to DEBUG to capture all levels
logging.getLogger('').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)
logger.info("Logging is set up successfully.")

# -------------------- Signal Handlers --------------------
def signal_handler(sig, frame):
    logger.info("Shutdown signal received. Terminating script.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signals

# -------------------- Load Configuration --------------------
def load_config(config_file='config.json'):
    logger.info("Entering load_config function.")
    start_time = time.time()
    try:
        logger.debug(f"Attempting to open configuration file: {config_file}")
        with open(config_file, 'r') as file:
            config = json.load(file)
        logger.info("Configuration loaded successfully.")
        logger.debug(f"Configuration contents: {json.dumps(config, indent=4)}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_file}' not found.")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in configuration file: {e}")
        logger.debug(traceback.format_exc())
        raise
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting load_config function. Time taken: {elapsed_time:.2f}s")

# -------------------- Database Setup --------------------
def setup_database(db_uri):
    logger.info("Entering setup_database function.")
    start_time = time.time()
    logger.debug(f"Database URI: {db_uri}")  # Be cautious with sensitive info
    try:
        logger.debug("Creating SQLAlchemy engine.")
        engine = create_engine(db_uri, echo=False)
        logger.debug("Engine created successfully.")
        
        logger.debug("Creating sessionmaker.")
        Session = sessionmaker(bind=engine)
        logger.debug("Sessionmaker created successfully.")
        
        logger.debug("Instantiating session.")
        session = Session()
        logger.info("Database connection established successfully.")
        return session
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting setup_database function. Time taken: {elapsed_time:.2f}s")

# -------------------- Ensure Directories Exist --------------------
def ensure_directory(directory_path):
    logger.info(f"Ensuring directory exists: {directory_path}")
    start_time = time.time()
    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            logger.info(f"Created directory: {directory_path}")
        else:
            logger.info(f"Directory already exists: {directory_path}")
    except Exception as e:
        logger.error(f"Failed to ensure directory '{directory_path}': {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        elapsed_time = time.time() - start_time
        logger.debug(f"ensure_directory completed for '{directory_path}'. Time taken: {elapsed_time:.2f}s")

# -------------------- Transcript Generation --------------------
def generate_transcript(
    audio_file: str,
    auth_token: str,
    whisper_model_path: str,
    language: str = "en"
) -> str:
    """
    Generates a speaker diarization and transcription transcript from an audio file.

    Parameters:
    - audio_file (str): Path to the input audio file (e.g., "audio.mp3").
    - auth_token (str): Hugging Face authentication token for accessing models.
    - whisper_model_path (str): Path to the locally saved Whisper model.
    - language (str, optional): Language code for transcription (e.g., "en" for English). Defaults to "en".

    Returns:
    - final_transcript (str): The formatted transcript with speaker identification.
    """
    logger.info("Starting transcript generation.")
    start_time = time.time()
    try:
        logger.info("Loading speaker diarization pipeline...")
        # Step 1: Load models
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=auth_token
        )
        logger.debug("Speaker diarization pipeline loaded successfully.")
        
        logger.info(f"Loading Whisper model and processor from '{whisper_model_path}'...")
        # Load Whisper model and processor from Hugging Face
        processor = WhisperProcessor.from_pretrained(whisper_model_path)
        model = WhisperForConditionalGeneration.from_pretrained(whisper_model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug("Whisper model and processor loaded successfully.")
        
        logger.info("Performing speaker diarization...")
        # Step 2: Perform speaker diarization
        diarization = diarization_pipeline(audio_file)
        logger.debug("Speaker diarization completed successfully.")
        
        logger.info("Extracting speaker segments...")
        # Step 3: Extract and consolidate speaker segments
        def consolidate_segments(segments):
            """
            Merge consecutive segments from the same speaker.
            """
            logger.debug("Consolidating speaker segments...")
            if not segments:
                logger.debug("No segments to consolidate.")
                return []
            
            consolidated = [segments[0]]
            logger.debug(f"Initial segment added: {segments[0]}")
            
            for current in segments[1:]:
                last = consolidated[-1]
                if current['speaker'] == last['speaker']:
                    # Extend the last segment's end time
                    logger.debug(f"Extending segment for speaker '{current['speaker']}' from {last['end']} to {current['end']}")
                    last['end'] = current['end']
                else:
                    consolidated.append(current)
                    logger.debug(f"Adding new segment: {current}")
            
            logger.debug(f"Total consolidated segments: {len(consolidated)}")
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
            logger.debug(f"Extracted segment: Speaker={speaker}, Start={turn.start}, End={turn.end}")

        consolidated_segments = consolidate_segments(speaker_segments)
        logger.info(f"Total consolidated speaker segments: {len(consolidated_segments)}")
        
        # Step 4: Transcribe each consolidated segment
        def load_audio_segment(file_path, start_time, end_time):
            try:
                logger.debug(f"Loading audio segment from {start_time}s to {end_time}s")
                audio, sr = librosa.load(file_path, sr=16000)
                start_sample = int(start_time * sr)
                end_sample = int(end_time * sr)
                audio_segment = audio[start_sample:end_sample]
                logger.debug(f"Loaded audio segment length: {len(audio_segment)/sr:.2f}s")
                return audio_segment
            except Exception as e:
                logger.error(f"Error loading audio segment: {e}")
                logger.debug(traceback.format_exc())
                raise

        transcriptions = []
        
        logger.info("Transcribing audio segments...")
        for idx, segment in enumerate(consolidated_segments, 1):
            start_time = segment['start']
            end_time = segment['end']
            speaker = segment['speaker']
            logger.debug(f"Transcribing segment {idx}/{len(consolidated_segments)}: Speaker={speaker}, Start={start_time}, End={end_time}")
            audio_segment = load_audio_segment(audio_file, start_time, end_time)
            
            try:
                # Transcribe using Hugging Face's transformers
                inputs = processor(audio_segment, sampling_rate=16000, return_tensors="pt")
                input_features = inputs.input_features.to(model.device)
                logger.debug("Input features prepared for Whisper model.")
                
                predicted_ids = model.generate(input_features)
                logger.debug("Whisper model generated predicted IDs.")
                
                transcript = processor.decode(predicted_ids[0], skip_special_tokens=True).strip()
                logger.debug(f"Transcript for segment {idx}: {transcript}")
            except Exception as e:
                logger.error(f"Error during transcription of segment {idx}: {e}")
                logger.debug(traceback.format_exc())
                transcript = ""
            
            if transcript:
                transcriptions.append({
                    "speaker": speaker,
                    "text": transcript
                })
                logger.debug(f"Added transcript for segment {idx}: {transcript}")
            else:
                logger.debug(f"No transcript generated for segment {idx}.")

            # Optional: Print progress every 10 segments
            if idx % 10 == 0 or idx == len(consolidated_segments):
                logger.info(f"Transcribed {idx}/{len(consolidated_segments)} segments.")

        # Step 5: Format the transcript
        logger.info("Formatting the final transcript.")
        def format_transcript(transcriptions):
            formatted_output = ""
            for entry in transcriptions:
                speaker = entry['speaker']
                text = entry['text']
                speaker_line = f"{speaker}:"
                text_lines = textwrap.wrap(text, width=70)
                centered_text = "\n".join([line.center(70) for line in text_lines])
                formatted_output += f"{speaker_line}\n{centered_text}\n\n"
                logger.debug(f"Formatted transcript for speaker {speaker}.")
            return formatted_output

        final_transcript = format_transcript(transcriptions)
        logger.info("Transcript formatting completed successfully.")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Transcript generation process completed in {elapsed_time:.2f}s")
        return final_transcript

    except Exception as e:
        logger.error(f"Failed to generate transcript: {e}")
        logger.debug(traceback.format_exc())
        return ""
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting generate_transcript function. Total time: {elapsed_time:.2f}s")

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
    logger.info(f"Fetching playlist URL from {source_url}")
    start_time = time.time()
    try:
        if not source_url:
            logger.warning("Empty source URL provided. Cannot fetch playlist URL.")
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
        logger.debug("Initializing Chrome WebDriver.")
        driver = webdriver.Chrome(service=Service(chromedriver_path), options=chrome_options)
        logger.debug("Chrome WebDriver initialized successfully.")

        # Navigate to the source URL
        logger.debug(f"Navigating to URL: {source_url}")
        driver.get(source_url)
        logger.info("Navigated to URL.")

        # Wait for a few seconds to ensure the page is fully loaded
        logger.debug("Waiting for the page to load.")
        time.sleep(5)  # Consider replacing with WebDriverWait for better reliability

        # Capture the network logs
        logger.debug("Capturing network logs.")
        logs = driver.get_log("performance")
        logger.debug(f"Captured {len(logs)} performance log entries.")

        # Parse the logs to find the .m3u8 URL
        m3u8_url = None
        logger.debug("Parsing network logs to find .m3u8 URL.")
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
                        logger.info(f"Found m3u8 URL: {m3u8_url}")
                        break  # Stop once we find the .m3u8 URL
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode error while parsing performance log: {e}")
                continue
            except Exception as e:
                logger.debug(f"Unexpected error while parsing performance log: {e}")
                continue

        # Close the browser
        logger.debug("Closing Chrome WebDriver.")
        driver.quit()
        logger.debug("Chrome WebDriver closed.")

        if m3u8_url:
            logger.info(f"Found m3u8 URL: {m3u8_url}")
            return m3u8_url
        else:
            logger.warning("No m3u8 URL found.")
            return None
    except Exception as e:
        logger.error(f"Error fetching playlist URL: {e}")
        logger.debug(traceback.format_exc())
        return None
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting fetch_playlist_url function. Time taken: {elapsed_time:.2f}s")

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
    logger.info(f"Starting audio download/extraction for source: {input_source}")
    start_time = time.time()
    try:
        # Check if FFmpeg is installed
        if not shutil.which("ffmpeg"):
            logger.error("FFmpeg is not installed or not available in PATH.")
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
                logger.error(f"The specified file does not exist: {input_source}")
                return False
            if not input_source.lower().endswith('.mp4'):
                logger.error("The specified file is not an MP4 file.")
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

        logger.info(f"{action}...")
        logger.debug(f"Running FFmpeg command: {' '.join(command)}")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Audio processed successfully as '{output_filename}'.")
        return True
    except subprocess.CalledProcessError as e:
        # Capture and log FFmpeg's stderr output for debugging
        error_message = e.stderr.decode('utf-8') if e.stderr else 'No error message provided.'
        logger.error(f"Error processing audio: {error_message}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during audio processing: {e}")
        logger.debug(traceback.format_exc())
        return False
    finally:
        elapsed_time = time.time() - start_time
        logger.debug(f"download_audio completed. Time taken: {elapsed_time:.2f}s")

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
    logger.info(f"Processing source URL: {source_url}")
    start_time = time.time()
    try:
        if not source_url:
            logger.warning("Empty source URL provided. Skipping transcription.")
            return ""

        # Fetch the .m3u8 playlist URL
        logger.debug("Fetching playlist URL.")
        playlist_url = fetch_playlist_url(source_url, chromedriver_path)

        if not playlist_url:
            logger.warning(f"Could not find playlist URL for {source_url}. Skipping.")
            return ""

        # Define the audio file path
        timestamp = int(time.time())
        audio_filename = os.path.join(output_dir, f"audio_{timestamp}.mp3")
        logger.debug(f"Audio filename set to: {audio_filename}")

        # Download the audio from the playlist
        logger.debug("Downloading audio.")
        success = download_audio(playlist_url, audio_filename)
        if not success:
            logger.warning(f"Audio download failed for {playlist_url}. Skipping transcription.")
            return ""

        # Generate transcript
        logger.debug("Generating transcript.")
        transcript = generate_transcript(
            audio_file=audio_filename,
            auth_token=auth_token,
            whisper_model_path=config['transcript'].get('whisper_model_path'),
            language=config['transcript'].get('language', 'en')
        )

        if not transcript:
            logger.warning(f"Transcript generation failed for {source_url}.")
            return ""

        # Optionally, delete the audio file after transcription to save space
        try:
            os.remove(audio_filename)
            logger.info(f"Deleted temporary audio file: {audio_filename}")
        except Exception as e:
            logger.warning(f"Could not delete temporary audio file '{audio_filename}': {e}")

        elapsed_time = time.time() - start_time
        logger.debug(f"download_and_transcribe completed. Time taken: {elapsed_time:.2f}s")
        return transcript
    except Exception as e:
        logger.error(f"Failed to download and transcribe {source_url}: {e}")
        logger.debug(traceback.format_exc())
        return ""
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting download_and_transcribe function. Total time: {elapsed_time:.2f}s")

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
    logger.info(f"Starting to process local video files in '{video_folder}'")
    start_time = time.time()
    try:
        files_processed = 0
        files_skipped = 0

        for filename in os.listdir(video_folder):
            if not filename.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
                logger.debug(f"Skipping non-video file: {filename}")
                files_skipped += 1
                continue  # Skip non-video files

            filepath = os.path.join(video_folder, filename)
            logger.debug(f"Found video file: {filepath}")

            # Extract meetingid from filename assuming format 'meeting_<id>.mp4'
            match = re.match(r'meeting_(\d+)\.(mp4|mkv|avi|mov)$', filename, re.IGNORECASE)
            if not match:
                logger.warning(f"Filename '{filename}' does not match expected pattern 'meeting_<id>.<ext>'. Skipping.")
                files_skipped += 1
                continue

            meeting_id = int(match.group(1))
            logger.debug(f"Extracted meeting_id: {meeting_id}")

            # Find the Meeting entry
            meeting = session.query(Meeting).filter(Meeting.meetingid == meeting_id).first()
            if not meeting:
                logger.warning(f"No Meeting found with meetingid={meeting_id}. Skipping.")
                files_skipped += 1
                continue

            if meeting.transcript != 'none':
                logger.info(f"Transcript already exists for Meeting ID {meeting_id}. Skipping.")
                files_skipped += 1
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
                logger.info(f"Added Media entry for '{filename}' to the database.")

            # Transcribe the video file
            logger.info(f"Transcribing local video file for Meeting ID {meeting_id}: {filename}")
            transcript = generate_transcript(
                audio_file=filepath,
                auth_token=auth_token,
                whisper_model_path=config['transcript'].get('whisper_model_path'),
                language=config['transcript'].get('language', 'en')
            )

            if transcript:
                # Update the Meeting entry with the transcript
                meeting.transcript = transcript
                meeting.processed_at = datetime.utcnow()
                session.commit()
                logger.info(f"Transcript saved for Meeting ID {meeting_id}.")
                files_processed += 1
            else:
                logger.warning(f"No transcript generated for Meeting ID {meeting_id}.")
                files_skipped += 1

        logger.info(f"Processing of local video files completed. Processed: {files_processed}, Skipped: {files_skipped}")

    except Exception as e:
        logger.error(f"Error processing local video files: {e}")
        logger.debug(traceback.format_exc())
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting process_local_videos function. Total time: {elapsed_time:.2f}s")

# -------------------- Process Database Meetings --------------------
def process_db_meetings(session, auth_token, config):
    """
    Processes all meetings in the database that do not have a transcript.

    Parameters:
    - session (Session): SQLAlchemy session object.
    - auth_token (str): Hugging Face authentication token.
    - config (dict): Configuration dictionary.
    """
    logger.info("Starting to process untranscribed meetings from the database.")
    start_time = time.time()
    try:
        # Fetch all meetings where transcript is 'none' and videourl or meetingurl is present
        untranscribed_meetings = session.query(Meeting).filter(
            Meeting.transcript == 'none',
            (Meeting.videourl != None) | (Meeting.meetingurl != None)
        ).all()

        logger.info(f"Found {len(untranscribed_meetings)} untranscribed meetings.")

        chromedriver_path = config['video'].get('chromedriver_path')
        if not chromedriver_path:
            logger.error("ChromeDriver path not specified in configuration.")
            return

        output_dir = config['video'].get('output_directory')
        if not output_dir:
            logger.error("Output directory for audio files not specified in configuration.")
            return

        ensure_directory(output_dir)

        meetings_processed = 0
        meetings_skipped = 0

        for meeting in untranscribed_meetings:
            meeting_id = meeting.meetingid
            source_url = meeting.videourl if meeting.videourl else None
            logger.info(f"Processing Meeting ID {meeting_id} with URL: {source_url}")

            if not source_url:
                logger.warning(f"Meeting ID {meeting_id} has no videourl or meetingurl. Skipping.")
                meetings_skipped += 1
                continue

            # Validate URL format
            parsed_url = urllib.parse.urlparse(source_url)
            if not parsed_url.scheme in ('http', 'https', 'ftp'):
                logger.warning(f"Meeting ID {meeting_id} has an invalid video URL: {source_url}. Skipping.")
                meetings_skipped += 1
                continue

            # Download and transcribe
            logger.debug("Starting download and transcription process.")
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
                logger.info(f"Transcript saved for Meeting ID {meeting.meetingid}.")
                meetings_processed += 1
            else:
                logger.warning(f"Transcript generation failed for Meeting ID {meeting.meetingid}.")
                meetings_skipped += 1

        logger.info(f"Processing of database meetings completed. Processed: {meetings_processed}, Skipped: {meetings_skipped}")

    except Exception as e:
        logger.error(f"Error processing database meetings: {e}")
        logger.debug(traceback.format_exc())
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting process_db_meetings function. Total time: {elapsed_time:.2f}s")

# -------------------- Main Function --------------------
def main():
    logger.info("Entering main function.")
    start_time = time.time()
    try:
        # Load configuration
        config = load_config()

        # Setup database
        session = setup_database(database.DATABASE_URL)

        # Extract video configuration
        video_config = config.get('video', {})
        if not video_config:
            logger.error("No 'video' configuration found in config file.")
            raise ValueError("Missing 'video' configuration.")

        video_output_dir = video_config.get('output_directory')
        video_folder = video_config.get('video_folder')  # Folder containing existing video files
        chromedriver_path = video_config.get('chromedriver_path')

        if not all([video_output_dir, chromedriver_path, video_folder]):
            logger.error("Incomplete 'video' configuration. 'output_directory', 'video_folder', and 'chromedriver_path' are required.")
            raise ValueError("Incomplete 'video' configuration.")

        logger.debug(f"Video Output Directory: {video_output_dir}")
        logger.debug(f"Video Folder: {video_folder}")
        logger.debug(f"ChromeDriver Path: {chromedriver_path}")

        # Ensure directories exist
        ensure_directory(video_output_dir)
        ensure_directory(video_folder)

        # Process existing local video files
        logger.info("Starting to process local video files.")
        process_local_videos(
            session=session,
            video_folder=video_folder,
            auth_token=config['transcript'].get('pynote_auth'),
            config=config
        )

        # Process untranscribed meetings from the database
        logger.info("Starting to process untranscribed meetings from the database.")
        process_db_meetings(
            session=session,
            auth_token=config['transcript'].get('pynote_auth'),
            config=config
        )

        logger.info("All processing completed successfully.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        logger.debug(traceback.format_exc())
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Exiting main function. Total time: {elapsed_time:.2f}s")
        logger.info("Script has been terminated.")

if __name__ == "__main__":
    main()