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
from models import Base, Meeting  # Ensure this imports your SQLAlchemy models
import database  # Ensure this module provides DATABASE_URL
import requests
from bs4 import BeautifulSoup
import urllib.parse
import signal
import sys
import traceback
from logging.handlers import RotatingFileHandler

# -------------------- Setup Logging --------------------
# Remove any existing handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup Rotating File Handler
rotating_handler = RotatingFileHandler(
    'city_council_livestream_download.log', maxBytes=5*1024*1024, backupCount=5
)  # 5 MB per file, 5 backups
rotating_handler.setLevel(logging.DEBUG)
rotating_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
rotating_handler.setFormatter(rotating_formatter)
logging.getLogger('').addHandler(rotating_handler)

# Setup Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Adjust as needed
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger('').addHandler(console_handler)

logging.info("Logging is set up successfully.")

# -------------------- Signal Handlers --------------------
def signal_handler(sig, frame):
    logging.info("Shutdown signal received. Terminating script.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)    # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)   # Handle termination signals

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

# -------------------- Ensure Videos Directory Exists --------------------
def ensure_videos_directory(output_dir):
    logging.info("Entering ensure_videos_directory function.")
    try:
        logging.debug(f"Checking if directory '{output_dir}' exists.")
        if not os.path.exists(output_dir):
            logging.debug(f"Directory '{output_dir}' does not exist. Creating it.")
            os.makedirs(output_dir)
            logging.info(f"Created directory: {output_dir}")
        else:
            logging.info(f"Directory already exists: {output_dir}")
    except Exception as e:
        logging.error(f"Failed to create directory '{output_dir}': {e}")
        logging.debug(traceback.format_exc())
        raise
    finally:
        logging.info("Exiting ensure_videos_directory function.")

# -------------------- Load Processed Events --------------------
def load_processed_events(processed_events_file):
    logging.info("Entering load_processed_events function.")
    try:
        logging.debug(f"Checking if processed events file '{processed_events_file}' exists.")
        if os.path.exists(processed_events_file):
            logging.debug(f"Found processed events file: {processed_events_file}. Loading it.")
            with open(processed_events_file, 'r') as file:
                processed_events = json.load(file)
            logging.info(f"Loaded {len(processed_events)} processed events.")
        else:
            logging.info("No processed events file found. Starting fresh.")
            processed_events = []
        return processed_events
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in processed events file: {e}")
        logging.debug(traceback.format_exc())
        return []
    except Exception as e:
        logging.error(f"Failed to load processed events: {e}")
        logging.debug(traceback.format_exc())
        return []
    finally:
        logging.info("Exiting load_processed_events function.")

# -------------------- Save Processed Events --------------------
def save_processed_events(processed_events_file, processed_events):
    logging.info("Entering save_processed_events function.")
    try:
        logging.debug(f"Saving {len(processed_events)} processed events to '{processed_events_file}'.")
        with open(processed_events_file, 'w') as file:
            json.dump(processed_events, file, indent=4)
        logging.info(f"Saved {len(processed_events)} processed events successfully.")
    except Exception as e:
        logging.error(f"Failed to save processed events: {e}")
        logging.debug(traceback.format_exc())
    finally:
        logging.info("Exiting save_processed_events function.")

# -------------------- Load Processed Videos --------------------
def load_processed_videos(processed_videos_file):
    logging.info("Entering load_processed_videos function.")
    try:
        logging.debug(f"Checking if processed videos file '{processed_videos_file}' exists.")
        if os.path.exists(processed_videos_file):
            logging.debug(f"Found processed videos file: {processed_videos_file}. Loading it.")
            with open(processed_videos_file, 'r') as file:
                processed_videos = json.load(file)
            logging.info(f"Loaded {len(processed_videos)} processed videos.")
        else:
            logging.info("No processed videos file found. Starting fresh.")
            processed_videos = []
        return processed_videos
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error in processed videos file: {e}")
        logging.debug(traceback.format_exc())
        return []
    except Exception as e:
        logging.error(f"Failed to load processed videos: {e}")
        logging.debug(traceback.format_exc())
        return []
    finally:
        logging.info("Exiting load_processed_videos function.")

# -------------------- Save Processed Videos --------------------
def save_processed_videos(processed_videos_file, processed_videos):
    logging.info("Entering save_processed_videos function.")
    try:
        logging.debug(f"Saving {len(processed_videos)} processed videos to '{processed_videos_file}'.")
        with open(processed_videos_file, 'w') as file:
            json.dump(processed_videos, file, indent=4)
        logging.info(f"Saved {len(processed_videos)} processed videos successfully.")
    except Exception as e:
        logging.error(f"Failed to save processed videos: {e}")
        logging.debug(traceback.format_exc())
    finally:
        logging.info("Exiting save_processed_videos function.")

# -------------------- Generate Filename from Meeting ID --------------------
def generate_filename_from_meetingid(output_dir, meeting_id):
    logging.info("Entering generate_filename_from_meetingid function.")
    """
    Generates a filename using the meeting ID.
    Format: meeting_<meetingid>.mp4
    """
    filename = f"meeting_{meeting_id}.mp4"
    filepath = os.path.join(output_dir, filename)
    logging.debug(f"Generated filename: {filepath}")
    logging.info("Exiting generate_filename_from_meetingid function.")
    return filepath

# -------------------- Scrape "In Progress" Video URL --------------------
def get_in_progress_video_url(webpage_url):
    logging.info("Entering get_in_progress_video_url function.")
    """
    Scrapes the specified webpage to find the video URL with link text "In Progress".
    
    Args:
        webpage_url (str): The URL of the webpage to scrape.
    
    Returns:
        str or None: The extracted video URL if found; otherwise, None.
    """
    try:
        logging.debug(f"Sending GET request to '{webpage_url}'.")
        response = requests.get(webpage_url, timeout=10)  # Added timeout for reliability
        response.raise_for_status()
        logging.info(f"Fetched webpage successfully: {webpage_url}")
        
        logging.debug("Parsing HTML content with BeautifulSoup.")
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all <a> tags with exact text "In Progress"
        logging.debug("Searching for '<a>' tags with text 'In Progress'.")
        in_progress_links = soup.find_all('a', string=re.compile(r'^\s*In Progress\s*$', re.IGNORECASE))
        
        if not in_progress_links:
            logging.warning(f"No 'In Progress' links found on the webpage: {webpage_url}")
            return None
        
        # Assuming the first "In Progress" link is the desired one
        in_progress_link = in_progress_links[0]
        video_url = in_progress_link.get('href')
        logging.info(f"Extracted 'In Progress' link href: {video_url}")
        
        # If the href is relative, make it absolute
        if video_url and not urllib.parse.urlparse(video_url).scheme:
            video_url = urllib.parse.urljoin(webpage_url, video_url)
            logging.info(f"Converted relative video URL to absolute: {video_url}")
        else:
            logging.info(f"Video URL is already absolute: {video_url}")
        
        return video_url
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP request failed while fetching '{webpage_url}': {e}")
        logging.debug(traceback.format_exc())
        return None
    except Exception as e:
        logging.error(f"Failed to scrape 'In Progress' video URL: {e}")
        logging.debug(traceback.format_exc())
        return None
    finally:
        logging.info("Exiting get_in_progress_video_url function.")

# -------------------- Function to Save Video Stream --------------------
def save_video_stream(stream_url, output_file, max_duration, silence_duration, stop_event):
    logging.info("Entering save_video_stream function.")
    """
    Uses FFmpeg to capture and save the video stream.
    Terminates after max_duration seconds, silence_duration seconds of silence,
    or when stop_event is set.
    
    Args:
        stream_url (str): URL of the video stream to capture.
        output_file (str): Path to save the captured video file.
        max_duration (int): Maximum duration to capture in seconds.
        silence_duration (int): Duration of silence to detect in seconds to terminate recording.
        stop_event (threading.Event): Event to signal termination of recording.
    """
    logging.info(f"Starting video capture for file: {output_file}")
    
    # FFmpeg command with silencedetect filter
    command = [
        'ffmpeg',
        '-i', stream_url,                                      # Input stream URL
        '-c', 'copy',                                          # Copy codec (no re-encoding)
        '-f', 'mp4',                                           # Output format
        '-y',                                                  # Overwrite output file without asking
        '-af', f'silencedetect=n=-50dB:d={silence_duration}',  # Audio filter for silence detection
        '-loglevel', 'info',                                   # Set log level to info
        output_file
    ]
    logging.debug(f"FFmpeg command: {' '.join(command)}")
    
    try:
        logging.debug("Starting FFmpeg subprocess.")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        logging.info("FFmpeg process started successfully.")
    except FileNotFoundError:
        logging.error("FFmpeg executable not found. Ensure FFmpeg is installed and in the system PATH.")
        return
    except Exception as e:
        logging.error(f"Failed to start FFmpeg: {e}")
        logging.debug(traceback.format_exc())
        return
    
    # Variables to track silence
    silence_start = None
    silence_detected = False
    
    # Compile regex for silencedetect
    silence_start_re = re.compile(r'silence_start: (\d+\.\d+)')
    silence_end_re = re.compile(r'silence_end: (\d+\.\d+) \| silence_duration: (\d+\.\d+)')
    
    # Start time
    start_time = time.time()
    
    # Function to monitor FFmpeg stderr for silence
    def monitor_ffmpeg():
        nonlocal silence_start, silence_detected
        try:
            for line in process.stderr:
                # Log each line from FFmpeg stderr at DEBUG level
                logging.debug(f"FFmpeg stderr: {line.strip()}")
                
                # Detect silence start
                match_start = silence_start_re.search(line)
                if match_start:
                    silence_start = float(match_start.group(1))
                    logging.info(f"Silence started at {silence_start} seconds.")
    
                # Detect silence end
                match_end = silence_end_re.search(line)
                if match_end and silence_start is not None:
                    silence_duration_detected = float(match_end.group(2))
                    logging.info(f"Silence ended at {match_end.group(1)} seconds, duration: {silence_duration_detected} seconds.")
                    if silence_duration_detected >= silence_duration:
                        logging.info(f"Detected {silence_duration_detected} seconds of silence. Terminating FFmpeg.")
                        silence_detected = True
                        process.terminate()
                        break
        except Exception as e:
            logging.error(f"Error while monitoring FFmpeg stderr: {e}")
            logging.debug(traceback.format_exc())
            process.terminate()
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_ffmpeg, name="FFmpegMonitorThread")
    monitor_thread.start()
    logging.info("Started FFmpeg monitoring thread.")
    
    # Monitor for max_duration, silence, or stop_event
    try:
        while True:
            elapsed_time = time.time() - start_time
            logging.debug(f"Elapsed time: {elapsed_time:.2f} seconds.")
            
            if stop_event.is_set():
                logging.info("Stop event received. Terminating FFmpeg.")
                process.terminate()
                break
            if elapsed_time >= max_duration:
                logging.info(f"Reached maximum duration of {max_duration} seconds. Terminating FFmpeg.")
                process.terminate()
                break
            if silence_detected:
                logging.info("Silence detected. FFmpeg termination initiated.")
                break
            if process.poll() is not None:
                logging.warning("FFmpeg process terminated unexpectedly.")
                break
            time.sleep(1)
    except Exception as e:
        logging.error(f"Error during FFmpeg monitoring loop: {e}")
        logging.debug(traceback.format_exc())
        process.terminate()
    finally:
        monitor_thread.join()
        logging.info("FFmpeg monitoring thread has ended.")
    
    # Wait for the process to terminate
    try:
        logging.debug("Waiting for FFmpeg process to terminate.")
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode != 0:
            logging.error(f"FFmpeg terminated with return code {process.returncode}.")
            logging.error(f"FFmpeg stderr: {stderr}")
        else:
            logging.info("FFmpeg process completed successfully.")
    except subprocess.TimeoutExpired:
        logging.error("FFmpeg did not terminate in time. Killing the process.")
        process.kill()
        stdout, stderr = process.communicate()
        logging.debug(f"FFmpeg killed. stderr: {stderr}")
    
    logging.info(f"Video capture for file '{output_file}' has ended.")
    logging.info("Exiting save_video_stream function.")

# -------------------- Function to Fetch Upcoming Meetings --------------------
def get_upcoming_meetings(session, limit=10):
    logging.info("Entering get_upcoming_meetings function.")
    """
    Retrieves a list of upcoming meetings ordered by datetime.
    """
    current_time = datetime.now()
    try:
        logging.debug(f"Fetching meetings with datetime >= {current_time}. Limit: {limit}")
        meetings = session.query(Meeting).filter(Meeting.datetime >= current_time).order_by(Meeting.datetime).limit(limit).all()
        logging.info(f"Fetched {len(meetings)} upcoming meetings.")
        logging.debug(f"Upcoming meetings IDs: {[meeting.meetingid for meeting in meetings]}")
        return meetings
    except Exception as e:
        logging.error(f"Failed to fetch upcoming meetings: {e}")
        logging.debug(traceback.format_exc())
        return []
    finally:
        logging.info("Exiting get_upcoming_meetings function.")

def display_upcoming_meetings(meetings):
    logging.info("Entering display_upcoming_meetings function.")
    """
    Displays the list of upcoming meetings.
    """
    try:
        print("\nUpcoming City Council Meetings:")
        print("-" * 60)
        for meeting in meetings:
            meeting_time = meeting.datetime.strftime("%Y-%m-%d %H:%M")
            meeting_info = (
                f"Meeting ID     : {meeting.meetingid}\n"
                f"Name           : {meeting.name}\n"
                f"Date & Time    : {meeting_time}\n"
                f"Location       : {meeting.location}\n"
                f"Agenda Status  : {meeting.agendastatus}\n"
                f"Minutes Status : {meeting.minutesstatus}\n"
                f"Video URL      : {meeting.videourl}\n"
                f"{'-' * 60}"
            )
            print(meeting_info)
            logging.debug(f"Displayed meeting: ID={meeting.meetingid}, Name={meeting.name}")
    except Exception as e:
        logging.error(f"Failed to display upcoming meetings: {e}")
        logging.debug(traceback.format_exc())
    finally:
        logging.info("Exiting display_upcoming_meetings function.")

# -------------------- Function to Monitor Meetings --------------------
def monitor_meetings(session, processed_videos, processed_videos_file, config, webpage_url, interval=60):
    logging.info("Entering monitor_meetings function.")
    """
    Continuously monitors for upcoming meetings and initiates recordings.
    For each meeting, it checks for the "In Progress" URL every minute for up to 10 minutes.
    If the URL is found within this period, it proceeds with recording.
    Otherwise, it snoozes until the next meeting.
    
    Args:
        session: SQLAlchemy session object.
        processed_videos (list): List of processed meeting IDs.
        processed_videos_file (str): Path to the processed videos JSON file.
        config (dict): Configuration dictionary.
        webpage_url (str): URL of the webpage to scrape for the "In Progress" video URL.
        interval (int, optional): Fallback interval in seconds if no meetings are found. Defaults to 60.
    """
    try:
        while True:
            logging.info("Fetching upcoming meetings.")
            upcoming_meetings = get_upcoming_meetings(session, limit=10)
            display_upcoming_meetings(upcoming_meetings)

            # Filter out already processed meetings
            unprocessed_meetings = [
                meeting for meeting in upcoming_meetings if meeting.meetingid not in processed_videos
            ]
            logging.debug(f"Unprocessed meetings count: {len(unprocessed_meetings)}")

            if not unprocessed_meetings:
                logging.info("No unprocessed upcoming meetings found. Sleeping for fallback interval.")
                time.sleep(interval)
                continue

            # Sort meetings by datetime
            unprocessed_meetings.sort(key=lambda m: m.datetime)
            next_meeting = unprocessed_meetings[0]
            time_until_meeting = (next_meeting.datetime - datetime.now()).total_seconds()
            buffer_time = 60  # Start recording 1 minute before the meeting starts
            sleep_duration = max(time_until_meeting - buffer_time, 0)

            logging.debug(f"Next meeting ID {next_meeting.meetingid} starts in {time_until_meeting} seconds.")
            logging.info(f"Sleeping for {sleep_duration} seconds until recording meeting ID {next_meeting.meetingid}.")

            if sleep_duration > 0:
                sleep_until = datetime.now() + timedelta(seconds=sleep_duration)
                logging.info(f"Sleeping until {sleep_until.strftime('%Y-%m-%d %H:%M:%S')} (in {sleep_duration} seconds).")
                time.sleep(sleep_duration)

            # After waking up, verify if the meeting is still upcoming and not processed
            current_time = datetime.now()
            if next_meeting.datetime - timedelta(seconds=buffer_time) > current_time:
                logging.warning(f"Meeting ID {next_meeting.meetingid} is no longer starting soon. Skipping.")
                continue

            # Initiate recording after verifying "In Progress" URL
            logging.info(f"Initiating check for 'In Progress' video URL for meeting '{next_meeting.name}' (ID: {next_meeting.meetingid}).")

            in_progress_found = False
            check_attempts = 10  # Number of minutes to check
            check_interval = 60  # Seconds between checks

            for attempt in range(1, check_attempts + 1):
                logging.info(f"Check attempt {attempt} for 'In Progress' URL.")
                in_progress_video_url = get_in_progress_video_url(webpage_url)
                
                if in_progress_video_url:
                    logging.info(f"'In Progress' URL found for meeting ID {next_meeting.meetingid}: {in_progress_video_url}")
                    in_progress_found = True
                    break
                else:
                    logging.warning(f"'In Progress' URL not found for meeting ID {next_meeting.meetingid}. Attempt {attempt} of {check_attempts}.")
                    if attempt < check_attempts:
                        logging.info(f"Waiting for {check_interval} seconds before next check.")
                        time.sleep(check_interval)

            if in_progress_found:
                # Proceed with recording
                logging.info(f"Proceeding with recording for meeting ID {next_meeting.meetingid}.")

                # Generate filename using meetingid
                output_file = generate_filename_from_meetingid(config['video']['output_directory'], next_meeting.meetingid)

                # Create a stop_event to signal termination after 10 minutes
                stop_event = threading.Event()

                # Start video recording in a separate thread
                recording_thread = threading.Thread(
                    target=save_video_stream,
                    args=(
                        in_progress_video_url,  # Use the scraped video URL
                        output_file,
                        config['video']['max_duration'],
                        config['video']['silence_duration'],
                        stop_event
                    ),
                    daemon=True,
                    name=f"RecordingThread-{next_meeting.meetingid}"
                )
                recording_thread.start()
                logging.info(f"Recording thread started for meeting ID: {next_meeting.meetingid}")

                # Add meeting ID to processed videos
                processed_videos.append(next_meeting.meetingid)
                save_processed_videos(processed_videos_file, processed_videos)
                logging.info(f"Meeting ID {next_meeting.meetingid} added to processed videos.")

                # Monitor the recording for 10 minutes, checking every minute
                monitoring_duration = 10  # in minutes
                monitoring_interval = 60   # in seconds

                logging.info(f"Monitoring recording for meeting ID {next_meeting.meetingid} for the next {monitoring_duration} minutes.")

                for minute in range(1, monitoring_duration + 1):
                    logging.info(f"Monitoring minute {minute} of {monitoring_duration} for meeting ID {next_meeting.meetingid}.")
                    time.sleep(monitoring_interval)
                    if not recording_thread.is_alive():
                        logging.warning(f"Recording thread for meeting ID {next_meeting.meetingid} has stopped unexpectedly.")
                        break

                # After 10 minutes, signal the recording to stop if it's still running
                if recording_thread.is_alive():
                    logging.info(f"10-minute monitoring period ended. Terminating recording for meeting ID {next_meeting.meetingid}.")
                    stop_event.set()  # Signal the recording thread to terminate
                    recording_thread.join()
                    logging.info(f"Recording for meeting ID {next_meeting.meetingid} has been terminated.")
            else:
                logging.warning(f"Failed to find 'In Progress' URL for meeting ID {next_meeting.meetingid} after {check_attempts} attempts. Snoozing until next meeting.")
                # Optionally, mark as processed to avoid retrying
                processed_videos.append(next_meeting.meetingid)
                save_processed_videos(processed_videos_file, processed_videos)
                continue

            # Proceed to the next meeting
            logging.info(f"Finished handling meeting ID {next_meeting.meetingid}. Proceeding to the next meeting.")

    except Exception as e:
        logging.error(f"Error during meeting monitoring: {e}")
        logging.debug(traceback.format_exc())
    finally:
        logging.info("Exiting monitor_meetings function.")

# -------------------- Main Function --------------------
def main():
    logging.info("Entering main function.")
    try:
        # Load configuration
        config = load_config()

        # Validate configuration
        video_config = config.get('video', {})
        if not video_config:
            logging.error("No 'video' configuration found in config file.")
            raise ValueError("Missing 'video' configuration.")
        
        database_config = config.get('database', {})
        if not database_config:
            logging.error("No 'database' configuration found in config file.")
            raise ValueError("Missing 'database' configuration.")
        
        # Setup database
        logging.debug("Attempting to setup database connection.")
        session = setup_database(database_config['DATABASE_URL'])
        logging.debug("Database session established.")
    
        # Extract video configuration
        video_output_dir = video_config.get('video_folder')  # Ensure consistency in config field names
        max_duration = video_config.get('max_duration', 10800)       # Default to 3 hours
        silence_duration = video_config.get('silence_duration', 300) # Default to 5 minutes
        processed_videos_file = video_config.get('processed_videos_file', 'processed_videos.json')
        webpage_url = video_config.get('stream_url')  # Assuming 'stream_url' is the webpage URL to scrape

        if not all([video_output_dir, webpage_url]):
            logging.error("Incomplete 'video' configuration. 'output_directory' and 'stream_url' are required.")
            raise ValueError("Incomplete 'video' configuration.")
    
        logging.debug(f"Video Output Directory: {video_output_dir}")
        logging.debug(f"Stream URL: {webpage_url}")
        logging.debug(f"Max Duration: {max_duration} seconds")
        logging.debug(f"Silence Duration: {silence_duration} seconds")
        logging.debug(f"Processed Videos File: {processed_videos_file}")
    
        # Ensure videos directory exists
        ensure_videos_directory(video_output_dir)
    
        # Load processed videos
        logging.debug(f"Loading processed videos from '{processed_videos_file}'.")
        processed_videos = load_processed_videos(processed_videos_file)
        logging.debug(f"Current processed videos: {processed_videos}")

        # Display upcoming meetings at startup
        logging.info("Fetching and displaying upcoming meetings at startup.")
        upcoming_meetings = get_upcoming_meetings(session, limit=10)
        display_upcoming_meetings(upcoming_meetings)

        # Start monitoring meetings
        logging.info("Starting to monitor meetings.")
        monitor_meetings(
            session,
            processed_videos,
            processed_videos_file,
            config,
            webpage_url,  # The webpage URL to scrape for "In Progress" link
            interval=60    # Fallback sleep interval in seconds
        )
    except Exception as e:
        logging.error(f"An unexpected error occurred in main: {e}")
        logging.debug(traceback.format_exc())
    finally:
        logging.info("Exiting main function.")
        logging.info("Script has been terminated.")

if __name__ == "__main__":
    main()