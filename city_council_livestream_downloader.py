#city_council_livestream_downloader

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
import city_council_video_transcriber

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# -------------------- Setup Logging --------------------
# Remove any existing handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup Rotating File Handler
rotating_handler = RotatingFileHandler(
    'city_council_livestream_download.log', maxBytes=5*1024*1024, backupCount=5
)  # 5 MB per file, 5 backups
rotating_handler.setLevel(logging.DEBUG)  # Capture all DEBUG and above logs
rotating_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
rotating_handler.setFormatter(rotating_formatter)
logging.getLogger('').addHandler(rotating_handler)

# Setup Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # Capture INFO and above logs to console
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger('').addHandler(console_handler)

# **Set the root logger's level to DEBUG**
logging.getLogger('').setLevel(logging.DEBUG)

logging.info("Logging is set up successfully.")

# -------------------- Signal Handlers --------------------
def signal_handler(sig, frame):
    logging.info("Shutdown signal received. Terminating script.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)    # Handle Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)   # Handle termination signals

# -------------------- Load Configuration --------------------
def load_config(config_file='config.json'):
    logger = logging.getLogger('load_config')
    logger.info("Entering load_config function.")
    try:
        logger.debug(f"Attempting to open configuration file: {config_file}")
        with open(config_file, 'r') as file:
            config = json.load(file)
        logger.info("Configuration loaded successfully.")
        logger.debug(f"Configuration contents: {config}")
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
        logger.info("Exiting load_config function.")

# -------------------- Database Setup --------------------
def setup_database(db_uri):
    logger = logging.getLogger('setup_database')
    logger.info("Entering setup_database function.")
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
        logger.info("Exiting setup_database function.")

# -------------------- Ensure Videos Directory Exists --------------------
def ensure_videos_directory(output_dir):
    logger = logging.getLogger('ensure_videos_directory')
    logger.info("Entering ensure_videos_directory function.")
    try:
        logger.debug(f"Checking if directory '{output_dir}' exists.")
        if not os.path.exists(output_dir):
            logger.debug(f"Directory '{output_dir}' does not exist. Creating it.")
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")
        else:
            logger.info(f"Directory already exists: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create directory '{output_dir}': {e}")
        logger.debug(traceback.format_exc())
        raise
    finally:
        logger.info("Exiting ensure_videos_directory function.")

# -------------------- Load Processed Events --------------------
def load_processed_events(processed_events_file):
    logger = logging.getLogger('load_processed_events')
    logger.info("Entering load_processed_events function.")
    try:
        logger.debug(f"Checking if processed events file '{processed_events_file}' exists.")
        if os.path.exists(processed_events_file):
            logger.debug(f"Found processed events file: {processed_events_file}. Loading it.")
            with open(processed_events_file, 'r') as file:
                processed_events = json.load(file)
            logger.info(f"Loaded {len(processed_events)} processed events.")
        else:
            logger.info("No processed events file found. Starting fresh.")
            processed_events = []
        return processed_events
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in processed events file: {e}")
        logger.debug(traceback.format_exc())
        return []
    except Exception as e:
        logger.error(f"Failed to load processed events: {e}")
        logger.debug(traceback.format_exc())
        return []
    finally:
        logger.info("Exiting load_processed_events function.")

# -------------------- Save Processed Events --------------------
def save_processed_events(processed_events_file, processed_events):
    logger = logging.getLogger('save_processed_events')
    logger.info("Entering save_processed_events function.")
    try:
        logger.debug(f"Saving {len(processed_events)} processed events to '{processed_events_file}'.")
        with open(processed_events_file, 'w') as file:
            json.dump(processed_events, file, indent=4)
        logger.info(f"Saved {len(processed_events)} processed events successfully.")
    except Exception as e:
        logger.error(f"Failed to save processed events: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("Exiting save_processed_events function.")

# -------------------- Load Processed Videos --------------------
def load_processed_videos(processed_videos_file):
    logger = logging.getLogger('load_processed_videos')
    logger.info("Entering load_processed_videos function.")
    try:
        logger.debug(f"Checking if processed videos file '{processed_videos_file}' exists.")
        if os.path.exists(processed_videos_file):
            logger.debug(f"Found processed videos file: {processed_videos_file}. Loading it.")
            with open(processed_videos_file, 'r') as file:
                processed_videos = json.load(file)
            logger.info(f"Loaded {len(processed_videos)} processed videos.")
        else:
            logger.info("No processed videos file found. Starting fresh.")
            processed_videos = []
        return processed_videos
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in processed videos file: {e}")
        logger.debug(traceback.format_exc())
        return []
    except Exception as e:
        logger.error(f"Failed to load processed videos: {e}")
        logger.debug(traceback.format_exc())
        return []
    finally:
        logger.info("Exiting load_processed_videos function.")

# -------------------- Save Processed Videos --------------------
def save_processed_videos(processed_videos_file, processed_videos):
    logger = logging.getLogger('save_processed_videos')
    logger.info("Entering save_processed_videos function.")
    try:
        logger.debug(f"Saving {len(processed_videos)} processed videos to '{processed_videos_file}'.")
        with open(processed_videos_file, 'w') as file:
            json.dump(processed_videos, file, indent=4)
        logger.info(f"Saved {len(processed_videos)} processed videos successfully.")
    except Exception as e:
        logger.error(f"Failed to save processed videos: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("Exiting save_processed_videos function.")

# -------------------- Generate Filename from Meeting ID --------------------
def generate_filename_from_meetingid(output_dir, meeting_id):
    logger = logging.getLogger('generate_filename_from_meetingid')
    logger.info("Entering generate_filename_from_meetingid function.")
    """
    Generates a filename using the meeting ID.
    Format: meeting_<meetingid>.mp4
    """
    filename = f"meeting_{meeting_id}.mp4"
    filepath = os.path.join(output_dir, filename)
    logger.debug(f"Generated filename: {filepath}")
    logger.info("Exiting generate_filename_from_meetingid function.")
    return filepath

# -------------------- Scrape "In Progress" Video URL --------------------

def get_in_progress_video_url(webpage_url, browser='chrome'):
    """
    Scrapes the specified webpage to find the video URL with link text "In Progress".
    It interacts with the page by clicking the "List View" link to load the necessary HTML content.

    Args:
        webpage_url (str): The URL of the webpage to scrape.
        browser (str): The browser to use ('chrome' or 'firefox').

    Returns:
        str or None: The extracted video URL if found; otherwise, None.
    """
    logger = logging.getLogger('get_in_progress_video_url')
    logger.info("Entering get_in_progress_video_url function.")
    
    # Configure Selenium WebDriver options
    if browser.lower() == 'chrome':
        options = ChromeOptions()
        options.add_argument('--headless')  # Run in headless mode
        options.add_argument('--disable-gpu')
        driver = webdriver.Chrome(options=options)
    elif browser.lower() == 'firefox':
        options = FirefoxOptions()
        options.add_argument('--headless')  # Run in headless mode
        driver = webdriver.Firefox(options=options)
    else:
        logger.error(f"Unsupported browser: {browser}. Choose 'chrome' or 'firefox'.")
        return None
    
    try:
        logger.debug(f"Navigating to '{webpage_url}'.")
        driver.get(webpage_url)
        logger.info(f"Page loaded successfully: {webpage_url}")
        
        # Click the "List View" link/button to load all meetings
        try:
            # Update the selector based on the actual "List View" link/button's attributes
            # For example, if it has an ID or a specific class
            list_view_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.LINK_TEXT, 'List View'))
            )
            list_view_button.click()
            logger.info("Clicked the 'List View' button successfully.")
        except TimeoutException:
            logger.warning("List View button not found or not clickable within the timeout period.")
            # If there's no List View, proceed without clicking
        except NoSuchElementException:
            logger.warning("List View button not found on the page.")
            # If there's no List View, proceed without clicking

        # Wait for the page to load updated content after clicking
        #time.sleep(10)  # Adjust sleep time as needed or use explicit waits

        # Alternatively, wait for a specific element that appears after clicking "List View"
        # Example:
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, 'videolink')))

        # Now, parse the page source with BeautifulSoup
        logger.debug("Parsing updated HTML content with BeautifulSoup.")
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find all <a> tags with exact text "In Progress"
        logger.debug("Searching for '<a>' tags with text 'In Progress'.")
        in_progress_links = soup.find_all('a', class_='videolink', string=re.compile(r'In\s*Progress', re.IGNORECASE))
        
        if not in_progress_links:
            logger.warning(f"No 'In Progress' links found on the webpage: {webpage_url}")
            
            # For debugging: Log all <a> tags' texts
            all_links = soup.find_all('a')
            logger.debug("Listing all '<a>' tag texts for debugging:")
            for link in all_links:
                link_text = link.get_text(strip=True)
                logger.debug(f"Link Text: '{link_text}' | Href: '{link.get('href')}'")
            
            return None
        
        # Assuming the first "In Progress" link is the desired one
        in_progress_link = in_progress_links[0]
        video_url = in_progress_link.get('href')
        logger.info(f"Extracted 'In Progress' link href: {video_url}")
        stream_url = city_council_video_transcriber.fetch_playlist_url(video_url, config['video']['chromedriver_path'])

        # If the href is relative, make it absolute
        if video_url and not urllib.parse.urlparse(video_url).scheme:
            video_url = urllib.parse.urljoin(webpage_url, video_url)
            logger.info(f"Converted relative video URL to absolute: {video_url}")
        else:
            logger.info(f"Video URL is already absolute: {video_url}")
        
        return stream_url
    except Exception as e:
        logger.error(f"Failed to scrape 'In Progress' video URL: {e}")
        logger.debug(traceback.format_exc())
        return None
    finally:
        driver.quit()
        logger.info("Exiting get_in_progress_video_url function and closing the browser.")

# -------------------- Function to Save Video Stream --------------------
def save_video_stream(stream_url, output_file, max_duration, silence_duration, stop_event):
    logger = logging.getLogger('save_video_stream')
    logger.info("Entering save_video_stream function.")
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
    logger.info(f"Starting video capture for file: {output_file}")
    
    # Remove existing output file to prevent conflicts
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
            logger.info(f"Existing file '{output_file}' removed.")
        except Exception as e:
            logger.error(f"Failed to remove existing file '{output_file}': {e}")
            return
    
    # FFmpeg command with adjusted codecs
    command = [
        'ffmpeg',
        '-i', stream_url,                                      # Input stream URL
        '-c:v', 'copy',                                        # Copy video stream without re-encoding
        '-c:a', 'aac',                                         # Re-encode audio stream to AAC
        '-f', 'mp4',                                           # Output format
        '-y',                                                  # Overwrite output file without asking
        '-af', f'silencedetect=n=-50dB:d={silence_duration}',  # Audio filter for silence detection
        '-loglevel', 'info',                                   # Set log level to info
        output_file
    ]
    logger.debug(f"FFmpeg command: {' '.join(command)}")
    
    try:
        logger.debug("Starting FFmpeg subprocess.")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        logger.info("FFmpeg process started successfully.")
    except FileNotFoundError:
        logger.error("FFmpeg executable not found. Ensure FFmpeg is installed and in the system PATH.")
        return
    except Exception as e:
        logger.error(f"Failed to start FFmpeg: {e}")
        logger.debug(traceback.format_exc())
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
                logger.debug(f"FFmpeg stderr: {line.strip()}")
                
                # Detect silence start
                match_start = silence_start_re.search(line)
                if match_start:
                    silence_start = float(match_start.group(1))
                    logger.info(f"Silence started at {silence_start} seconds.")

                # Detect silence end
                match_end = silence_end_re.search(line)
                if match_end and silence_start is not None:
                    silence_duration_detected = float(match_end.group(2))
                    logger.info(f"Silence ended at {match_end.group(1)} seconds, duration: {silence_duration_detected} seconds.")
                    if silence_duration_detected >= silence_duration:
                        logger.info(f"Detected {silence_duration_detected} seconds of silence. Terminating FFmpeg.")
                        silence_detected = True
                        process.terminate()
                        break
        except Exception as e:
            logger.error(f"Error while monitoring FFmpeg stderr: {e}")
            logger.debug(traceback.format_exc())
            process.terminate()
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitor_ffmpeg, name="FFmpegMonitorThread")
    monitor_thread.start()
    logger.info("Started FFmpeg monitoring thread.")
    
    # Monitor for max_duration, silence, or stop_event
    try:
        while True:
            elapsed_time = time.time() - start_time
            logger.debug(f"Elapsed time: {elapsed_time:.2f} seconds.")
            
            if stop_event.is_set():
                logger.info("Stop event received. Terminating FFmpeg.")
                process.terminate()
                break
            if elapsed_time >= max_duration:
                logger.info(f"Reached maximum duration of {max_duration} seconds. Terminating FFmpeg.")
                process.terminate()
                break
            if silence_detected:
                logger.info("Silence detected. FFmpeg termination initiated.")
                break
            if process.poll() is not None:
                logger.warning("FFmpeg process terminated unexpectedly.")
                break
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error during FFmpeg monitoring loop: {e}")
        logger.debug(traceback.format_exc())
        process.terminate()
    finally:
        monitor_thread.join()
        logger.info("FFmpeg monitoring thread has ended.")
    
    # Wait for the process to terminate
    try:
        logger.debug("Waiting for FFmpeg process to terminate.")
        stdout, stderr = process.communicate(timeout=10)
        if process.returncode != 0:
            logger.error(f"FFmpeg terminated with return code {process.returncode}.")
            logger.error(f"FFmpeg stderr: {stderr}")
        else:
            logger.info("FFmpeg process completed successfully.")
    except subprocess.TimeoutExpired:
        logger.error("FFmpeg did not terminate in time. Killing the process.")
        process.kill()
        stdout, stderr = process.communicate()
        logger.debug(f"FFmpeg killed. stderr: {stderr}")
    
    logger.info(f"Video capture for file '{output_file}' has ended.")
    logger.info("Exiting save_video_stream function.")

# -------------------- Function to Fetch Upcoming Meetings --------------------
def get_upcoming_meetings(session, limit=10):
    logger = logging.getLogger('get_upcoming_meetings')
    logger.info("Entering get_upcoming_meetings function.")
    """
    Retrieves a list of upcoming meetings ordered by datetime.
    """
    current_time = datetime.now()
    try:
        logger.debug(f"Fetching meetings with datetime >= {current_time}. Limit: {limit}")
        meetings = session.query(Meeting).filter(Meeting.datetime >= current_time).order_by(Meeting.datetime).limit(limit).all()
        logger.info(f"Fetched {len(meetings)} upcoming meetings.")
        logger.debug(f"Upcoming meetings IDs: {[meeting.meetingid for meeting in meetings]}")
        return meetings
    except Exception as e:
        logger.error(f"Failed to fetch upcoming meetings: {e}")
        logger.debug(traceback.format_exc())
        session.rollback()  # Ensure session is rolled back
        return []
    finally:
        logger.info("Exiting get_upcoming_meetings function.")

def display_upcoming_meetings(meetings):
    logger = logging.getLogger('display_upcoming_meetings')
    logger.info("Entering display_upcoming_meetings function.")
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
            logger.debug(f"Displayed meeting: ID={meeting.meetingid}, Name={meeting.name}")
    except Exception as e:
        logger.error(f"Failed to display upcoming meetings: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("Exiting display_upcoming_meetings function.")

# -------------------- Function to Monitor Meetings --------------------
def monitor_meetings(session, processed_videos, processed_videos_file, config, webpage_url, interval=60):
    logger = logging.getLogger('monitor_meetings')
    logger.info("Entering monitor_meetings function.")
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
            logger.info("Fetching upcoming meetings.")
            upcoming_meetings = get_upcoming_meetings(session, limit=10)
            display_upcoming_meetings(upcoming_meetings)

            # Filter out already processed meetings
            unprocessed_meetings = [
                meeting for meeting in upcoming_meetings if meeting.meetingid not in processed_videos
            ]
            logger.debug(f"Unprocessed meetings count: {len(unprocessed_meetings)}")

            if not unprocessed_meetings:
                logger.info("No unprocessed upcoming meetings found. Sleeping for fallback interval.")
                logger.debug(f"Sleeping for {interval} seconds.")
                time.sleep(interval)
                continue

            # Sort meetings by datetime
            unprocessed_meetings.sort(key=lambda m: m.datetime)
            next_meeting = unprocessed_meetings[0]
            time_until_meeting = (next_meeting.datetime - datetime.now()).total_seconds()
            buffer_time = 60  # Start recording 1 minute before the meeting starts
            sleep_duration = max(time_until_meeting - buffer_time, 0)

            logger.debug(f"Next meeting ID {next_meeting.meetingid} starts in {time_until_meeting} seconds.")
            logger.info(f"Sleeping for {sleep_duration} seconds until recording meeting ID {next_meeting.meetingid}.")
            logger.debug(f"Sleeping until {datetime.now() + timedelta(seconds=sleep_duration)}.")

            if sleep_duration > 0:
                time.sleep(sleep_duration)

            # After waking up, verify if the meeting is still upcoming and not processed
            current_time = datetime.now()
            if next_meeting.datetime - timedelta(seconds=buffer_time) > current_time:
                logger.warning(f"Meeting ID {next_meeting.meetingid} is no longer starting soon. Skipping.")
                continue

            # Initiate recording after verifying "In Progress" URL
            logger.info(f"Initiating check for 'In Progress' video URL for meeting '{next_meeting.name}' (ID: {next_meeting.meetingid}).")

            in_progress_found = False
            check_attempts = 10  # Number of minutes to check
            check_interval = 60  # Seconds between checks

            for attempt in range(1, check_attempts + 1):
                logger.info(f"Check attempt {attempt} for 'In Progress' URL.")
                in_progress_video_url = get_in_progress_video_url(webpage_url)
                
                if in_progress_video_url:
                    logger.info(f"'In Progress' URL found for meeting ID {next_meeting.meetingid}: {in_progress_video_url}")
                    in_progress_found = True
                    break
                else:
                    logger.warning(f"'In Progress' URL not found for meeting ID {next_meeting.meetingid}. Attempt {attempt} of {check_attempts}.")
                    if attempt < check_attempts:
                        logger.info(f"Waiting for {check_interval} seconds before next check.")
                        time.sleep(check_interval)

            if in_progress_found:
                # Proceed with recording
                logger.info(f"Proceeding with recording for meeting ID {next_meeting.meetingid}.")

                # Generate filename using meetingid
                output_file = generate_filename_from_meetingid(config['video']['output_directory'], next_meeting.meetingid)

                # Create a stop_event to signal termination after max_duration
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
                logger.info(f"Recording thread started for meeting ID: {next_meeting.meetingid}")

                # Add meeting ID to processed videos
                processed_videos.append(next_meeting.meetingid)
                save_processed_videos(processed_videos_file, processed_videos)
                logger.info(f"Meeting ID {next_meeting.meetingid} added to processed videos.")

                # Monitor the recording for max_duration, checking every minute
                monitoring_duration = config['video']['max_duration'] // 60  # in minutes
                monitoring_interval = 60   # in seconds

                logger.info(f"Monitoring recording for meeting ID {next_meeting.meetingid} for the next {monitoring_duration} minutes.")

                for minute in range(1, monitoring_duration + 1):
                    logger.info(f"Monitoring minute {minute} of {monitoring_duration} for meeting ID {next_meeting.meetingid}.")
                    time.sleep(monitoring_interval)
                    if not recording_thread.is_alive():
                        logger.warning(f"Recording thread for meeting ID {next_meeting.meetingid} has stopped unexpectedly.")
                        break

                # After max_duration, signal the recording to stop if it's still running
                if recording_thread.is_alive():
                    logger.info(f"Max monitoring period ended. Terminating recording for meeting ID {next_meeting.meetingid}.")
                    stop_event.set()  # Signal the recording thread to terminate
                    recording_thread.join()
                    logger.info(f"Recording for meeting ID {next_meeting.meetingid} has been terminated.")
            else:
                logger.warning(f"Failed to find 'In Progress' URL for meeting ID {next_meeting.meetingid} after {check_attempts} attempts. Snoozing until next meeting.")
                # Optionally, mark as processed to avoid retrying
                processed_videos.append(next_meeting.meetingid)
                save_processed_videos(processed_videos_file, processed_videos)
                continue

            # Proceed to the next meeting
            logger.info(f"Finished handling meeting ID {next_meeting.meetingid}. Proceeding to the next meeting.")

    except Exception as e:
        logger.error(f"Error during meeting monitoring: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("Exiting monitor_meetings function.")

# -------------------- Main Function --------------------
def main():
    logger = logging.getLogger('main')
    logger.info("Entering main function.")
    try:
        # Load configuration
        config = load_config()

        # Validate configuration
        video_config = config.get('video', {})
        if not video_config:
            logger.error("No 'video' configuration found in config file.")
            raise ValueError("Missing 'video' configuration.")
        
        database_config = config.get('database', {})
        if not database_config:
            logger.error("No 'database' configuration found in config file.")
            raise ValueError("Missing 'database' configuration.")
        
        # Setup database
        logger.debug("Attempting to setup database connection.")
        session = setup_database(database_config['DATABASE_URL'])
        logger.debug("Database session established.")
    
        # Extract video configuration
        video_output_dir = video_config.get('video_folder')  # Ensure consistency in config field names
        max_duration = video_config.get('max_duration', 10800)       # Default to 3 hours
        silence_duration = video_config.get('silence_duration', 300) # Default to 5 minutes
        processed_videos_file = video_config.get('processed_videos_file', 'processed_videos.json')
        webpage_url = video_config.get('stream_url')  # Assuming 'stream_url' is the webpage URL to scrape

        if not all([video_output_dir, webpage_url]):
            logger.error("Incomplete 'video' configuration. 'output_directory' and 'stream_url' are required.")
            raise ValueError("Incomplete 'video' configuration.")
    
        logger.debug(f"Video Output Directory: {video_output_dir}")
        logger.debug(f"Stream URL: {webpage_url}")
        logger.debug(f"Max Duration: {max_duration} seconds")
        logger.debug(f"Silence Duration: {silence_duration} seconds")
        logger.debug(f"Processed Videos File: {processed_videos_file}")
    
        # Ensure videos directory exists
        ensure_videos_directory(video_output_dir)
    
        # Load processed videos
        logger.debug(f"Loading processed videos from '{processed_videos_file}'.")
        processed_videos = load_processed_videos(processed_videos_file)
        logger.debug(f"Current processed videos: {processed_videos}")

        # Display upcoming meetings at startup
        logger.info("Fetching and displaying upcoming meetings at startup.")
        upcoming_meetings = get_upcoming_meetings(session, limit=10)
        display_upcoming_meetings(upcoming_meetings)

        # Start monitoring meetings
        logger.info("Starting to monitor meetings.")
        monitor_meetings(
            session,
            processed_videos,
            processed_videos_file,
            config,
            webpage_url,  # The webpage URL to scrape for "In Progress" link
            interval=60    # Fallback sleep interval in seconds
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}")
        logger.debug(traceback.format_exc())
    finally:
        logger.info("Exiting main function.")
        logger.info("Script has been terminated.")

if __name__ == "__main__":
    main()