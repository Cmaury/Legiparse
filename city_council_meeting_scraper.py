# city_council_meeting_scraper.py

import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json
import time
from datetime import datetime
import traceback
import signal
import sys
import threading
import re

import agenda_processor
import llm_processor
import city_council_video_transcriber

from database import SessionLocal  # Ensure this import is present
from models import Meeting  # Import Meeting model for database queries

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler("city_council_meeting_scraper.log"),
        logging.StreamHandler()
    ]
)

# Initialize a shutdown event for graceful termination
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """
    Handles interrupt signals to allow graceful shutdown.
    Sets the shutdown_event to signal threads to terminate.
    """
    logging.info(f"Received signal {signum}. Shutting down gracefully.")
    shutdown_event.set()

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def process_agenda(meeting_url):
    """
    Processes a single agenda item by saving its data to the database.
    
    Args:
        meeting_url (str): The URL of the meeting to process.
    """
    agenda_processor.save_meeting_data_to_db(meeting_url)

def load_configuration():
    """
    Loads configuration parameters from 'config.json'.
    
    Returns:
        dict: Configuration parameters.
    """
    logging.debug("Entering load_configuration()")
    try:
        with open('config.json') as config_file:
            config = json.load(config_file)
        logging.info("Configuration loaded successfully.")
        logging.debug("Exiting load_configuration()")
        return config
    except FileNotFoundError:
        logging.error("Configuration file 'config.json' not found.")
        logging.debug("Exiting load_configuration() due to FileNotFoundError")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing 'config.json': {e}")
        logging.debug("Exiting load_configuration() due to JSONDecodeError")
        sys.exit(1)

# Initialize global variables
calendar_page_url = "https://pittsburgh.legistar.com/Calendar.aspx"  # Default URL
delay_between_requests = 1  # Seconds

def setup_requests_session():
    """
    Sets up a requests session with custom headers and retry strategy.
    
    Returns:
        requests.Session: Configured HTTP session.
    """
    logging.debug("Entering setup_requests_session()")
    # Define custom headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                      'AppleWebKit/537.36 (KHTML, like Gecko) ' \
                      'Chrome/58.0.3029.110 Safari/537.3'
    }

    # Define retry strategy with exponential backoff
    retry_strategy = requests.adapters.Retry(
        total=5,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # HTTP methods to retry
        backoff_factor=1  # Exponential backoff factor (1s, 2s, 4s, 8s, 16s)
    )

    # Create an HTTP adapter with the retry strategy
    adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)

    # Create a requests session and mount the adapter
    http = requests.Session()
    http.headers.update(headers)
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    logging.debug("Exiting setup_requests_session()")
    return http

def fetch_calendar_page(http_session, url):
    """
    Fetches the main calendar page content.
    
    Args:
        http_session (requests.Session): Configured HTTP session.
        url (str): URL of the calendar page.
    
    Returns:
        BeautifulSoup object or None if fetching fails.
    """
    try:
        logging.info(f"Fetching calendar page from: {url}")
        response = http_session.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        logging.info("Successfully fetched and parsed calendar page.")
        return soup
    except requests.RequestException as e:
        logging.error(f"Error fetching calendar page: {e}")
        logging.debug(traceback.format_exc())
        return None

def extract_meetings(soup):
    """
    Extracts meeting detail URLs and their corresponding datetime from the calendar page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML of the calendar page.
    
    Returns:
        List of tuples: [(meeting_url, meeting_datetime), ...]
    """
    meetings = []
    try:
        # Find all <a> tags with id containing 'hypMeetingDetail'
        meeting_links = soup.find_all('a', id=re.compile('hypMeetingDetail'))
        logging.info(f"Found {len(meeting_links)} meeting detail links.")

        for link in meeting_links:
            if shutdown_event.is_set():
                logging.info("Shutdown signal received. Stopping meeting extraction.")
                break

            # Extract href and construct absolute URL
            relative_url = link.get('href')
            if not relative_url:
                logging.debug("No href found in meeting link. Skipping.")
                continue
            meeting_url = urljoin(calendar_page_url, relative_url)

            # Find the parent <tr> to locate the datetime
            parent_tr = link.find_parent('tr')
            if not parent_tr:
                logging.debug("No parent <tr> found for meeting link. Skipping.")
                continue

            # Extract all <td> in the row
            tds = parent_tr.find_all('td')
            if not tds or len(tds) < 2:
                logging.debug("Not enough <td> elements in row to extract datetime. Skipping.")
                continue

            # Assuming that the datetime is in the first or second <td>
            # Adjust the index based on the actual HTML structure
            datetime_str = tds[0].get_text(strip=True)  # Example: first <td> has date
            meeting_datetime = parse_meeting_datetime(datetime_str)
            if not meeting_datetime:
                logging.debug(f"Failed to parse datetime '{datetime_str}'. Skipping.")
                continue

            meetings.append((meeting_url, meeting_datetime))
            logging.debug(f"Extracted meeting: URL={meeting_url}, datetime={meeting_datetime}")

        logging.info(f"Extracted {len(meetings)} meetings from the calendar page.")
        return meetings
    except Exception as e:
        logging.error(f"Error extracting meetings: {e}")
        logging.debug(traceback.format_exc())
        return meetings

def parse_meeting_datetime(datetime_str):
    """
    Parses a datetime string into a datetime object.
    
    Args:
        datetime_str (str): Datetime string (e.g., '09/18/2024 10:00 AM').
    
    Returns:
        datetime or None: Parsed datetime object or None if parsing fails.
    """
    try:
        # Define possible datetime formats based on the website's format
        datetime_formats = [
            '%m/%d/%Y %I:%M %p',  # e.g., '09/18/2024 10:00 AM'
            '%m-%d-%Y %I:%M %p',  # e.g., '09-18-2024 10:00 AM'
            '%B %d, %Y %I:%M %p', # e.g., 'September 18, 2024 10:00 AM'
            '%d %B %Y %I:%M %p',  # e.g., '18 September 2024 10:00 AM'
            '%Y-%m-%d %H:%M',     # e.g., '2024-09-18 10:00'
            '%m/%d/%Y',            # e.g., '09/18/2024' without time
            '%m-%d-%Y',            # e.g., '09-18-2024' without time
            '%B %d, %Y',           # e.g., 'September 18, 2024' without time
            '%d %B %Y',            # e.g., '18 September 2024' without time
            '%Y-%m-%d',            # e.g., '2024-09-18' without time
        ]

        for fmt in datetime_formats:
            try:
                parsed_datetime = datetime.strptime(datetime_str, fmt)
                logging.debug(f"Successfully parsed datetime '{datetime_str}' with format '{fmt}'.")
                return parsed_datetime
            except ValueError:
                continue  # Try next format

        logging.error(f"Failed to parse datetime string: '{datetime_str}'. No matching format found.")
        return None

    except Exception as e:
        logging.error(f"Unexpected error while parsing datetime: {e}")
        logging.debug(traceback.format_exc())
        return None

def process_calendar_page(http_session, url):
    """
    Processes the main calendar page: fetches, parses, extracts meetings, and processes them.
    
    Args:
        http_session (requests.Session): Configured HTTP session.
        url (str): URL of the calendar page.
    """
    soup = fetch_calendar_page(http_session, url)
    if not soup:
        logging.error("Failed to fetch or parse the calendar page. Exiting processing.")
        return

    meetings = extract_meetings(soup)
    if not meetings:
        logging.info("No meetings found to process.")
        return

    now = datetime.now()
    for meeting_url, meeting_datetime in meetings:
        if shutdown_event.is_set():
            logging.info("Shutdown signal received. Stopping meeting processing.")
            break

        if meeting_datetime >= now:
            logging.info(f"Processing future meeting scheduled on {meeting_datetime} - URL: {meeting_url}")
            try:
                process_agenda(meeting_url)
                logging.info(f"Successfully processed meeting: {meeting_url}")
            except Exception as e:
                logging.error(f"Error processing meeting {meeting_url}: {e}")
                logging.debug(traceback.format_exc())
        else:
            logging.info(f"Skipping past meeting scheduled on {meeting_datetime} - URL: {meeting_url}")

        # Respect rate limits by sleeping between requests
        logging.debug(f"Sleeping for {delay_between_requests} seconds between processing meetings.")
        time.sleep(delay_between_requests)

def run_scraper_thread(http_session, url):
    """
    Runs the calendar page processing in a separate daemon thread.
    
    Args:
        http_session (requests.Session): Configured HTTP session.
        url (str): URL of the calendar page.
    
    Returns:
        threading.Thread: The scraper thread.
    """
    scraper_thread = threading.Thread(target=process_calendar_page, args=(http_session, url), daemon=True, name="Scraper-Thread")
    scraper_thread.start()
    return scraper_thread

def main():
    global calendar_page_url, delay_between_requests, http_session

    # Load configuration
    config = load_configuration()
    calendar_page_url = config.get("calendar_page_url", "https://pittsburgh.legistar.com/Calendar.aspx")
    delay_between_requests = config.get("delay_between_requests", 1)

    logging.info(f"Using calendar page URL: {calendar_page_url}")
    logging.info(f"Delay between requests: {delay_between_requests} seconds")

    # Setup HTTP session
    http_session = setup_requests_session()

    # Start processing calendar page in a separate thread
    scraper_thread = run_scraper_thread(http_session, calendar_page_url)

    # Monitor the scraper thread and handle shutdown
    try:
        while scraper_thread.is_alive():
            scraper_thread.join(timeout=1)
            if shutdown_event.is_set():
                logging.info("Shutdown flag set. Waiting for scraper thread to terminate.")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Setting shutdown flag.")
        shutdown_event.set()
        scraper_thread.join()

    if shutdown_event.is_set():
        logging.info("Shutdown flag is set. Exiting main script.")
        sys.exit(0)

    # Start LLM processing
    try:
        logging.info("Starting LLM Processor.")
        llm_processor.run_llm_processing()  # Ensure this function exists and is correctly implemented
        logging.info("LLM Processor completed successfully.")
    except Exception as e:
        logging.error(f"LLM Processor encountered an error: {e}")
        logging.debug(traceback.format_exc())

    # Start Transcription
    try:
        logging.info("Starting Transcription.")
        city_council_video_transcriber.main()  # Ensure video_transcriber has a main() function
        logging.info("Transcription completed successfully.")
    except Exception as e:
        logging.error(f"Transcription encountered an error: {e}")
        logging.debug(traceback.format_exc())

if __name__ == "__main__":
    main()