# city_council_meeting_scraper.py

import logging
import feedparser
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import time
from datetime import datetime
import traceback
import signal
import sys
import threading

import agenda_processor
import llm_processor
import city_council_video_transcriber

from database import SessionLocal  # Ensure this import is present

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

# Load the configuration
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
base_rss_url = ""
delay_between_requests = 1


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
    retry_strategy = Retry(
        total=5,  # Total number of retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
        allowed_methods=["HEAD", "GET", "OPTIONS"],  # HTTP methods to retry
        backoff_factor=1  # Exponential backoff factor (1s, 2s, 4s, 8s, 16s)
    )

    # Create an HTTP adapter with the retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)

    # Create a requests session and mount the adapter
    http = requests.Session()
    http.headers.update(headers)
    http.mount("https://", adapter)
    http.mount("http://", adapter)

    logging.debug("Exiting setup_requests_session()")
    return http

def process_rss_feed(shutdown_event):
    """
    Fetches and processes RSS feed entries from the specified URL.
    Implements custom headers, retry mechanism with exponential backoff,
    and enhanced error handling.
    
    Args:
        shutdown_event (threading.Event): Event to signal shutdown.
    """
    logging.info("Starting RSS feed processing.")

    # Initialize HTTP session with retries
    http = setup_requests_session()

    # Get current datetime for comparison
    now = datetime.now()

    while not shutdown_event.is_set():
        try:
            # Fetch RSS feed
            rss_url = base_rss_url
            logging.info(f"Fetching RSS feed from: {rss_url}")
            response = http.get(rss_url, timeout=10)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Parse the RSS feed content
            feed = feedparser.parse(response.content)

            # Check for feed parsing errors
            if feed.bozo:
                logging.error(f"Error parsing RSS feed: {feed.bozo_exception}")
                break  # Exit loop on parsing error

            # Check if there are no entries in the feed
            if len(feed.entries) == 0:
                logging.info("No more entries found in the RSS feed.")
                break  # Exit loop if no entries

            # Process each entry in the feed
            for entry in feed.entries:
                if shutdown_event.is_set():
                    logging.info("Shutdown signal received. Exiting RSS feed processing loop.")
                    break

                meeting_url = entry.link
                meeting_title = entry.title
                logging.info(f"Found meeting: {meeting_title} - {meeting_url}")

                # Skip processing if the meeting is in the past

                # Check if the meeting already exists in the DB
                with SessionLocal() as session:
                    existing_meeting = agenda_processor.get_meeting_details(meeting_url)
                    if existing_meeting.get('meeting_details') and existing_meeting['meeting_details'].get('Meeting Date'):
                        if existing_meeting.meeting_details['Meeting Date'] >= now:
                            logging.info(f"Meeting '{meeting_title}' already exists and is upcoming. Updating if necessary.")
                            # Agenda processor will handle updates
                        else:
                            logging.info(f"Meeting '{meeting_title}' exists but is in the past. Skipping.")
                            continue
                    else:
                        logging.info(f"Meeting '{meeting_title}' is new. Processing.")

                # If the meeting is new or needs updating, process it
                try:
                    logging.info(f"Processing agenda for: {meeting_url}")
                    process_agenda(meeting_url)  # Replace with your actual processing function
                    logging.info(f"Processed agenda for: {meeting_title}")
                except Exception as e:
                    logging.error(f"Failed to process agenda for {meeting_title}: {e}")
                    logging.debug(traceback.format_exc())
                    continue

                # Optionally, mark the meeting as processed in the DB
                with SessionLocal() as session:
                    meeting = session.query(Meeting).filter_by(meetingurl=meeting_url).first()
                    if meeting:
                        meeting.processed = True
                        session.commit()
                        logging.debug(f"Marked Meeting '{meeting_title}' as processed.")

                # Delay between processing entries to respect rate limits
                logging.debug(f"Sleeping for {delay_between_requests} seconds between entries.")
                time.sleep(delay_between_requests)

            # Delay before the next fetch iteration
            logging.debug(f"Sleeping for {delay_between_requests} seconds before fetching the next RSS feed.")
            time.sleep(delay_between_requests)

        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred while fetching RSS feed: {http_err}")
            logging.debug(traceback.format_exc())
            break  # Exit loop on HTTP error
        except requests.exceptions.ConnectionError as conn_err:
            logging.error(f"Connection error occurred while fetching RSS feed: {conn_err}")
            logging.debug(traceback.format_exc())
            logging.info("Waiting before retrying due to connection error.")
            time.sleep(delay_between_requests * 2)  # Wait before retrying
            continue  # Continue to next iteration
        except requests.exceptions.Timeout as timeout_err:
            logging.error(f"Timeout error occurred while fetching RSS feed: {timeout_err}")
            logging.debug(traceback.format_exc())
            logging.info("Waiting before retrying due to timeout.")
            time.sleep(delay_between_requests * 2)  # Wait before retrying
            continue  # Continue to next iteration
        except requests.exceptions.RequestException as req_err:
            logging.error(f"Request exception occurred while fetching RSS feed: {req_err}")
            logging.debug(traceback.format_exc())
            break  # Exit loop on other request exceptions
        except Exception as e:
            logging.error(f"Unexpected error occurred in RSS feed processing: {e}")
            logging.debug(traceback.format_exc())
            break  # Exit loop on unexpected errors

    logging.info("Finished RSS feed processing.")

def run_rss_feed_thread():
    """
    Runs the RSS feed processing in a separate daemon thread.
    
    Returns:
        threading.Thread: The RSS feed processing thread.
    """
    rss_thread = threading.Thread(target=process_rss_feed, args=(shutdown_event,), daemon=True, name="RSS-Thread")
    rss_thread.start()
    return rss_thread

def main():
    global base_rss_url, delay_between_requests

    # Load configuration
    config = load_configuration()
    base_rss_url = config.get("calendar_rss_url")
    delay_between_requests = config.get("delay_between_requests", 1)

    if not base_rss_url:
        logging.error("Missing required configuration parameter: 'calendar_rss_url'.")
        sys.exit(1)

    # Start processing RSS feed in a separate thread
    rss_thread = run_rss_feed_thread()

    # Monitor the RSS thread and handle shutdown
    try:
        while rss_thread.is_alive():
            rss_thread.join(timeout=1)
            if shutdown_event.is_set():
                logging.info("Shutdown flag set. Waiting for RSS feed processing to terminate.")
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Setting shutdown flag.")
        shutdown_event.set()
        rss_thread.join()

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
        city_council_video_transcriber.main()  # Ensure video_main is correctly imported and has a main() function
        logging.info("Transcription completed successfully.")
    except Exception as e:
        logging.error(f"Transcription encountered an error: {e}")
        logging.debug(traceback.format_exc())

if __name__ == "__main__":
    main()