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

from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException, NoSuchElementException

from logging.handlers import RotatingFileHandler

# ============================
# 1. Logging Configuration
# ============================

# Remove any existing handlers to prevent duplicate logs
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# Setup Rotating File Handler
rotating_handler = RotatingFileHandler(
    'city_council_meeting_scraper.log', maxBytes=5*1024*1024, backupCount=5
)  # 5 MB per file, 5 backups
rotating_handler.setLevel(logging.DEBUG)  # Changed to DEBUG for detailed logs
rotating_formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)
rotating_handler.setFormatter(rotating_formatter)
logging.getLogger('').addHandler(rotating_handler)

# Setup Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # INFO level for console
console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
console_handler.setFormatter(console_formatter)
logging.getLogger('').addHandler(console_handler)

# Set the root logger level to DEBUG to capture all levels
logging.getLogger('').setLevel(logging.DEBUG)

logging.info("Logging is set up successfully.")

# ============================
# 2. Graceful Shutdown Handling
# ============================

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

# ============================
# 3. Helper Functions
# ============================

def  process_agenda(meeting_url):

    agenda_processor.save_meeting_data_to_db(meeting_url)

def is_valid_url(url):
    """
    Validates the URL structure.
    
    Args:
        url (str): URL string to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return all([parsed.scheme, parsed.netloc])

def load_configuration():
    """
    Loads configuration parameters from 'config.json'.
    
    Returns:
        dict: Configuration parameters.
    """
    logger = logging.getLogger('load_configuration')
    logger.debug("Entering load_configuration()")
    try:
        with open('config.json') as config_file:
            config = json.load(config_file)
        logger.info("Configuration loaded successfully.")
        logger.debug("Exiting load_configuration()")
        return config
    except FileNotFoundError:
        logger.error("Configuration file 'config.json' not found.")
        logger.debug("Exiting load_configuration() due to FileNotFoundError")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing 'config.json': {e}")
        logger.debug("Exiting load_configuration() due to JSONDecodeError")
        sys.exit(1)

def setup_requests_session():
    """
    Sets up a requests session with custom headers and retry strategy.
    
    Returns:
        requests.Session: Configured HTTP session.
    """
    logger = logging.getLogger('setup_requests_session')
    logger.debug("Entering setup_requests_session()")
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

    logger.debug("Exiting setup_requests_session()")
    return http

def fetch_calendar_page(http_session, url, browser='chrome'):
    """
    Fetches the main calendar page content using Selenium and parses it with BeautifulSoup.
    
    Args:
        http_session (requests.Session): Configured HTTP session.
        url (str): URL of the calendar page.
        browser (str): Browser to use ('chrome' or 'firefox').
    
    Returns:
        BeautifulSoup object or None if fetching fails.
    """
    logger = logging.getLogger('fetch_calendar_page')
    logger.info(f"Entering fetch_calendar_page with URL: {url} and browser: {browser}")

    try:
        if browser.lower() == 'chrome':
            options = ChromeOptions()
            options.add_argument('--headless')  # Run in headless mode
            options.add_argument('--disable-gpu')
            driver = webdriver.Chrome(options=options)
            logger.debug("Initialized Chrome WebDriver.")
        elif browser.lower() == 'firefox':
            options = FirefoxOptions()
            options.add_argument('--headless')  # Run in headless mode
            driver = webdriver.Firefox(options=options)
            logger.debug("Initialized Firefox WebDriver.")
        else:
            logger.error(f"Unsupported browser: {browser}. Choose 'chrome' or 'firefox'.")
            return None

        logger.debug(f"Navigating to '{url}'.")
        driver.get(url)
        logger.info(f"Page loaded successfully: {url}")
        
        # Click the "List View" link/button to load all meetings
        try:
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

        # Wait for specific elements to load after clicking "List View"
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'videolink'))
            )
            logger.info("List View loaded successfully.")
        except TimeoutException:
            logger.warning("Timed out waiting for elements after clicking 'List View'.")
            # Decide whether to proceed or return
            # For now, proceed

        # Now, parse the page source with BeautifulSoup
        page_source = driver.page_source
        logger.debug("Retrieved page source after interacting with Selenium.")

        soup = BeautifulSoup(page_source, 'html.parser')
        logger.info("Successfully fetched and parsed calendar page.")
        return soup

    except Exception as e:
        logger.error(f"Error fetching calendar page: {e}")
        logger.debug(traceback.format_exc())
        return None
    finally:
        driver.quit()
        logger.info("Selenium WebDriver has been closed.")

def extract_meetings(soup, calendar_page_url):
    """
    Extracts meeting detail URLs from the calendar page.
    
    Args:
        soup (BeautifulSoup): Parsed HTML of the calendar page.
        calendar_page_url (str): Base URL to construct absolute URLs.
    
    Returns:
        List of meeting URLs.
    """
    meetings = []
    logger = logging.getLogger('extract_meetings')
    try:
        # Find all <a> tags with id containing 'hypMeetingDetail'
        meeting_links = soup.find_all('a', id=re.compile('hypMeetingDetail'))
        logger.info(f"Found {len(meeting_links)} meeting detail links.")

        for link in meeting_links:
            if shutdown_event.is_set():
                logger.info("Shutdown signal received. Stopping meeting extraction.")
                break

            # Extract href and construct absolute URL
            relative_url = link.get('href')
            if not relative_url:
                logger.debug("No href found in meeting link. Skipping.")
                continue

            meeting_url = urljoin(calendar_page_url, relative_url)
            logger.debug(f"Extracted meeting URL: {meeting_url}")

            # Validate URL
            if is_valid_url(meeting_url):
                meetings.append(meeting_url)
                logger.debug(f"Added valid meeting URL: {meeting_url}")
            else:
                logger.warning(f"Invalid meeting URL skipped: {meeting_url}")

        logger.info(f"Extracted {len(meetings)} meetings from the calendar page.")
        return meetings
    except Exception as e:
        logger.error(f"Error extracting meetings: {e}")
        logger.debug(traceback.format_exc())
        return meetings

def process_calendar_page(http_session, url, browser='chrome'):
    """
    Processes the main calendar page: fetches, parses, extracts meetings, and processes them.
    
    Args:
        http_session (requests.Session): Configured HTTP session.
        url (str): URL of the calendar page.
        browser (str): Browser to use ('chrome' or 'firefox').
    """
    logger = logging.getLogger('process_calendar_page')
    logger.info(f"Starting to process calendar page: {url} with browser: {browser}")

    soup = fetch_calendar_page(http_session, url, browser=browser)
    if not soup:
        logger.error("Failed to fetch or parse the calendar page. Exiting processing.")
        return

    meetings = extract_meetings(soup, calendar_page_url)
    if not meetings:
        logger.info("No meetings found to process.")
        return

    now = datetime.now()
    for meeting_url in meetings:
        if shutdown_event.is_set():
            logger.info("Shutdown signal received. Stopping meeting extraction.")
            break

        logger.info(f"Processing meeting URL: {meeting_url}")
        try:
            process_agenda(meeting_url)
            logger.info(f"Successfully processed meeting: {meeting_url}")
        except Exception as e:
            logger.error(f"Error processing meeting {meeting_url}: {e}")
            logger.debug(traceback.format_exc())

        # Respect rate limits by sleeping between requests
        logger.debug(f"Sleeping for {delay_between_requests} seconds between processing meetings.")
        time.sleep(delay_between_requests)

def run_scraper_thread(http_session, url, browser='chrome'):
    """
    Runs the calendar page processing in a separate daemon thread.
    
    Args:
        http_session (requests.Session): Configured HTTP session.
        url (str): URL of the calendar page.
        browser (str): Browser to use ('chrome' or 'firefox').
    
    Returns:
        threading.Thread: The scraper thread.
    """
    scraper_thread = threading.Thread(target=process_calendar_page, args=(http_session, url, browser), daemon=True, name="Scraper-Thread")
    scraper_thread.start()
    return scraper_thread

# ============================
# 4. Main Execution
# ============================

def main():
    global calendar_page_url, delay_between_requests, http_session

    # Load configuration
    config = load_configuration()
    calendar_page_url = config.get("calendar_page_url", "https://pittsburgh.legistar.com/Calendar.aspx")
    delay_between_requests = config.get("delay_between_requests", 1)
    browser = config.get("browser", "chrome")  # Added browser from config

    logging.info(f"Using calendar page URL: {calendar_page_url}")
    logging.info(f"Delay between requests: {delay_between_requests} seconds")
    logging.info(f"Browser for Selenium: {browser}")

    # Setup HTTP session
    http_session = setup_requests_session()

    # Start processing calendar page in a separate thread
    scraper_thread = run_scraper_thread(http_session, calendar_page_url, browser=browser)

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