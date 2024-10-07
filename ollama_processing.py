# ollama_processing.py

import logging
import requests
from typing import Optional
import traceback
import signal
import sys
import threading
import time
import json
import random

# Initialize a shutdown event for graceful termination
shutdown_event = threading.Event()

def signal_handler(signum, frame):
    """
    Handles interrupt signals to allow graceful shutdown.
    """
    logging.info(f"Received signal {signum}. Shutting down gracefully.")
    shutdown_event.set()

# Register signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def load_configuration() -> dict:
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
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing 'config.json': {e}")
        logging.debug("Exiting load_configuration() due to JSONDecodeError")
        return {}

# Load configuration at module level
config = load_configuration()
ollama_config = config.get("ollama", {})

# Configuration parameters with defaults
OLLAMA_API_ENDPOINT = ollama_config.get("api_endpoint", "http://localhost:11434/api/generate")
MODEL = ollama_config.get("model", "llama3:70b")
MAX_TOKENS_EVALUATE_INTEREST = ollama_config.get("max_tokens_evaluate_interest", 10)
MAX_TOKENS_REWRITE_DESCRIPTION = ollama_config.get("max_tokens_rewrite_description", 100)
MAX_TOKENS_REWRITE_LEGISLATION = ollama_config.get("max_tokens_rewrite_legislation", 500)
RETRIES = ollama_config.get("retries", 3)
BACKOFF_FACTOR = ollama_config.get("backoff_factor", 2)
TIMEOUT = ollama_config.get("timeout", 30)  # Increased timeout to 30 seconds
MAX_CONCURRENT_REQUESTS = ollama_config.get("max_concurrent_requests", 5)  # New parameter
semaphore = threading.Semaphore(MAX_CONCURRENT_REQUESTS)

def exponential_backoff(attempt, backoff_factor):
    backoff_time = backoff_factor ** attempt + random.uniform(0, 1)
    logging.debug(f"Exponential backoff calculated: {backoff_time} seconds for attempt {attempt}")
    return backoff_time

def make_generate_request(payload: dict) -> Optional[dict]:
    """
    Makes a POST request to the Ollama API to generate a response.
    
    Args:
        payload (dict): The JSON payload to send in the request.
    
    Returns:
        Optional[dict]: The JSON response from the API or None if failed.
    """
    logging.debug("Entering make_generate_request()")
    headers = {
        "Content-Type": "application/json"
    }
    try:
        with semaphore:
            logging.info(f"Thread {threading.current_thread().name}: Sending payload to Ollama API.")
            logging.debug(f"Payload: {json.dumps(payload)}")
            response = requests.post(
                OLLAMA_API_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=TIMEOUT
            )
            logging.info(f"Thread {threading.current_thread().name}: Received response from Ollama API.")
        response.raise_for_status()
        logging.debug(f"Ollama API response status: {response.status_code}")
        logging.debug(f"Ollama API response content: {response.text}")
        logging.debug("Exiting make_generate_request() with successful response")
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"Ollama API HTTP error: {http_err}")
        if 'response' in locals() and response.content:
            logging.debug(f"Ollama API response content: {response.text}")
        logging.debug("Exiting make_generate_request() due to HTTPError")
        return None
    except requests.exceptions.RequestException as req_err:
        logging.error(f"Ollama API request exception: {req_err}")
        logging.debug("Exiting make_generate_request() due to RequestException")
        return None
    except json.JSONDecodeError as json_err:
        logging.error(f"JSON decode error: {json_err} - Response content: {response.text if 'response' in locals() else 'No response'}")
        logging.debug("Exiting make_generate_request() due to JSONDecodeError")
        return None
    except Exception as e:
        logging.error(f"Unexpected error during Ollama API request: {e}")
        logging.debug(traceback.format_exc())
        logging.debug("Exiting make_generate_request() due to unexpected Exception")
        return None

def evaluate_interest(text: str, agenda_item_id: int) -> Optional[bool]:
    """
    Evaluates whether the provided text is 'interesting' using Ollama's LLM.

    Args:
        text (str): The text to evaluate.
        agenda_item_id (int): The ID of the AgendaItem for logging purposes.

    Returns:
        Optional[bool]: True if interesting, False otherwise, or None if evaluation failed.
    """
    logging.debug(f"Entering evaluate_interest() for AgendaItem ID: {agenda_item_id}")
    prompt = (
        f"The following is an agenda item for a city council meeting. It is of general interest to the community if it has an impact on their day-to-day lives. "
        f"Such an impact could be, but is not limited to, fines from trash, new red-light cameras, or protections for citizens. IT could also be interesting if the agenda item has a broad appeal, "
        f"such as anything related to the NFL draft, or other large events. Please return \"1\" if the following is of general interest to the community, otherwise return \"0\". "
        f"Your output text should only be the number \"0\" or \"1\": \"{text}\""
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "options": {
            "max_tokens": MAX_TOKENS_EVALUATE_INTEREST
        },
        "stream": False
    }

    for attempt in range(1, RETRIES + 1):
        if shutdown_event.is_set():
            logging.info(f"Shutdown flag set. Exiting evaluate_interest for AgendaItem ID: {agenda_item_id}.")
            return None
        try:
            logging.info(f"Attempt {attempt} to evaluate interest for AgendaItem ID: {agenda_item_id}")
            response = make_generate_request(payload)
            if response and 'response' in response:
                answer = response['response'].strip().lower()
                logging.debug(f"Received answer: {answer} for AgendaItem ID: {agenda_item_id}")
                if answer == '1':
                    logging.info(f"Ollama evaluated the text as interesting for AgendaItem ID: {agenda_item_id}.")
                    return True
                elif answer == '0':
                    logging.info(f"Ollama evaluated the text as not interesting for AgendaItem ID: {agenda_item_id}.")
                    return False
                else:
                    logging.error(f"Unexpected response from Ollama during interest evaluation for AgendaItem ID {agenda_item_id}: {response}")
                    return None
            else:
                logging.error(f"Unexpected response structure from Ollama during interest evaluation for AgendaItem ID {agenda_item_id}: {response}")
                return None
        except Exception as e:
            logging.error(f"Ollama API error while evaluating interest for AgendaItem ID {agenda_item_id}: {e}. Attempt {attempt} of {RETRIES}.")
            logging.debug(traceback.format_exc())
            if attempt < RETRIES:
                sleep_time = exponential_backoff(attempt, BACKOFF_FACTOR)
                logging.info(f"Sleeping for {sleep_time:.2f} seconds before retrying.")
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed to evaluate 'is_interesting' for AgendaItem ID: {agenda_item_id} after {RETRIES} attempts.")
                return False  # Default to False if all retries fail

    logging.debug(f"Exiting evaluate_interest() for AgendaItem ID: {agenda_item_id} with fallback value False")
    return False  # Fallback

def rewrite_description(description: str) -> Optional[str]:
    """
    Rewrites the agenda item description using Ollama in the style of an independent news blogger.

    Args:
        description (str): The original agenda item description.

    Returns:
        Optional[str]: The rewritten description or None if rewriting failed.
    """
    logging.debug("Entering rewrite_description()")
    prompt = (
        f'Rewrite the following agenda item text to be readable at a 5th grade level. '
        f'The rewritten text should focus on key details, be concise, and no longer than 2 sentences. '
        f'Use future tense for upcoming events. '
        f'Do not include statements like "Here is a rewritten version...": "{description}"'
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "options": {
            "max_tokens": MAX_TOKENS_REWRITE_DESCRIPTION
        },
        "stream": False
    }

    for attempt in range(1, RETRIES + 1):
        if shutdown_event.is_set():
            logging.info("Shutdown flag set. Exiting rewrite_description.")
            return None
        try:
            logging.info(f"Attempt {attempt} to rewrite description.")
            response = make_generate_request(payload)
            if response and 'response' in response:
                rewritten = response['response'].strip()
                logging.debug(f"Received rewritten description: {rewritten}")
                logging.info("Successfully rewrote description.")
                return rewritten
            else:
                logging.error(f"Unexpected response structure from Ollama during description rewriting: {response}")
                return None
        except Exception as e:
            logging.error(f"Ollama API error while rewriting description: {e}. Attempt {attempt} of {RETRIES}.")
            logging.debug(traceback.format_exc())
            if attempt < RETRIES:
                sleep_time = exponential_backoff(attempt, BACKOFF_FACTOR)
                logging.info(f"Sleeping for {sleep_time:.2f} seconds before retrying description rewriting.")
                time.sleep(sleep_time)
            else:
                logging.error("Failed to rewrite description after multiple attempts.")
                return None  # Indicate failure

    logging.debug("Exiting rewrite_description() with fallback value None")
    return None  # Fallback

def rewrite_legislation_text(text: str) -> Optional[str]:
    """
    Rewrites the legislation text using Ollama in plain language suitable for the general public.

    Args:
        text (str): The original legislation text.

    Returns:
        Optional[str]: The rewritten legislation text or None if rewriting failed.
    """
    logging.debug("Entering rewrite_legislation_text()")
    prompt = (
        f'Please summarize the following legislation in a couple of sentences. '
        f'The summary should be at a 5th grade level and include the key pieces of information community members need to know. '
        f'Responses should be no more than 10 sentences. Do not include any statements such as "Here is a rewritten version...": "{text}"'
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "options": {
            "max_tokens": MAX_TOKENS_REWRITE_LEGISLATION
        },
        "stream": False
    }

    for attempt in range(1, RETRIES + 1):
        if shutdown_event.is_set():
            logging.info("Shutdown flag set. Exiting rewrite_legislation_text.")
            return None
        try:
            logging.info(f"Attempt {attempt} to rewrite legislation text.")
            response = make_generate_request(payload)
            if response and 'response' in response:
                readable_text = response['response'].strip()
                logging.debug(f"Received rewritten legislation text: {readable_text}")
                logging.info("Successfully rewrote legislation text.")
                return readable_text
            else:
                logging.error(f"Unexpected response structure from Ollama during legislation text rewriting: {response}")
                return None
        except Exception as e:
            logging.error(f"Ollama API error while rewriting legislation text: {e}. Attempt {attempt} of {RETRIES}.")
            logging.debug(traceback.format_exc())
            if attempt < RETRIES:
                sleep_time = exponential_backoff(attempt, BACKOFF_FACTOR)
                logging.info(f"Sleeping for {sleep_time:.2f} seconds before retrying legislation text rewriting.")
                time.sleep(sleep_time)
            else:
                logging.error("Failed to rewrite legislation text after multiple attempts.")
                return None  # Indicate failure

    logging.debug("Exiting rewrite_legislation_text() with fallback value None")
    return None  # Fallback