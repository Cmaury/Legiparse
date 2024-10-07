# llm_processor.py

import logging
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from database import SessionLocal, engine  # Ensure 'engine' is imported correctly
from models import AgendaItem, Legislation
from ollama_processing import (
    rewrite_description,
    rewrite_legislation_text,
    evaluate_interest
)
import traceback
import signal
import sys
import threading

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs during development
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_processing.log"),
        logging.StreamHandler()
    ]
)

# Define batch size and maximum workers for ThreadPoolExecutor
BATCH_SIZE = 100
MAX_WORKERS = 5  # Increased from 1 to allow multiple concurrent threads

# Initialize a shutdown event
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

# Setup scoped session for thread safety
session_factory = sessionmaker(bind=engine)
SessionScoped = scoped_session(session_factory)

def process_agenda_item(item: AgendaItem) -> Optional[AgendaItem]:
    """
    Processes a single AgendaItem by rewriting its description using Ollama
    and evaluating its 'is_interesting' status.

    Args:
        item (AgendaItem): The AgendaItem instance to process.

    Returns:
        Optional[AgendaItem]: The updated AgendaItem instance or None if processing failed.
    """
    logging.debug(f"Entering process_agenda_item() for AgendaItem ID: {item.agendaitemid}")
    if shutdown_event.is_set():
        logging.info(f"Shutdown flag set. Skipping AgendaItem ID: {item.agendaitemid}.")
        return None

    if not item.description or item.description.lower() == 'none':
        logging.warning(f"AgendaItem ID: {item.agendaitemid} has no valid description. Skipping.")
        return None

    # Rewrite the description using Ollama
    try:
        rewritten = rewrite_description(item.description)
    except Exception as e:
        logging.error(f"Exception during rewriting description for AgendaItem ID: {item.agendaitemid}: {e}")
        logging.debug(traceback.format_exc())
        return None

    if rewritten:
        item.readabledescription = rewritten
        logging.debug(f"Updated 'readabledescription' for AgendaItem ID: {item.agendaitemid}.")
    else:
        logging.error(f"Failed to rewrite description for AgendaItem ID: {item.agendaitemid}.")
        return None  # Skip further processing if rewriting failed

    # Evaluate if the agenda item is interesting
    try:
        is_interesting_flag = evaluate_interest(item.readabledescription, item.agendaitemid)
    except Exception as e:
        logging.error(f"Exception during interest evaluation for AgendaItem ID: {item.agendaitemid}: {e}")
        logging.debug(traceback.format_exc())
        is_interesting_flag = False  # Default to False on exception

    if is_interesting_flag is not None:
        item.is_interesting = is_interesting_flag
        logging.debug(f"Set 'is_interesting' for AgendaItem ID: {item.agendaitemid} to {is_interesting_flag}.")
    else:
        logging.error(f"Failed to evaluate 'is_interesting' for AgendaItem ID: {item.agendaitemid}.")
        item.is_interesting = False  # Defaulting to False if evaluation failed

    logging.debug(f"Exiting process_agenda_item() for AgendaItem ID: {item.agendaitemid}")
    return item

def process_legislation_record(leg: Legislation) -> Optional[Legislation]:
    """
    Processes a single Legislation record by rewriting its text using Ollama
    and evaluating its 'is_interesting' status.

    Args:
        leg (Legislation): The Legislation instance to process.

    Returns:
        Optional[Legislation]: The updated Legislation instance or None if processing failed.
    """
    logging.debug(f"Entering process_legislation_record() for Legislation ID: {leg.legislationid}")
    if shutdown_event.is_set():
        logging.info(f"Shutdown flag set. Skipping Legislation ID: {leg.legislationid}.")
        return None

    if not leg.text or leg.text.lower() == 'none':
        logging.warning(f"Legislation ID: {leg.legislationid} has no valid text. Skipping.")
        return None

    # Rewrite the legislation text using Ollama
    try:
        rewritten = rewrite_legislation_text(leg.text)
    except Exception as e:
        logging.error(f"Exception during rewriting legislation text for Legislation ID: {leg.legislationid}: {e}")
        logging.debug(traceback.format_exc())
        return None

    if rewritten:
        leg.readabletext = rewritten
        logging.debug(f"Updated 'readabletext' for Legislation ID: {leg.legislationid}.")
    else:
        logging.error(f"Failed to rewrite legislation text for Legislation ID: {leg.legislationid}.")
        return None  # Skip further processing if rewriting failed

    # Evaluate if the legislation text is interesting
    try:
        is_interesting_flag = evaluate_interest(leg.readabletext, leg.legislationid)
    except Exception as e:
        logging.error(f"Exception during interest evaluation for Legislation ID: {leg.legislationid}: {e}")
        logging.debug(traceback.format_exc())
        is_interesting_flag = False  # Default to False on exception

    if is_interesting_flag is not None:
        leg.is_interesting = is_interesting_flag
        logging.debug(f"Set 'is_interesting' for Legislation ID: {leg.legislationid} to {is_interesting_flag}.")
    else:
        logging.error(f"Failed to evaluate 'is_interesting' for Legislation ID: {leg.legislationid}.")
        leg.is_interesting = False  # Defaulting to False if evaluation failed

    logging.debug(f"Exiting process_legislation_record() for Legislation ID: {leg.legislationid}")
    return leg

def fetch_records_in_batches(session: Session, model, filter_condition, batch_size: int = BATCH_SIZE):
    """
    Generator to fetch records in batches based on pre-fetched IDs.

    Args:
        session (Session): SQLAlchemy session object.
        model: SQLAlchemy model to query.
        filter_condition: SQLAlchemy filter condition.
        batch_size (int): Number of records per batch.

    Yields:
        List: A list of records fetched in the current batch.
    """
    logging.debug(f"Entering fetch_records_in_batches() for model: {model.__tablename__}")
    # Determine the ID field based on the model
    id_field = model.agendaitemid if model == AgendaItem else model.legislationid
    # Fetch all IDs matching the filter condition
    ids = session.query(id_field).filter(filter_condition).all()
    id_list = [id_[0] for id_ in ids]
    total = len(id_list)
    logging.info(f"Total {model.__tablename__}: {total}")

    # Process in batches
    for i in range(0, total, batch_size):
        if shutdown_event.is_set():
            logging.info("Shutdown flag set. Stopping record fetching.")
            break
        batch_ids = id_list[i:i + batch_size]
        logging.debug(f"Fetching records for batch {i // batch_size + 1}: IDs {batch_ids}")
        batch = session.query(model).filter(id_field.in_(batch_ids)).all()
        logging.info(f"Fetched batch {i // batch_size + 1}: {len(batch)} records.")
        yield batch
    logging.debug("Exiting fetch_records_in_batches()")

def update_agenda_items(session: Session):
    """
    Processes and updates AgendaItems with 'readabledescription' as 'none' in batches.

    Args:
        session (Session): SQLAlchemy session object.
    """
    logging.debug("Entering update_agenda_items()")
    # Fetch AgendaItems where 'readabledescription' is 'none'
    filter_condition = AgendaItem.readabledescription == 'none'
    for batch_num, batch in enumerate(fetch_records_in_batches(session, AgendaItem, filter_condition), start=1):
        if shutdown_event.is_set():
            logging.info("Shutdown flag set. Exiting AgendaItems processing.")
            break

        logging.info(f"Processing batch {batch_num} of AgendaItems with {len(batch)} records.")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_agenda_item, item): item for item in batch}
            for future in as_completed(futures):
                if shutdown_event.is_set():
                    logging.info("Shutdown flag set. Canceling remaining AgendaItems processing.")
                    break
                item = futures[future]
                try:
                    updated_item = future.result()
                    if updated_item:
                        session.add(updated_item)  # Mark the item as modified
                        logging.debug(f"AgendaItem ID: {updated_item.agendaitemid} marked for update.")
                except Exception as e:
                    logging.error(f"Error processing AgendaItem ID: {item.agendaitemid}: {e}")
                    logging.debug(traceback.format_exc())

        # Commit after each batch
        if not shutdown_event.is_set():
            try:
                session.commit()
                logging.info(f"Committed batch {batch_num} of {len(batch)} AgendaItems.")
            except SQLAlchemyError as e:
                session.rollback()
                logging.error(f"Database error while committing AgendaItems batch {batch_num}: {e}")
                logging.debug(traceback.format_exc())
            except Exception as e:
                session.rollback()
                logging.error(f"Unexpected error while committing AgendaItems batch {batch_num}: {e}")
                logging.debug(traceback.format_exc())
    logging.debug("Exiting update_agenda_items()")

def update_legislation_records(session: Session):
    """
    Processes and updates Legislation records with 'readabletext' as 'none' in batches.

    Args:
        session (Session): SQLAlchemy session object.
    """
    logging.debug("Entering update_legislation_records()")
    # Fetch Legislation records where 'readabletext' is 'none'
    filter_condition = Legislation.readabletext == 'none'
    for batch_num, batch in enumerate(fetch_records_in_batches(session, Legislation, filter_condition), start=1):
        if shutdown_event.is_set():
            logging.info("Shutdown flag set. Exiting Legislation records processing.")
            break

        logging.info(f"Processing batch {batch_num} of Legislation records with {len(batch)} records.")
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_legislation_record, leg): leg for leg in batch}
            for future in as_completed(futures):
                if shutdown_event.is_set():
                    logging.info("Shutdown flag set. Canceling remaining Legislation records processing.")
                    break
                leg = futures[future]
                try:
                    updated_leg = future.result()
                    if updated_leg:
                        session.add(updated_leg)  # Mark the record as modified
                        logging.debug(f"Legislation ID: {updated_leg.legislationid} marked for update.")
                except Exception as e:
                    logging.error(f"Error processing Legislation ID: {leg.legislationid}: {e}")
                    logging.debug(traceback.format_exc())

        # Commit after each batch
        if not shutdown_event.is_set():
            try:
                session.commit()
                logging.info(f"Committed batch {batch_num} of {len(batch)} Legislation records.")
            except SQLAlchemyError as e:
                session.rollback()
                logging.error(f"Database error while committing Legislation batch {batch_num}: {e}")
                logging.debug(traceback.format_exc())
            except Exception as e:
                session.rollback()
                logging.error(f"Unexpected error while committing Legislation batch {batch_num}: {e}")
                logging.debug(traceback.format_exc())
    logging.debug("Exiting update_legislation_records()")

def run_llm_processing():
    """
    Executes the LLM processing for AgendaItems and Legislation records.
    """
    logging.info("LLM Processor Started.")

    # Create a new database session
    session = SessionScoped()
    logging.debug("Database session created.")

    try:
        # Process AgendaItems
        logging.info("Starting to process AgendaItems.")
        update_agenda_items(session)

        if shutdown_event.is_set():
            logging.info("Shutdown flag set. Skipping Legislation records processing.")
            return

        # Process Legislation
        logging.info("Starting to process Legislation records.")
        update_legislation_records(session)

    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM processing: {e}")
        logging.debug(traceback.format_exc())
    finally:
        # Close the session
        session.close()
        logging.info("Database session closed.")

    logging.info("LLM Processor Finished.")