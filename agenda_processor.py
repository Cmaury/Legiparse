# agenda_processor.py

import logging
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse, urljoin
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime
from database import SessionLocal
from models import (
    GoverningBody, MeetingType, Meeting, Agenda, AgendaItem,
    Legislation, LegislationHistory, Attachment, Media
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("agenda_processor.log"),
        logging.StreamHandler()
    ]
)


def parse_meeting_datetime(date_str, time_str=None):
    """
    Parses the meeting date and time strings into a single datetime object.
    
    Args:
        date_str (str): The date string to parse (e.g., '9/18/2024').
        time_str (str, optional): The time string to parse (e.g., '10:00 AM'). Defaults to None.
    
    Returns:
        datetime or None: The parsed datetime object or None if parsing fails.
    """
    if not date_str:
        logging.error("Date string is empty.")
        return None

    # Define possible date and time formats
    date_formats = [
        '%m/%d/%Y',           # e.g., '09/18/2024'
        '%m-%d-%Y',           # e.g., '09-18-2024'
        '%B %d, %Y',          # e.g., 'September 18, 2024'
        '%d %B %Y',           # e.g., '18 September 2024'
        '%Y-%m-%d',           # e.g., '2024-09-18'
    ]

    time_formats = [
        '%I:%M %p',           # e.g., '10:00 AM'
        '%H:%M',              # e.g., '10:00'
        '%I %p',              # e.g., '10 AM'
        '%H',                 # e.g., '10'
    ]

    parsed_date = None
    # Parse the date
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(date_str.strip(), fmt)
            logging.debug(f"Successfully parsed date '{date_str}' with format '{fmt}'.")
            break
        except ValueError:
            continue  # Try next format

    if not parsed_date:
        logging.error(f"Failed to parse date string: '{date_str}'. No matching format found.")
        return None

    # Parse the time if provided
    if time_str:
        parsed_time = None
        for fmt in time_formats:
            try:
                parsed_time = datetime.strptime(time_str.strip(), fmt).time()
                logging.debug(f"Successfully parsed time '{time_str}' with format '{fmt}'.")
                break
            except ValueError:
                continue  # Try next format

        if parsed_time:
            # Combine date and time
            parsed_date = parsed_date.replace(hour=parsed_time.hour, minute=parsed_time.minute, second=0, microsecond=0)
        else:
            logging.error(f"Failed to parse time string: '{time_str}'. Setting time to 00:00.")
            parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        # If time is not provided, set to default 00:00
        parsed_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
        logging.debug(f"No time provided. Set to default time: {parsed_date.time()}.")

    return parsed_date


def get_meeting_details(url):
    """
    Fetches and extracts meeting details from the given URL.

    Args:
        url (str): The URL of the meeting page.

    Returns:
        dict: A dictionary containing meeting details.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        meeting_details = {}

        # Extract Meeting Name
        meeting_name = soup.find('span', {'id': 'ctl00_ContentPlaceHolder1_lblNameX'})
        meeting_name_value = soup.find('a', {'id': 'ctl00_ContentPlaceHolder1_hypName'})
        
        if meeting_name and meeting_name_value:
            key = meeting_name.get_text(strip=True)
            value = meeting_name_value.get_text(strip=True)
            meeting_details['Meeting Name'] = value
            logging.debug(f"Extracted {key}: {value}")

        # Extract Meeting Date
        meeting_date_label = soup.find('span', {'id': 'ctl00_ContentPlaceHolder1_lblDateX'})
        meeting_date_value = soup.find('span', {'id': 'ctl00_ContentPlaceHolder1_lblDate'})
        
        if meeting_date_label and meeting_date_value:
            key = meeting_date_label.get_text(strip=True)
            value = meeting_date_value.get_text(strip=True)
            meeting_details['Meeting Date'] = value  # Changed key to 'Meeting Date'
            logging.debug(f"Extracted {key}: {value}")

        # Extract Meeting Time
        meeting_time_value = soup.find('span', {'id': 'ctl00_ContentPlaceHolder1_lblTime'})
        
        if meeting_time_value:
            value = meeting_time_value.get_text(strip=True)
            meeting_details["Meeting Time"] = value
            logging.debug(f"Extracted Time: {value}")

        # Extract Meeting Location
        meeting_location_label = soup.find('span', {'id': 'ctl00_ContentPlaceHolder1_lblLocationX'})
        meeting_location_value = soup.find('span', {'id': 'ctl00_ContentPlaceHolder1_lblLocation'})
        
        if meeting_location_label and meeting_location_value:
            key = meeting_location_label.get_text(strip=True)
            value = meeting_location_value.get_text(strip=True)
            meeting_details['Meeting Location'] = value
            logging.debug(f"Extracted {key}: {value}")

        # Extract Meeting Video URL
        # Method 1: Using specific ID
        video_link_tag = soup.find('a', {'id': 'ctl00_ContentPlaceHolder1_hypVideo'})
        video_url = ''

        if video_link_tag and 'onclick' in video_link_tag.attrs:
            onclick_attr = video_link_tag['onclick']
            # Extract the URL inside window.open('...', ...)
            match = re.search(r"window\.open\(['\"]([^'\"]+)['\"],", onclick_attr)
            if match:
                relative_video_url = match.group(1)
                # Construct absolute URL
                video_url = urljoin(url, relative_video_url)
                meeting_details['Meeting Video URL'] = video_url
                logging.debug(f"Extracted Meeting Video URL: {video_url}")
            else:
                logging.debug("No valid URL found in onclick attribute for video link.")
                meeting_details['Meeting Video URL'] = ''
        else:
            logging.debug("No Meeting Video URL found via specific ID.")
            meeting_details['Meeting Video URL'] = ''

        # If video URL still not found, attempt alternative methods
        if not video_url:
            # Method 2: Search for any <a> tag with href containing 'video'
            video_link_tag = soup.find('a', href=re.compile(r'video', re.IGNORECASE))
            if video_link_tag and 'href' in video_link_tag.attrs:
                potential_video_url = video_link_tag['href']
                if not potential_video_url.startswith('http'):
                    potential_video_url = urljoin(url, potential_video_url)
                meeting_details['Meeting Video URL'] = potential_video_url
                logging.debug(f"Extracted Meeting Video URL via href search: {potential_video_url}")
            else:
                # Method 3: Check for iframes or embedded videos
                iframe_tag = soup.find('iframe', src=re.compile(r'video', re.IGNORECASE))
                if iframe_tag and 'src' in iframe_tag.attrs:
                    iframe_video_url = iframe_tag['src']
                    if not iframe_video_url.startswith('http'):
                        iframe_video_url = urljoin(url, iframe_video_url)
                    meeting_details['Meeting Video URL'] = iframe_video_url
                    logging.debug(f"Extracted Meeting Video URL via iframe: {iframe_video_url}")
                else:
                    logging.debug("No Meeting Video URL found through any method.")
                    meeting_details['Meeting Video URL'] = ''

        # Extract Meeting URL (Current page URL)
        meeting_details['Meeting URL'] = url
        logging.debug(f"Set Meeting URL: {url}")

        # Initialize Transcript attribute as 'none'
        meeting_details['Transcript'] = 'none'
        logging.debug("Initialized Transcript as 'none'.")

        logging.info(f"Scraped meeting details: {meeting_details}")
        return meeting_details

    except requests.RequestException as e:
        logging.error(f"HTTP error while fetching meeting details from {url}: {e}")
        return {}
    except Exception as e:
        logging.error(f"Unexpected error in get_meeting_details for {url}: {e}")
        return {}


def get_meeting_data(url):
    """
    Fetches and extracts meeting data (agenda items) from the given Legistar URL.

    Args:
        url (str): The Legistar URL of the meeting.

    Returns:
        tuple: A tuple containing headers (list) and data (list of lists).
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Legistar structure: agenda items are typically in 'table' tags with specific classes
        table = soup.find('table', {'class': 'rgMasterTable'})
        if not table:
            logging.warning("Agenda items table not found.")
            return [], []

        rows = table.find_all('tr', {'class': ['rgRow', 'rgAltRow']})  # Alternating row classes

        data = []
        headers = ['File #', 'Versions', 'Agenda #', 'Status', 'Description', 'Link']

        for row in rows:
            cells = row.find_all('td')
            if len(cells) < 5:
                logging.debug("Skipping row with insufficient columns.")
                continue  # Skip rows that don't have enough cells
            data_row = [cell.get_text(strip=True) for cell in cells[:5]]  # Get text content from the first 5 columns
            link_tag = row.find('a')
            if link_tag and 'href' in link_tag.attrs:
                link = link_tag['href']  # Extract the URL from the link in the row
                full_link = f"https://pittsburgh.legistar.com/{link}"
                data_row.append(full_link)  # Append full URL
                logging.debug(f"Extracted Link: {full_link}")
            else:
                data_row.append('')  # No link available
                logging.debug("No link found for this row.")
            data.append(data_row)

        logging.info(f"Scraped agenda data with {len(data)} items.")
        return headers, data

    except requests.RequestException as e:
        logging.error(f"HTTP error while fetching meeting data from {url}: {e}")
        return [], []
    except Exception as e:
        logging.error(f"Unexpected error in get_meeting_data for {url}: {e}")
        return [], []


def get_legislation_text(url):
    """
    Fetches and extracts the legislation text from the given URL.
    Appends "&FullText=1" to the URL to ensure full text is retrieved.

    Args:
        url (str): The URL of the legislation page.

    Returns:
        str: The extracted legislation text, or an empty string if not found.
    """
    try:
        # Parse the original URL
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)

        # Check if 'FullText' parameter already exists
        if 'FullText' not in query_params:
            # Add 'FullText=1' parameter
            query_params['FullText'] = ['1']
            logging.debug(f"Appending 'FullText=1' to URL: {url}")
        else:
            logging.debug(f"'FullText' parameter already present in URL: {url}")

        # Reconstruct the query string with the new parameter
        new_query = urlencode(query_params, doseq=True)

        # Reconstruct the full URL with the updated query
        updated_url = urlunparse((
            parsed_url.scheme,
            parsed_url.netloc,
            parsed_url.path,
            parsed_url.params,
            new_query,
            parsed_url.fragment
        ))

        logging.info(f"Fetching legislation text from updated URL: {updated_url}")

        # Fetch the updated URL
        response = requests.get(updated_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Define possible selectors where legislation text might be found
        selectors = [
            {'name': 'div', 'attrs': {'id': 'ctl00_ContentPlaceHolder1_divText'}},
            {'name': 'div', 'attrs': {'id': 'ctl00_ContentPlaceHolder1_divLegText'}},
            {'name': 'td', 'attrs': {'id': 'ctl00_ContentPlaceHolder1_tdLegText'}},
            {'name': 'div', 'attrs': {'class': 'legislationText'}},  # Hypothetical class
            # Add more selectors as needed for similar pages
        ]

        text_element = None
        for selector in selectors:
            logging.debug(f"Trying selector: {selector}")
            text_element = soup.find(selector['name'], selector['attrs'])
            if text_element:
                logging.debug(f"Found legislation text using selector: {selector}")
                break

        # If not found using predefined selectors, attempt a more flexible search
        if not text_element:
            logging.debug("Predefined selectors failed. Attempting flexible search.")
            # Example: Find a div containing a specific heading or keyword
            # This can be adjusted based on common patterns in similar pages
            possible_headers = ['Resolution', 'Title', 'Body']
            for header in possible_headers:
                header_tag = soup.find(['h1', 'h2', 'h3', 'p', 'span'], text=re.compile(header, re.IGNORECASE))
                if header_tag:
                    logging.debug(f"Found header '{header}' at {header_tag}")
                    # Assume the legislation text follows the header
                    parent = header_tag.find_parent()
                    if parent:
                        text_element = parent.find_next_sibling()
                        if text_element:
                            logging.debug(f"Extracted legislation text following header '{header}'")
                            break

        if text_element:
            # Clean and extract text
            legislation_text = text_element.get_text(separator='\n', strip=True)
            logging.info(f"Successfully extracted legislation text from {updated_url}")
            return legislation_text
        else:
            logging.warning(f"Legislation text not found in {updated_url}.")
            return ''

    except requests.Timeout:
        logging.error(f"Request to {url} timed out.")
        return ''
    except requests.RequestException as e:
        logging.error(f"HTTP error while fetching legislation text from {url}: {e}")
        return ''
    except Exception as e:
        logging.error(f"Unexpected error in get_legislation_text for {url}: {e}")
        return ''


def get_or_create_governing_body(session):
    """
    Retrieves the GoverningBody from the database or creates it if it doesn't exist.

    Args:
        session: Database session.

    Returns:
        GoverningBody instance or None if failed.
    """
    governing_body = session.query(GoverningBody).filter_by(
        name='City Council',
        city='Pittsburgh'
    ).first()
    if not governing_body:
        governing_body = GoverningBody(
            name='City Council',
            description='The legislative body of the city',
            contactinfo={'email': 'info@citycouncil.gov', 'phone': '123-456-7890'},
            city='Pittsburgh',
            county='Allegheny',
            state='Pennsylvania'
        )
        session.add(governing_body)
        try:
            session.commit()
            session.refresh(governing_body)
            logging.info(f"Created GoverningBody: {governing_body.name} (ID: {governing_body.governingbodyid})")
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while creating GoverningBody: {e}")
            return None
    else:
        logging.info(f"Retrieved GoverningBody: {governing_body.name} (ID: {governing_body.governingbodyid})")
    return governing_body


def get_or_create_meeting_type(session, meeting_name, governing_body):
    """
    Retrieves the MeetingType from the database or creates it if it doesn't exist.

    Args:
        session: Database session.
        meeting_name (str): The name of the meeting.
        governing_body (GoverningBody): The governing body associated with the meeting.

    Returns:
        MeetingType instance or None if failed.
    """
    meeting_type = session.query(MeetingType).filter_by(
        name=meeting_name,
        governingbodyid=governing_body.governingbodyid
    ).first()
    if not meeting_type:
        meeting_type = MeetingType(
            name=meeting_name,
            description='Regular committee meetings',
            governing_body=governing_body
        )
        session.add(meeting_type)
        try:
            session.commit()
            session.refresh(meeting_type)
            logging.info(f"Created MeetingType: {meeting_type.name} (ID: {meeting_type.meetingtypeid})")
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while creating MeetingType: {e}")
            return None
    else:
        logging.info(f"Retrieved MeetingType: {meeting_type.name} (ID: {meeting_type.meetingtypeid})")
    return meeting_type


def get_or_create_meeting(session, meeting_name, meeting_datetime, agenda_status, minutes_status,
                          meeting_location, published_agenda_url, published_minutes_url,
                          meeting_video_url, meeting_type, meeting_url, transcript):
    """
    Retrieves the Meeting from the database or creates it if it doesn't exist.

    Args:
        session: Database session.
        meeting_name (str): Name of the meeting.
        meeting_datetime (datetime): Date and time of the meeting.
        agenda_status (str): Status of the agenda.
        minutes_status (str): Status of the minutes.
        meeting_location (str): Location of the meeting.
        published_agenda_url (str): URL of the published agenda.
        published_minutes_url (str): URL of the published minutes.
        meeting_video_url (str): URL of the meeting video.
        meeting_type (MeetingType): Type of the meeting.
        meeting_url (str): URL of the meeting page.
        transcript (str): Transcript status.

    Returns:
        Meeting instance or None if failed.
    """
    meeting = session.query(Meeting).filter_by(
        meetingurl=meeting_url
    ).first()
    if not meeting:
        meeting = Meeting(
            name=meeting_name,
            agendastatus=agenda_status,
            minutesstatus=minutes_status,
            datetime=meeting_datetime,
            location=meeting_location,
            publishedagendaurl=published_agenda_url,
            publishedminutesurl=published_minutes_url,
            videourl=meeting_video_url,
            meeting_type=meeting_type,
            meetingurl=meeting_url,
            transcript=transcript,
            processed=False  # Initially not processed
        )
        session.add(meeting)
        try:
            session.commit()
            session.refresh(meeting)
            logging.info(f"Created Meeting: {meeting.name} (ID: {meeting.meetingid})")
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while creating Meeting: {e}")
            return None
    else:
        # Update meeting details if they have changed and meeting is still in the future
        if meeting.datetime >= datetime.now():
            updated = False
            if meeting.name != meeting_name:
                meeting.name = meeting_name
                updated = True
            if meeting.agendastatus != agenda_status:
                meeting.agendastatus = agenda_status
                updated = True
            if meeting.minutesstatus != minutes_status:
                meeting.minutesstatus = minutes_status
                updated = True
            if meeting.location != meeting_location:
                meeting.location = meeting_location
                updated = True
            if meeting.publishedagendaurl != published_agenda_url:
                meeting.publishedagendaurl = published_agenda_url
                updated = True
            if meeting.publishedminutesurl != published_minutes_url:
                meeting.publishedminutesurl = published_minutes_url
                updated = True
            if meeting.videourl != meeting_video_url:
                meeting.videourl = meeting_video_url
                updated = True
            if meeting.transcript != transcript:
                meeting.transcript = transcript
                updated = True
            if updated:
                try:
                    session.commit()
                    logging.info(f"Updated Meeting: {meeting.name} (ID: {meeting.meetingid})")
                except SQLAlchemyError as e:
                    session.rollback()
                    logging.error(f"Database error while updating Meeting: {e}")
                    return None
        else:
            logging.info(f"Meeting '{meeting.name}' is in the past. Skipping updates.")
            return None
    return meeting


def get_or_create_agenda(session, meeting):
    """
    Retrieves the Agenda from the database or creates it if it doesn't exist.

    Args:
        session: Database session.
        meeting (Meeting): The associated Meeting instance.

    Returns:
        Agenda instance or None if failed.
    """
    agenda = session.query(Agenda).filter_by(
        meetingid=meeting.meetingid
    ).first()
    if not agenda:
        agenda = Agenda(
            meetingid=meeting.meetingid,
            title=f"Agenda for {meeting.name} on {meeting.datetime.strftime('%Y-%m-%d')}",
            description=''
        )
        session.add(agenda)
        try:
            session.commit()
            session.refresh(agenda)
            logging.info(f"Created Agenda: {agenda.title} (ID: {agenda.agendaid})")
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while creating Agenda: {e}")

# Function to save Legislation
def save_legislation(session, file_number, version, status, description, link, index):
    """
    Saves or updates a Legislation record.

    Args:
        session: Database session.
        file_number (str): File number of the legislation.
        version (str): Version of the legislation.
        status (str): Status of the legislation.
        description (str): Description/title of the legislation.
        link (str): URL to the legislation details.
        index (int): Index of the agenda item being processed.

    Returns:
        Legislation instance or None if failed.
    """
    try:
        legislation = session.query(Legislation).filter_by(
            filenumber=file_number,
            version=version
        ).first()
        if not legislation:
            # Get legislation text
            legislation_text = get_legislation_text(link)
            logging.debug(f"Legislation text extracted: {legislation_text}")

            # Rewrite legislation text
            readable_text = "none"

            legislation = Legislation(
                filenumber=file_number,
                version=int(version) if version.isdigit() else None,
                status=status,
                title=description,
                type='Unknown',  # Update if you have the Type
                sponsors=[],
                indexes='',
                text=legislation_text,
                readabletext=readable_text,
                incontrol='',
                filecreateddate=None,
                onagendadate=None,
                finalactiondate=None,
                enactmentdate=None,
                enactmentnumber='',
                effectivedate=None
            )
            session.add(legislation)
            session.commit()
            session.refresh(legislation)
            logging.info(f"Created Legislation: {legislation.title} (ID: {legislation.legislationid})")
        else:
            # Update legislation text if it's empty
            if not legislation.text:
                legislation_text = get_legislation_text(link)
                legislation.text = legislation_text
                # Optionally, update readable text as well
                readable_text = "none"
                legislation.readabletext = readable_text
                session.commit()
                logging.info(f"Updated Legislation: {legislation.title} (ID: {legislation.legislationid})")
        return legislation
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error while processing Legislation for agenda item {index}: {e}")
        return None
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error while processing Legislation for agenda item {index}: {e}")
        return None

# Function to save AgendaItem
def save_agenda_item(session, agenda, description, rewritten_description, legislation, agenda_number, index):
    """
    Saves an AgendaItem and associates it with Legislation.

    Args:
        session: Database session.
        agenda (Agenda): Agenda instance to associate the item with.
        description (str): Description/title of the agenda item.
        rewritten_description (str): Rewritten description for readability.
        legislation (Legislation): Associated Legislation instance.
        agenda_number (str): Agenda number of the item.
        index (int): Index of the agenda item being processed.

    Returns:
        AgendaItem instance or None if failed.
    """
    try:
        order = int(agenda_number) if agenda_number.isdigit() else None
    except ValueError:
        logging.warning(f"Non-numeric agenda number '{agenda_number}' for AgendaItem {index}. Setting order to None.")
        order = None

    try:
        agenda_item = AgendaItem(
            agendaid=agenda.agendaid,
            order=order,
            title=description,
            description=description,  # Original description
            readabledescription=rewritten_description  # Rewritten description
        )
        session.add(agenda_item)
        session.commit()
        session.refresh(agenda_item)
        logging.info(f"Created AgendaItem {index}: {agenda_item.title} (ID: {agenda_item.agendaitemid})")
    except SQLAlchemyError as e:
        session.rollback()
        logging.error(f"Database error while inserting AgendaItem {index}: {e}")
        return None
    except Exception as e:
        session.rollback()
        logging.error(f"Unexpected error while inserting AgendaItem {index}: {e}")
        return None

    # Associate AgendaItem with Legislation
    if legislation and legislation not in agenda_item.legislations:
        try:
            agenda_item.legislations.append(legislation)
            session.commit()
            logging.debug(f"Associated Legislation ID {legislation.legislationid} with AgendaItem ID {agenda_item.agendaitemid}")
        except SQLAlchemyError as e:
            session.rollback()
            logging.error(f"Database error while associating Legislation with AgendaItem {agenda_item.agendaitemid}: {e}")
        except Exception as e:
            session.rollback()
            logging.error(f"Unexpected error while associating Legislation with AgendaItem {agenda_item.agendaitemid}: {e}")
    
    return agenda_item

# Main function to scrape and save to the database
def save_meeting_data_to_db(url):
    logging.info(f"Starting to save meeting data for URL: {url}")
    headers, data = get_meeting_data(url)
    meeting_details = get_meeting_details(url)

    logging.info(f"Writing to database: {meeting_details}")
    logging.info(f"Writing to database: Headers - {headers}, Number of Agenda Items - {len(data)}")

    # Create a new database session
    session = SessionLocal()

    try:
        # Process meeting details
        meeting_name = meeting_details.get('Meeting Name', 'Unknown Meeting')
        agenda_status = meeting_details.get('Agenda Status', '')
        minutes_status = meeting_details.get('Minutes Status', '')
        meeting_date_str = meeting_details.get('Meeting Date', '').strip()
        meeting_time_str = meeting_details.get('Meeting Time', '').strip()
        meeting_location = meeting_details.get('Meeting Location', '')
        published_agenda_url = meeting_details.get('Published agenda', '')
        published_minutes_url = meeting_details.get('Published minutes', '')
        meeting_video_url = meeting_details.get('Meeting Video URL', '')
        meeting_url = meeting_details.get('Meeting URL', '')
        transcript = meeting_details.get('Transcript', 'none')

        # Parse the meeting date and time
        if meeting_date_str:
            meeting_datetime = parse_meeting_datetime(meeting_date_str, meeting_time_str)
        else:
            logging.warning("Meeting date/time not found or empty.")
            meeting_datetime = None

        if meeting_datetime is None:
            logging.warning("Meeting date/time is missing or invalid. Skipping this meeting.")
            return  # Or use 'continue' if inside a loop

        # Get or create GoverningBody
        governing_body = get_or_create_governing_body(session)
        if not governing_body:
            logging.error("Failed to retrieve or create GoverningBody. Skipping this meeting.")
            return

        # Get or create MeetingType
        meeting_type = get_or_create_meeting_type(session, meeting_name, governing_body)
        if not meeting_type:
            logging.error("Failed to retrieve or create MeetingType. Skipping this meeting.")
            return

        # Get or create Meeting
        meeting = get_or_create_meeting(session, meeting_name, meeting_datetime, agenda_status, minutes_status,
                                       meeting_location, published_agenda_url, published_minutes_url,
                                       meeting_video_url, meeting_type, meeting_url, transcript)
        if not meeting:
            logging.error("Failed to retrieve or create Meeting. Skipping this meeting.")
            return

        # Get or create Agenda
        agenda = get_or_create_agenda(session, meeting)
        if not agenda:
            logging.error("Failed to retrieve or create Agenda. Skipping this meeting.")
            return

        # Process agenda items and legislation
        logging.info("Starting to process agenda items.")
        for index, row in enumerate(data, start=1):
            # row = [File #, Versions, Agenda #, Status, Description, Link]
            file_number = row[0]
            version = row[1]
            agenda_number = row[2]
            status = row[3]
            description = row[4]
            link = row[5]

            logging.info(f"Processing agenda item {index}: File #{file_number}, Agenda #{agenda_number}, Description: {description}")

            # Validate essential fields
            if not description:
                logging.warning(f"Skipping agenda item {index} with missing description for Meeting ID: {meeting.meetingid}")
                continue

            # Rewrite the agenda item description
            rewritten_description = "none"  

            # Save or retrieve Legislation
            legislation = save_legislation(session, file_number, version, status, description, link, index)
            if not legislation:
                logging.warning(f"Skipping agenda item {index} due to Legislation processing failure.")
                continue

            # Save AgendaItem and associate with Legislation
            agenda_item = save_agenda_item(session, agenda, description, rewritten_description, legislation, agenda_number, index)
            if not agenda_item:
                logging.warning(f"Failed to save AgendaItem {index}. Continuing with next item.")
                continue

        logging.info("Data saved to the database successfully.")

    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        session.rollback()
    finally:
        session.close()