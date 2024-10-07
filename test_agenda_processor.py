# test_agenda_processor.py

import pytest
from agenda_processor import get_meeting_data, rewrite_description_ollama, save_meeting_data_to_db

def test_get_meeting_data():
    test_url = "https://pittsburgh.legistar.com/MeetingDetail.aspx?ID=1229215&GUID=39EC6151-7735-4FCC-B344-931CEDE05CFA&Search="
    headers, data = get_meeting_data(test_url)
    
    # Validate headers
    expected_headers = ['File #', 'Versions', 'Agenda #', 'Status', 'Description', 'Link']
    assert headers == expected_headers, f"Headers mismatch: {headers}"
    
    # Validate row count
    assert len(data) > 0, "No data extracted from table"
    
    # Validate that each row contains 6 columns (5 + 1 for link)
    for row in data:
        assert len(row) == 6, f"Row has incorrect number of columns: {row}"

def test_rewrite_description_ollama():
    test_description = "Resolution providing for the sale of property"
    rewritten_description = rewrite_description_ollama(test_description)
    
    # Validate that Ollama returns a different description
    assert rewritten_description != test_description, "Rewritten description is the same as original"
    assert len(rewritten_description) > 0, "Ollama returned an empty description"

def test_save_meeting_data_to_db():
    test_url = "https://pittsburgh.legistar.com/MeetingDetail.aspx?ID=1229215&GUID=39EC6151-7735-4FCC-B344-931CEDE05CFA&Search="
    
    save_meeting_data_to_db(test_url)
    
    # Optionally, you can add assertions here to verify data was saved correctly
    # For example, query the database to check if the meeting exists
    # This requires setting up a test database or using fixtures