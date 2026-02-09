import pytest
import io
import csv
from app.services.text_extractor import TextExtractor

def test_csv_extraction():
    # Create a dummy CSV content
    csv_content = "question,answer\nWhat is 2+2?,4\nWhat is the capital of France?,Paris"
    file_bytes = io.BytesIO(csv_content.encode("utf-8"))

    # We can't easily mock UploadFile with async read, so let's test the underlying static method directly
    # Wait, the static method _extract_from_csv takes BytesIO, perfect.

    extracted_text = TextExtractor._extract_from_csv(file_bytes)
    print(f"Extracted CSV Text:\n{extracted_text}")

    assert "| question | answer |" in extracted_text
    assert "| What is 2+2? | 4 |" in extracted_text

def test_csv_to_dicts():
    csv_content = "question,answer\nWhat is 2+2?,4\nWhat is the capital of France?,Paris"
    file_bytes = csv_content.encode("utf-8")

    dicts = TextExtractor.parse_csv_to_dicts(file_bytes)
    print(f"Parsed Dicts: {dicts}")

    assert len(dicts) == 2
    assert dicts[0]["question"] == "What is 2+2?"
    assert dicts[0]["answer"] == "4"

if __name__ == "__main__":
    test_csv_extraction()
    test_csv_to_dicts()
