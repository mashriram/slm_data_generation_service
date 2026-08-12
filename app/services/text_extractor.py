import io
import csv
import logging
import docx
import pypdf
from typing import List, Dict, Union
from fastapi import UploadFile
from langchain_core.tools import tool

from app.utils.exceptions import FileExtractionError

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extracts text content from various file formats."""

    @staticmethod
    async def extract(file: UploadFile) -> str:
        """
        Extracts text from an uploaded file based on its content type.
        """
        filename = file.filename
        logger.info(f"Starting text extraction from '{filename}'.")

        try:
            content = await file.read()
            # Reset cursor for future reads if necessary, though file.read() usually consumes it.
            # We are creating BytesIO from content, so we are good.

            if filename.endswith(".pdf"):
                text = TextExtractor._extract_from_pdf(io.BytesIO(content))
            elif filename.endswith(".docx"):
                text = TextExtractor._extract_from_docx(io.BytesIO(content))
            elif filename.endswith(".csv"):
                text = TextExtractor._extract_from_csv(io.BytesIO(content))
            else:
                raise FileExtractionError(
                    f"Unsupported file type for extraction: {filename}"
                )

            if not text.strip():
                raise FileExtractionError(
                    f"No text could be extracted from '{filename}'. The file might be empty or image-based."
                )

            logger.info(f"Successfully extracted text from '{filename}'.")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from '{filename}': {e}")
            raise FileExtractionError(f"Could not process file '{filename}': {e}")

    @staticmethod
    def _extract_from_pdf(file_stream: io.BytesIO) -> str:
        """Extracts text from a PDF file stream."""
        reader = pypdf.PdfReader(file_stream)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

    @staticmethod
    def _extract_from_docx(file_stream: io.BytesIO) -> str:
        """Extracts text from a DOCX file stream."""
        doc = docx.Document(file_stream)
        text = [p.text for p in doc.paragraphs]
        return "\n".join(text)

    @staticmethod
    def _extract_from_csv(file_stream: io.BytesIO) -> str:
        """Extracts text from a CSV file stream as a markdown table."""
        try:
            # Decode bytes to string
            content_str = file_stream.getvalue().decode("utf-8")
            reader = csv.reader(io.StringIO(content_str))
            rows = list(reader)
            if not rows:
                return ""

            # Simple markdown table conversion
            markdown_lines = []
            header = rows[0]
            markdown_lines.append("| " + " | ".join(header) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

            for row in rows[1:]:
                markdown_lines.append("| " + " | ".join(row) + " |")

            return "\n".join(markdown_lines)
        except Exception as e:
            logger.error(f"Error parsing CSV: {e}")
            return ""

    @staticmethod
    def parse_csv_to_dicts(file_content: bytes) -> List[Dict[str, str]]:
        """Parses a CSV file content into a list of dictionaries (for few-shot examples)."""
        try:
            content_str = file_content.decode("utf-8")
            reader = csv.DictReader(io.StringIO(content_str))
            return [row for row in reader]
        except Exception as e:
            logger.error(f"Error parsing CSV to dicts: {e}")
            raise FileExtractionError(f"Invalid CSV format: {e}")

# --- LangChain Tools ---

@tool
def read_file(file_path: str) -> str:
    """
    Reads a file from the local file system.
    Supported formats: .txt, .md (plain text), .csv (as text).
    Note: Does not support complex parsing of PDF/DOCX from local path in this simple tool
    unless extended, but useful for agent to read logs or config files if needed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {file_path}: {e}"
