import io
import logging

import docx
import pypdf
from fastapi import UploadFile

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

            if filename.endswith(".pdf"):
                text = TextExtractor._extract_from_pdf(io.BytesIO(content))
            elif filename.endswith(".docx"):
                text = TextExtractor._extract_from_docx(io.BytesIO(content))
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
            # Re-raise as a custom exception
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
