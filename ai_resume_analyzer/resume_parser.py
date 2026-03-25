"""
resume_parser.py
----------------
Handles extracting raw text from uploaded resume files.
Supports: PDF (.pdf) and Word Documents (.docx)
"""

import io
import logging
from pathlib import Path

# PDF parsing
try:
    from pypdf import PdfReader
except ImportError:
    from PyPDF2 import PdfReader  # fallback to PyPDF2

# DOCX parsing
from docx import Document

# Set up module logger
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract all text from a PDF file.

    Args:
        file_bytes (bytes): Raw bytes of the PDF file.

    Returns:
        str: Concatenated text from all pages.

    Raises:
        ValueError: If the PDF has no extractable text (e.g., scanned image).
    """
    text_parts = []

    try:
        reader = PdfReader(io.BytesIO(file_bytes))

        if len(reader.pages) == 0:
            raise ValueError("The uploaded PDF has no pages.")

        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            else:
                logger.warning(f"Page {page_num + 1} returned no text — may be an image.")

    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise RuntimeError(f"Failed to parse PDF: {e}") from e

    full_text = "\n".join(text_parts).strip()

    if not full_text:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned/image-based PDF. Please upload a text-based PDF."
        )

    return full_text


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract all text from a DOCX (Word) file.

    Args:
        file_bytes (bytes): Raw bytes of the DOCX file.

    Returns:
        str: Concatenated text from all paragraphs.

    Raises:
        RuntimeError: If the DOCX file cannot be parsed.
    """
    try:
        document = Document(io.BytesIO(file_bytes))
        paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
        full_text = "\n".join(paragraphs).strip()
    except Exception as e:
        logger.error(f"Error reading DOCX: {e}")
        raise RuntimeError(f"Failed to parse DOCX: {e}") from e

    if not full_text:
        raise ValueError("No text could be extracted from this Word document.")

    return full_text


def extract_resume_text(uploaded_file) -> str:
    """
    Main entry point: extracts text from a Streamlit UploadedFile object.

    Detects file type from the filename extension and routes to the
    appropriate parser.

    Args:
        uploaded_file: A Streamlit UploadedFile object with .name and .read().

    Returns:
        str: Extracted resume text.

    Raises:
        ValueError: If file type is unsupported or extraction fails.
    """
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()

    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    logger.info(f"Parsing file: {uploaded_file.name} ({len(file_bytes)} bytes)")

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    else:
        extension = Path(file_name).suffix
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            "Please upload a PDF (.pdf) or Word (.docx) file."
        )
