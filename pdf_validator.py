import os
import shutil
import json
import string
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
from pydantic import BaseModel, Field

from config import (
    PDF_STAGING_DIR,
    PDF_SOURCE_DIR,
    PDF_QUARANTINE_DIR,
)

MIN_FILE_SIZE_BYTES = 1024           # 1 KB minimum
MAX_FILE_SIZE_BYTES = 150 * 1024 * 1024  # 150 MB maximum
MIN_PAGES = 1
MAX_PAGES_WARNING = 300
MIN_TEXT_CHARS_PER_PAGE = 50
PRINTABLE_CHAR_RATIO_THRESHOLD = 0.70  # Below 70% printable indicates font CMap corruption


class ValidationResult(BaseModel):
    file_path: str
    file_name: str
    is_valid: bool = True
    file_size_bytes: int = 0
    page_count: int = 0
    is_encrypted: bool = False
    needs_password: bool = False
    native_text_pages: int = 0
    scanned_pages: int = 0
    blank_pages: int = 0
    ocr_available: bool = False
    encoding_health_score: float = 1.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    summary_preview: str = ""


class PDFValidator:
    """
    Ingestion Hygiene Gatekeeper for PDF documents.
    Validates physical file integrity, DRM/encryption, structural parsability,
    and semantic readability before promoting files to the clean source directory.
    """

    def __init__(
        self,
        min_size: int = MIN_FILE_SIZE_BYTES,
        max_size: int = MAX_FILE_SIZE_BYTES,
        min_text_len: int = MIN_TEXT_CHARS_PER_PAGE,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_text_len = min_text_len
        self.tesseract_installed = self._check_ocr_engine()

    def _check_ocr_engine(self) -> bool:
        if not shutil.which("tesseract"):
            return False
        try:
            import pytesseract
            langs = pytesseract.get_languages()
            return "eng" in langs
        except Exception:
            return False

    def validate_file(self, file_path: str) -> ValidationResult:
        file_name = os.path.basename(file_path)
        result = ValidationResult(
            file_path=file_path,
            file_name=file_name,
            ocr_available=self.tesseract_installed,
        )

        # -------------------------------------------------------------
        # Gate 1: Signature & Physical Size
        # -------------------------------------------------------------
        if not os.path.isfile(file_path):
            result.is_valid = False
            result.errors.append(f"File not found or is a directory: {file_path}")
            return result

        try:
            file_size = os.path.getsize(file_path)
            result.file_size_bytes = file_size
        except OSError as e:
            result.is_valid = False
            result.errors.append(f"Could not read file size: {e}")
            return result

        if file_size < self.min_size:
            result.is_valid = False
            result.errors.append(f"File size too small ({file_size} bytes < {self.min_size} bytes threshold). Likely corrupt stub.")
            return result

        if file_size > self.max_size:
            result.is_valid = False
            result.errors.append(f"File size exceeds safety limit ({file_size / (1024*1024):.1f} MB > {self.max_size / (1024*1024):.1f} MB).")
            return result

        try:
            with open(file_path, "rb") as f:
                header = f.read(5)
                if not header.startswith(b"%PDF"):
                    result.is_valid = False
                    result.errors.append(f"Invalid magic bytes ({header!r}). File is not a valid PDF.")
                    return result
        except IOError as e:
            result.is_valid = False
            result.errors.append(f"Unable to read file header: {e}")
            return result

        # -------------------------------------------------------------
        # Gate 2: Security & Encryption
        # -------------------------------------------------------------
        try:
            doc = fitz.open(file_path)
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"PyMuPDF failed to parse document structure: {e}")
            return result

        if doc.is_encrypted:
            result.is_encrypted = True
            # Attempt to authenticate with empty password (some PDFs are encrypted with blank user passwords)
            if not doc.authenticate(""):
                result.is_valid = False
                result.needs_password = True
                result.errors.append("Document is password-protected or encrypted. Cannot parse automated text.")
                doc.close()
                return result

        # -------------------------------------------------------------
        # Gate 3: Structural Integrity & Page Limits
        # -------------------------------------------------------------
        page_count = doc.page_count
        result.page_count = page_count

        if page_count < MIN_PAGES:
            result.is_valid = False
            result.errors.append("Document contains 0 pages.")
            doc.close()
            return result

        if page_count > MAX_PAGES_WARNING:
            result.warnings.append(f"Large document ({page_count} pages). Ingestion may take several minutes.")

        # -------------------------------------------------------------
        # Gate 4: Content Readability & Font Health
        # -------------------------------------------------------------
        total_chars = 0
        printable_chars = 0
        first_substantive_sample = ""
        printable_set = set(string.printable)

        for page_idx in range(page_count):
            try:
                page = doc.load_page(page_idx)
            except Exception as e:
                result.is_valid = False
                result.errors.append(f"Failed to load page index {page_idx + 1}: {e}")
                doc.close()
                return result

            page_text = page.get_text() or ""
            cleaned_text = page_text.strip()
            char_count = len(cleaned_text)
            total_chars += char_count

            for ch in cleaned_text:
                if ch in printable_set or ch in "\n\r\t":
                    printable_chars += 1

            if char_count >= self.min_text_len:
                result.native_text_pages += 1
                if not first_substantive_sample:
                    first_substantive_sample = cleaned_text[:350].replace("\n", " ")
            else:
                # Check if page contains images (scanned page) or is purely blank
                images = page.get_images()
                if images:
                    result.scanned_pages += 1
                else:
                    result.blank_pages += 1

        doc.close()

        # Encoding health calculation
        if total_chars > 0:
            health_score = printable_chars / total_chars
            result.encoding_health_score = round(health_score, 3)
            if health_score < PRINTABLE_CHAR_RATIO_THRESHOLD:
                result.warnings.append(
                    f"Low printable character ratio ({health_score:.1%}). "
                    "Fonts may have corrupted CMap tables. OCR will be prioritized."
                )

        # Scanned pages check vs Tesseract availability
        if result.scanned_pages > 0 and not self.tesseract_installed:
            result.warnings.append(
                f"{result.scanned_pages} page(s) appear scanned, but system Tesseract OCR is not installed. "
                "These pages will lack searchable text until tesseract-ocr is installed."
            )

        if not first_substantive_sample:
            if result.scanned_pages > 0:
                result.summary_preview = f"Scanned document ({result.scanned_pages} pages). Text pending OCR extraction."
            else:
                result.is_valid = False
                result.errors.append("Document has no readable text and no embedded images.")
                return result
        else:
            result.summary_preview = first_substantive_sample

        return result

    def validate_and_route(self, file_path: str, move_files: bool = True) -> ValidationResult:
        """
        Validates a file from staging.
        If valid: promotes it to PDF_SOURCE_DIR.
        If invalid: quarantines it into PDF_QUARANTINE_DIR with an audit report.
        """
        os.makedirs(PDF_SOURCE_DIR, exist_ok=True)
        os.makedirs(PDF_QUARANTINE_DIR, exist_ok=True)

        result = self.validate_file(file_path)
        file_name = result.file_name

        if result.is_valid:
            dest_path = os.path.join(PDF_SOURCE_DIR, file_name)
            if move_files and file_path != dest_path:
                shutil.copy2(file_path, dest_path)
                print(f"[PROMOTED] '{file_name}' -> {PDF_SOURCE_DIR}")
        else:
            dest_path = os.path.join(PDF_QUARANTINE_DIR, file_name)
            audit_path = os.path.join(PDF_QUARANTINE_DIR, f"{file_name}_audit.json")
            if move_files:
                shutil.copy2(file_path, dest_path)
                with open(audit_path, "w", encoding="utf-8") as f:
                    json.dump(result.model_dump(), f, indent=4)
                print(f"[QUARANTINED] '{file_name}' -> {PDF_QUARANTINE_DIR} (Reason: {result.errors})")

        return result


def audit_staging_directory():
    """Audits all files currently residing in staging/."""
    print("=" * 70)
    print("INGESTION HYGIENE AUDIT: Scanning pdf_files/staging/")
    print("=" * 70)

    if not os.path.exists(PDF_STAGING_DIR):
        print(f"Staging directory not found: {PDF_STAGING_DIR}")
        return

    files = sorted([f for f in os.listdir(PDF_STAGING_DIR) if not f.startswith(".")])
    if not files:
        print("No files found in staging directory.")
        return

    validator = PDFValidator()
    print(f"System OCR Engine Available: {validator.tesseract_installed}\n")

    for f in files:
        file_path = os.path.join(PDF_STAGING_DIR, f)
        res = validator.validate_and_route(file_path, move_files=True)
        status = "PASSED" if res.is_valid else "FAILED"
        print(f"[{status}] {f}")
        print(f"  - Size: {res.file_size_bytes / (1024*1024):.2f} MB | Pages: {res.page_count}")
        print(f"  - Text Pages: {res.native_text_pages} | Scanned Pages: {res.scanned_pages} | Blank Pages: {res.blank_pages}")
        print(f"  - Font Health: {res.encoding_health_score:.1%}")
        if res.warnings:
            print(f"  - Warnings: {res.warnings}")
        if res.errors:
            print(f"  - Errors: {res.errors}")
        print(f"  - Content Sample: {res.summary_preview[:180]}...")
        print("-" * 70)


if __name__ == "__main__":
    audit_staging_directory()
