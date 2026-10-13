import fitz  # PyMuPDF
import os
import re
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import io
import pytesseract

# High-resolution DPI for OCR to ensure punctuation and decimals are captured
OCR_RENDER_DPI = 200

# Tesseract flags: OEM 1 (LSTM engine), PSM 3 (Auto page segmentation), preserve interword spacing
TESSERACT_CONFIG = "--oem 1 --psm 3 -c preserve_interword_spaces=1"

# Threshold for triggering OCR
MIN_TEXT_LENGTH_FOR_OCR_CHECK = 80


def preprocess_image_for_ocr(pil_image: Image.Image) -> Image.Image:
    """
    Applies grayscale conversion and contrast stretching so faint punctuation
    and fine details (like decimal points) stand out sharply against the background.
    """
    # 1. Convert to grayscale
    gray_img = pil_image.convert("L")
    # 2. Normalize and stretch contrast
    autocontrast_img = ImageOps.autocontrast(gray_img, cutoff=1)
    # 3. Apply mild contrast boost
    enhancer = ImageEnhance.Contrast(autocontrast_img)
    enhanced_img = enhancer.enhance(1.4)
    return enhanced_img


def clean_extracted_text(text: str) -> str:
    """
    Cleans raw OCR text:
    - Removes common demo/scanner watermarks.
    - Repaints broken decimal numbers (e.g. '8 . 3' -> '8.3').
    - Joins words broken across line-break hyphens (e.g. 'opera-\\ntional' -> 'operational').
    - Normalizes excessive whitespace.
    """
    if not text:
        return ""

    # 1. Remove Document2PDF Pilot watermark
    text = re.sub(
        r'[\u2018\u2019\'"]?This document is created with the unregistered version of Document2PDF Pilot\.?',
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 2. Fix spaced decimals (e.g. "8 . 3" or "8 , 3" -> "8.3")
    text = re.sub(r"(?<=\b\d)\s*[\.,]\s*(?=\d\b)", ".", text)

    # 3. De-hyphenate words split across line breaks (e.g. "opera-\ntional" -> "operational")
    text = re.sub(r"(\b\w+)-\n(\w+\b)", r"\1\2", text)

    # 4. Normalize excessive newlines (max 2 consecutive newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def process_pdf(pdf_path: str, output_dir: str):
    """
    Extracts text page by page with high-resolution OCR fallback.
    Extracts embedded raster images filtering by size and pixel variation.

    Args:
        pdf_path (str): The path to the PDF file.
        output_dir (str): The directory where extracted artifacts are saved.

    Returns:
        tuple: (list of page strings, list of saved image paths)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    extracted_texts = []
    saved_image_paths = []

    min_width = 50
    min_height = 50
    min_std_dev = 5

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # Native text extraction first
            text = page.get_text() or ""
            cleaned_native = clean_extracted_text(text)

            # Detect whether OCR is required:
            # 1. Native text is shorter than threshold, OR
            # 2. Page has large embedded raster images covering page with minimal text
            needs_ocr = len(cleaned_native) < MIN_TEXT_LENGTH_FOR_OCR_CHECK

            if needs_ocr:
                print(f"Page {page_num+1}: Low text density ({len(cleaned_native)} chars). Rendering at {OCR_RENDER_DPI} DPI for OCR...")
                try:
                    # Render high-resolution pixmap for crisp punctuation retention
                    pix = page.get_pixmap(dpi=OCR_RENDER_DPI)
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                    # Contrast preprocessing
                    preprocessed_img = preprocess_image_for_ocr(pil_image)

                    # Execute Tesseract with optimal parameters
                    ocr_raw = pytesseract.image_to_string(preprocessed_img, lang="eng", config=TESSERACT_CONFIG)
                    cleaned_ocr = clean_extracted_text(ocr_raw)

                    if len(cleaned_ocr) > len(cleaned_native):
                        text = cleaned_ocr
                        print(f"Page {page_num+1}: OCR successful ({len(cleaned_ocr)} chars extracted).")
                    else:
                        text = cleaned_native
                        print(f"Page {page_num+1}: Native text was preferred.")

                except Exception as ocr_e:
                    print(f"Page {page_num+1}: OCR failed: {ocr_e}. Falling back to native text.")
                    text = cleaned_native
            else:
                text = cleaned_native

            extracted_texts.append(text)

            # Extract embedded images if any
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    if base_image["width"] > min_width and base_image["height"] > min_height:
                        image_stream = io.BytesIO(image_bytes)
                        pil_img = Image.open(image_stream)
                        if pil_img.mode not in ["L", "LA"]:
                            pil_img = pil_img.convert("L")
                        image_array = np.array(pil_img)
                        std_dev = np.std(image_array)

                        if std_dev > min_std_dev:
                            embedded_image_filename = os.path.join(
                                output_dir, f"page{page_num+1}_embedded_img{img_index+1}.{image_ext}"
                            )
                            with open(embedded_image_filename, "wb") as img_file:
                                img_file.write(image_bytes)
                            saved_image_paths.append(embedded_image_filename)
                except Exception as img_e:
                    print(f"Warning: Could not process embedded image {xref} on page {page_num+1}: {img_e}")

        doc.close()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        raise

    return extracted_texts, saved_image_paths
