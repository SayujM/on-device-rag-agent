import fitz  # PyMuPDF
import os
import numpy as np
from PIL import Image
import io
import pytesseract # New import for OCR

# Define a threshold for when to trigger OCR
MIN_TEXT_LENGTH_FOR_OCR_CHECK = 50 # If extracted text is less than this, try OCR

def process_pdf(pdf_path: str, output_dir: str):
    """
    Extracts text page by page and renders each page as an image.
    Also attempts to extract embedded raster images, filtering by size and pixel variation.
    For scanned documents, OCR is used to extract text if PyMuPDF's text extraction is insufficient.

    Args:
        pdf_path (str): The path to the PDF file.
        output_dir (str): The directory where extracted images and rendered pages will be saved.

    Returns:
        tuple: A tuple containing:
            - list: A list of strings, where each string is the text content of a page.
            - list: A list of strings, where each string is the path to a saved image (either rendered page or embedded raster).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)

    extracted_texts = []
    saved_image_paths = []

    # Define minimum thresholds for embedded image filtering
    min_width = 50
    min_height = 50
    min_std_dev = 5 # Standard deviation of pixel intensities (0-255 range for grayscale)

    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)

            # Render page as an image (PNG) - captures all visual content including vector graphics
            pix = page.get_pixmap()
            page_image_filename = os.path.join(output_dir, f"page{page_num+1}.png")
            pix.save(page_image_filename)
            saved_image_paths.append(page_image_filename)

            # Extract text
            text = page.get_text()
            
            # --- New: Conditional OCR for scanned documents ---
            if len(text.strip()) < MIN_TEXT_LENGTH_FOR_OCR_CHECK:
                print(f"Page {page_num+1}: Insufficient text extracted by PyMuPDF. Attempting OCR...")
                try:
                    # Convert pixmap to PIL Image for Tesseract
                    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    # Perform OCR (lang='eng' for Version 1)
                    ocr_text = pytesseract.image_to_string(pil_image, lang='eng')
                    if len(ocr_text.strip()) > len(text.strip()): # Use OCR text if it's more substantial
                        text = ocr_text
                        print(f"Page {page_num+1}: OCR successful. Using OCR'd text.")
                    else:
                        print(f"Page {page_num+1}: OCR did not yield more substantial text. Using PyMuPDF text.")
                except Exception as ocr_e:
                    print(f"Page {page_num+1}: OCR failed: {ocr_e}. Using PyMuPDF text.")
            # --- End New ---

            extracted_texts.append(text)

            # Attempt to extract embedded raster images (if any), applying filters
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]

                    # Filter 1: Size check
                    if base_image["width"] > min_width and base_image["height"] > min_height:
                        # Filter 2: Variation check (using standard deviation of pixel intensities)
                        try:
                            image_stream = io.BytesIO(image_bytes)
                            pil_image = Image.open(image_stream)
                            # Convert to grayscale for simpler standard deviation calculation
                            if pil_image.mode not in ['L', 'LA']:
                                pil_image = pil_image.convert('L')
                            image_array = np.array(pil_image)
                            std_dev = np.std(image_array)

                            if std_dev > min_std_dev:
                                # Save image if both filters pass
                                embedded_image_filename = os.path.join(output_dir, f"page{page_num+1}_embedded_img{img_index+1}.{image_ext}")
                                with open(embedded_image_filename, "wb") as img_file:
                                    img_file.write(image_bytes)
                                saved_image_paths.append(embedded_image_filename)
                            else:
                                print(f"Skipping embedded image {xref} on page {page_num+1}: Low variation (std_dev={std_dev:.2f})")
                        except Exception as data_e:
                            print(f"Warning: Could not process image data for variation check (xref {xref}, page {page_num+1}): {data_e}")
                            # If image data processing fails, we skip this image
                    else:
                        print(f"Skipping embedded image {xref} on page {page_num+1}: Too small (width={base_image['width']}, height={base_image['height']})")

                except Exception as img_e:
                    print(f"Warning: Could not extract embedded image {xref} from page {page_num+1}: {img_e}")
        doc.close()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        raise

    return extracted_texts, saved_image_paths

