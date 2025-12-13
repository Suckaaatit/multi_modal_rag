import fitz  # PyMuPDF
from mistralai import Mistral
from llama_index.core.schema import TextNode
import time
import base64
import io
import pdfplumber
import pytesseract
from PIL import Image

class PDFIngestionPipeline:
    def __init__(self, api_key):
        # Initialize Mistral for Vision tasks
        self.client = Mistral(api_key=api_key)

    def _describe_image(self, image_bytes):
        """
        Sends image to Mistral to get a text description.
        This allows charts to be indexed textually.
        """
        # Convert image bytes to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')

        # Simple retry logic for API stability
        for attempt in range(2):
            try:
                # Try to use pixtral for images, fall back to text-only if not available
                response = self.client.chat.complete(
                    model="pixtral-12b-2409",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Describe this image details (data, text). If decorative, say DECORATIVE."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/png;base64,{base64_image}"
                                }
                            ]
                        }
                    ]
                )
                text = response.choices[0].message.content if response.choices else ""
                if "DECORATIVE" in text: return ""
                return text
            except Exception as e:
                time.sleep(1)
        return ""

    def process_pdf(self, file_stream, file_name):
        # Open PDF from memory stream
        doc = fitz.open(stream=file_stream, filetype="pdf")
        nodes = []
        
        print(f"\n=== DEBUG: Processing PDF: {file_name} ===")
        print(f"Number of pages: {len(doc)}")

        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            try:
                page_label = page.get_label()
            except Exception:
                page_label = str(page_num)
            
            # --- 1. Text Ingestion ---
            text = page.get_text()
            print(f"\n--- Page {page_num} ---")
            print(f"Text extracted: {len(text)} characters")
            if text.strip():
                print(f"Text preview: {text[:200]}...")
                nodes.append(TextNode(
                    text=text,
                    metadata={"page": page_num, "label": page_label, "file": file_name, "type": "text"}
                ))
            else:
                # OCR fallback for scanned pages
                try:
                    pix = page.get_pixmap(dpi=200)
                    png_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(png_bytes))
                    ocr_text = pytesseract.image_to_string(img)
                    if ocr_text.strip():
                        print(f"OCR extracted: {len(ocr_text)} characters")
                        nodes.append(TextNode(
                            text=ocr_text,
                            metadata={"page": page_num, "label": page_label, "file": file_name, "type": "ocr_text"}
                        ))
                except Exception as e:
                    print(f"OCR failed on page {page_num}: {e}")

            # --- 2. Image Ingestion (Multi-modal) ---
            image_list = page.get_images(full=True)
            print(f"Images found: {len(image_list)}")
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Filter tiny images (logos, lines)
                    if len(image_bytes) > 5000: 
                        # Convert Chart -> Text
                        desc = self._describe_image(image_bytes)
                        if desc:
                            print(f"Image {img_index+1} description: {desc[:100]}...")
                            nodes.append(TextNode(
                                text=f"[IMAGE]: {desc}",
                                metadata={"page": page_num, "label": page_label, "file": file_name, "type": "chart"}
                            ))
                except Exception as e:
                    print(f"Error processing image {img_index+1}: {e}")
                    continue

        # --- 3. Table Extraction via pdfplumber ---
        try:
            with pdfplumber.open(io.BytesIO(file_stream)) as pdf:
                for p_idx, p in enumerate(pdf.pages):
                    try:
                        tables = p.extract_tables()
                        for t_idx, table in enumerate(tables or []):
                            # Convert table to CSV-like lines
                            lines = []
                            for row in table:
                                if row:
                                    cells = [c.strip() if isinstance(c, str) else "" for c in row]
                                    lines.append(",".join(cells))
                            table_text = "\n".join(lines).strip()
                            if table_text:
                                # Derive label for plumber page if possible (may differ in annexes)
                                try:
                                    lbl = doc[p_idx].get_label()
                                except Exception:
                                    lbl = str(p_idx+1)
                                nodes.append(TextNode(
                                    text=f"[TABLE]:\n{table_text}",
                                    metadata={"page": p_idx+1, "label": lbl, "file": file_name, "type": "table"}
                                ))
                    except Exception as e:
                        print(f"Table extraction failed on page {p_idx+1}: {e}")
        except Exception as e:
            print(f"pdfplumber open failed: {e}")
        
        print(f"\n=== Total nodes created: {len(nodes)} ===")
        return nodes
