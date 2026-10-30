
import pymupdf
from pathlib import Path

# Define our project folders
PDF_FOLDER = Path("documents")
OUTPUT_FOLDER = Path("extracted_text")

# Create output folder if it doesn't exist
OUTPUT_FOLDER.mkdir(exist_ok=True)

# Find all PDF files
pdf_files = list(PDF_FOLDER.glob("*.pdf"))

if not pdf_files:
    print("No PDF files found in the documents folder.")

else:
    print(f"Found {len(pdf_files)} PDF(s).\n")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        try:
            document = pymupdf.open(pdf_path)
            extracted_pages = []

            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text()

                extracted_pages.append(
                    f"\n--- Page {page_number} ---\n{page_text}"
                )

            document.close()

            full_text = "\n".join(extracted_pages)

            # Save extracted text
            output_file = OUTPUT_FOLDER / f"{pdf_path.stem}.txt"

            output_file.write_text(
                full_text,
                encoding="utf-8"
            )

            print(f"  Pages: {len(extracted_pages)}")
            print(f"  Characters: {len(full_text)}")
            print(f"  Saved: {output_file}\n")

        except Exception as error:
            print(f"  Failed: {error}\n")

    print("PDF processing complete!")
