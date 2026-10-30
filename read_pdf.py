
import pymupdf

# Ask the user for the PDF location
pdf_path = input("Enter the full path of your PDF: ").strip().strip('"')

try:
    # Open the PDF
    document = pymupdf.open(pdf_path)

    print(f"\nPDF opened successfully!")
    print(f"Number of pages: {len(document)}")

    # Extract text page by page
    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        print(f"\n--- Page {page_number} ---")
        print(text[:1500])

    document.close()

except Exception as error:
    print(f"\nSomething went wrong: {error}")
