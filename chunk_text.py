
from pathlib import Path
import json
import re

INPUT_FOLDER = Path("extracted_text")
OUTPUT_FOLDER = Path("chunks")

CHUNK_SIZE = 1000
OVERLAP = 150

OUTPUT_FOLDER.mkdir(exist_ok=True)

all_chunks = []


def split_into_chunks(text, chunk_size, overlap):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


for file_path in INPUT_FOLDER.glob("*.txt"):

    text = file_path.read_text(encoding="utf-8")

    # Separate text using the page markers
    pages = re.split(r"--- Page (\d+) ---", text)

    # Page text occurs after each page number
    for i in range(1, len(pages), 2):

        page_number = int(pages[i])
        page_text = pages[i + 1].strip()

        chunks = split_into_chunks(
            page_text,
            CHUNK_SIZE,
            OVERLAP
        )

        for index, chunk in enumerate(chunks, start=1):

            all_chunks.append({
                "chunk_id": f"{file_path.stem}_p{page_number}_c{index}",
                "source": file_path.name,
                "page": page_number,
                "text": chunk
            })

# Save all chunks and their metadata in one JSON file
output_file = OUTPUT_FOLDER / "chunks.json"

output_file.write_text(
    json.dumps(all_chunks, indent=4, ensure_ascii=False),
    encoding="utf-8"
)

print(f"Total chunks created: {len(all_chunks)}")
print(f"Saved to: {output_file.resolve()}")

for chunk in all_chunks:
    print(
        f"{chunk['chunk_id']} | "
        f"{chunk['source']} | "
        f"Page {chunk['page']}"
    )
