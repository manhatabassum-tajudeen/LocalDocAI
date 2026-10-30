
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Project paths
BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "chunks" / "chunks.json"
OUTPUT_DIR = BASE_DIR / "embeddings"
MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

# Load chunks
print("Loading document chunks...")

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
    chunks = json.load(file)

if not isinstance(chunks, list) or not chunks:
    raise ValueError("chunks.json must contain a non-empty list.")

texts = [chunk["text"] for chunk in chunks]

if any(not isinstance(t, str) or not t.strip() for t in texts):
    raise ValueError("Invalid or empty chunk text found.")

print(f"Loaded {len(texts)} chunks.")

# Load embedding model
print("\nLoading embedding model...")

if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
    model = SentenceTransformer(str(MODEL_DIR))
else:
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    model.save(str(MODEL_DIR))

# Generate embeddings
print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Save vectors
np.save(OUTPUT_DIR / "vectors.npy", embeddings)

# Save metadata
metadata = [
    {
        "chunk_id": chunk["chunk_id"],
        "source": chunk["source"],
        "page": chunk["page"],
        "text": chunk["text"]
    }
    for chunk in chunks
]

with open(OUTPUT_DIR / "metadata.json", "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4, ensure_ascii=False)

# Verify
print("\n========== SUCCESS ==========")
print("Chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("Dimensions:", embeddings.shape[1])
print("Vectors:", OUTPUT_DIR / "vectors.npy")
print("Metadata:", OUTPUT_DIR / "metadata.json")
print("=============================")
