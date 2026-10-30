
import json
from pathlib import Path

import numpy as np
import chromadb

# Project paths
BASE_DIR = Path(__file__).resolve().parent

CHUNKS_FILE = BASE_DIR / "chunks" / "chunks.json"
VECTORS_FILE = BASE_DIR / "embeddings" / "vectors.npy"
DB_DIR = BASE_DIR / "vectordb"

COLLECTION_NAME = "localdocai_documents"

def main():

    print("Starting LocalDocAI vector database setup...")

    # 1. Check required files
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {CHUNKS_FILE}")

    if not VECTORS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {VECTORS_FILE}")

    # 2. Load document chunks
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    # 3. Load embeddings
    vectors = np.load(VECTORS_FILE)

    # 4. Validate data
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match "
            f"embedding count ({len(vectors)})."
        )

    if vectors.ndim != 2 or vectors.shape[1] != 384:
        raise ValueError(
            f"Expected embeddings with shape (n, 384), "
            f"but received {vectors.shape}."
        )

    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    ids = [chunk["chunk_id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs detected.")

    # 5. Prepare metadata and documents
    documents = [chunk["text"] for chunk in chunks]

    metadatas = [
        {
            "source": chunk["source"],
            "page": int(chunk["page"]),
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]

    # 6. Create persistent ChromaDB client
    DB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    # 7. Create or access collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # 8. Insert embeddings and associated data
    # Upsert prevents duplicate records when rerunning this script.
    collection.upsert(
        ids=ids,
        embeddings=vectors.astype(np.float32).tolist(),
        documents=documents,
        metadatas=metadatas
    )

    # 9. Verify database contents
    total = collection.count()

    print("\nDatabase setup completed!")
    print("--------------------------------")
    print("Collection:", COLLECTION_NAME)
    print("Database location:", DB_DIR)
    print("Chunks processed:", len(chunks))
    print("Embedding dimensions:", vectors.shape[1])
    print("Records in database:", total)
    print("--------------------------------")

    if total != len(chunks):
        raise RuntimeError("Database record count verification failed.")

    print("SUCCESS: All document chunks stored in ChromaDB.")

if __name__ == "__main__":
    main()
