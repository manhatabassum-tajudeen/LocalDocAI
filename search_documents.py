
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Project directory
BASE_DIR = Path(__file__).resolve().parent

# Paths
DB_DIR = BASE_DIR / "vectordb"
MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"

COLLECTION_NAME = "localdocai_documents"

def main():

    print("Loading LocalDocAI semantic search...")

    # 1. Verify model and database
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Embedding model not found: {MODEL_DIR}")

    if not DB_DIR.exists():
        raise FileNotFoundError(f"Vector database not found: {DB_DIR}")

    # 2. Load the locally saved embedding model
    model = SentenceTransformer(
        str(MODEL_DIR),
        local_files_only=True
    )

    # 3. Connect to persistent ChromaDB
    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    print("Model and database loaded successfully!")
    print("Stored chunks:", collection.count())

    # 4. Accept user question
    while True:

        question = input("\nAsk a question (or type 'exit'): ").strip()

        if question.lower() == "exit":
            print("Closing semantic search.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # 5. Generate query embedding
        query_vector = model.encode(
            [question],
            normalize_embeddings=True
        )

        # 6. Search ChromaDB
        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # 7. Display retrieved passages
        print("\n" + "=" * 55)
        print("SEARCH RESULTS")
        print("=" * 55)

        for i, document in enumerate(results["documents"][0]):

            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            print(f"\nResult {i + 1}")
            print("-" * 40)

            print("Source:", metadata["source"])
            print("Page:", metadata["page"])
            print("Cosine distance:", round(distance, 4))

            print("\nRetrieved text:")
            print(document)

        print("\n" + "=" * 55)


if __name__ == "__main__":
    main()
