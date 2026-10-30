import json
import urllib.request
import urllib.error
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DB_DIR = BASE_DIR / "vectordb"
MODEL_DIR = BASE_DIR / "models" / "all-MiniLM-L6-v2"

COLLECTION_NAME = "localdocai_documents"

LLM_API_URL = "http://127.0.0.1:5001/v1/chat/completions"

TOP_K = 3


# --------------------------------------------------
# CONNECT TO LOCAL LANGUAGE MODEL
# --------------------------------------------------

def generate_answer(question, retrieved_passages):

    context = ""

    for i, passage in enumerate(retrieved_passages, start=1):

        context += f"""
SOURCE {i}
Filename: {passage['source']}
Page: {passage['page']}

Content:
{passage['text']}

--------------------------------
"""

    system_prompt = """
You are LocalDocAI, a privacy-focused document assistant.

Answer the user's question using ONLY the supplied document passages.

Rules:
1. Do not use outside knowledge.
2. Do not invent facts or citations.
3. If the passages do not contain enough information, clearly say:
   "I could not find enough information to answer this in the provided documents."
4. Keep answers clear and direct.
5. Mention the source filename and page number supporting your answer.
6. If multiple passages support the answer, mention all relevant sources.
"""

    user_prompt = f"""
DOCUMENT PASSAGES:

{context}

USER QUESTION:
{question}

Answer using only the document passages above.
"""

    payload = {
        "model": "koboldcpp/Qwen2.5-1.5B-Instruct-Q4_K_M",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": 400,
        "stream": False
    }

    request = urllib.request.Request(
        LLM_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(request, timeout=120) as response:

            result = json.loads(response.read().decode("utf-8"))

            return result["choices"][0]["message"]["content"]

    except urllib.error.URLError as error:

        print("\nCould not connect to KoboldCpp.")
        print("Make sure KoboldCpp is running with Qwen loaded.")
        print("Details:", error)

        return None

    except (KeyError, IndexError, json.JSONDecodeError) as error:

        print("\nUnexpected response from the local AI model.")
        print("Details:", error)

        return None


# --------------------------------------------------
# MAIN RAG PIPELINE
# --------------------------------------------------

def main():

    print("=" * 60)
    print("             LocalDocAI - RAG Assistant")
    print("=" * 60)

    # 1. Verify required files

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Embedding model not found: {MODEL_DIR}"
        )

    if not DB_DIR.exists():
        raise FileNotFoundError(
            f"Vector database not found: {DB_DIR}"
        )

    # 2. Load local embedding model

    print("\nLoading local embedding model...")

    model = SentenceTransformer(
        str(MODEL_DIR),
        local_files_only=True
    )

    # 3. Connect to ChromaDB

    print("Connecting to vector database...")

    client = chromadb.PersistentClient(
        path=str(DB_DIR)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    total_chunks = collection.count()

    if total_chunks == 0:
        raise ValueError("The document database is empty.")

    print("\nLocal models and database loaded successfully!")
    print("Stored document chunks:", total_chunks)

    print("\nAsk questions about your documents.")
    print("Type 'exit' to close LocalDocAI.")

    # 4. Question-answering loop

    while True:

        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\nClosing LocalDocAI.")
            break

        if not question:
            print("Please enter a question.")
            continue

        # 5. Convert question into an embedding

        query_vector = model.encode(
            [question],
            normalize_embeddings=True
        )

        # 6. Retrieve relevant document passages

        results = collection.query(
            query_embeddings=query_vector.tolist(),
            n_results=min(TOP_K, total_chunks),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        retrieved_passages = []

        for i, document in enumerate(results["documents"][0]):

            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]

            retrieved_passages.append({
                "source": metadata["source"],
                "page": metadata["page"],
                "text": document,
                "distance": distance
            })

        if not retrieved_passages:
            print("\nNo relevant document passages were retrieved.")
            continue

        # 7. Generate answer using local Qwen model

        print("\nLocalDocAI is generating an answer...")

        answer = generate_answer(
            question,
            retrieved_passages
        )

        if answer is None:
            continue

        # 8. Display answer

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)

        print("\n" + answer)

        # 9. Display retrieved source information

        print("\n" + "-" * 60)
        print("RETRIEVED SOURCES")
        print("-" * 60)

        for i, passage in enumerate(retrieved_passages, start=1):

            print(f"\nSource {i}: {passage['source']}")
            print("Page:", passage["page"])
            print(
                "Cosine distance:",
                round(passage["distance"], 4)
            )

        print("\n" + "=" * 60)


if __name__ == "__main__":
    main()