\# LocalDocAI



LocalDocAI is a privacy-focused PDF question-answering assistant built around local document retrieval and language-model inference.



\## Features



\- Upload and process PDF documents

\- Extract text and split it into chunks

\- Generate local embeddings for semantic search

\- Retrieve relevant document passages to answer questions

\- Maintain document-based answers with source citations

\- Store document vectors locally using ChromaDB

\- Use a locally hosted language model through a KoboldCpp-compatible API



\## Technology Stack



\- Python

\- Streamlit

\- PyMuPDF

\- Sentence Transformers

\- ChromaDB

\- KoboldCpp

\- Retrieval-Augmented Generation (RAG)



\## Prerequisites



\- Python 3.12

\- A locally downloaded Sentence Transformers embedding model

\- A compatible language model running through KoboldCpp

\- The required Python packages



\## Installation



1\. Clone this repository:



&#x20;  ```bash

&#x20;  git clone https://github.com/manhatabassum-tajudeen/LocalDocAI.git

&#x20;  cd LocalDocAI

&#x20;  ```



2\. Create and activate a virtual environment.



&#x20;  Windows:



&#x20;  ```cmd

&#x20;  python -m venv .venv

&#x20;  .venv\\Scripts\\activate

&#x20;  ```



3\. Install the dependencies:



&#x20;  ```cmd

&#x20;  pip install -r requirements.txt

&#x20;  ```



4\. Configure your local embedding model and start your KoboldCpp server using the settings required by the application.



5\. Run the application:



&#x20;  ```cmd

&#x20;  streamlit run app.py

&#x20;  ```



\## Privacy Notes



Document processing and inference are designed to use local components. Actual offline operation depends on the model files being available locally and the application and its dependencies not requiring network access. Verify offline operation before describing the project as fully offline.



\## Project Status



This project is under development. Setup instructions and features may evolve as the application is tested.



\## License



No license has been added yet.



