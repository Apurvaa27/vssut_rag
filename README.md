# VSSUT RAG

A document question-answering project using retrieval-augmented generation (RAG).
It loads PDF, TXT, Markdown, and PHP files, finds relevant passages, and uses Google
Gemini to answer from those passages.

## Features

- Load owner-managed PDF, TXT, Markdown, and PHP documents from `data/`
- Create embeddings with Sentence Transformers
- Store and search document chunks with Chroma
- Let visitors ask questions in a chat interface and view source file/page citations
- Use the command line for ingestion and questions

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your Google AI Studio API key:

```env
GOOGLE_API_KEY=your_api_key_here
```

Never commit `.env` or your API key.

## Run the Streamlit app

```bash
streamlit run streamlit_app.py
```

Add or update PDF, TXT, Markdown, or PHP files anywhere under `data/`. PHP files
are converted to Markdown during ingestion. The app indexes those files in the background and
exposes only the chat interface to visitors. It shows source citations below
each answer. Locally, restart Streamlit after changing documents so the updated
index is loaded.

## Command-line usage

Put source files into `data/`, then index and ask questions:

```bash
python app.py ingest
python app.py ask "What is the grading system?"
python app.py chat
```

## Deploy on Streamlit Community Cloud

1. Add the documents you want to make available to visitors to `data/`, then
   push the project to GitHub. Only include documents you are allowed to publish.
   Do not commit `.env` or API keys.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and choose
   **Create app**. Select your repository, branch, and `streamlit_app.py` as the
   app file.
3. In **Advanced settings → Secrets**, add:

   ```toml
   GOOGLE_API_KEY = "your_api_key_here"
   ```

4. Deploy and share the app URL.

Visitors cannot upload or change documents through the UI. To update the
knowledge base, add/remove files in `data/` and push the changes; Streamlit
Cloud will redeploy and rebuild the index in the background. The Chroma index is
stored on the app server and may be lost when the app restarts; the app rebuilds
it from the documents in the repository. Avoid publishing sensitive documents.

## Project structure

```text
├── app.py                # Command-line interface
├── streamlit_app.py      # Streamlit UI
├── data/                 # Optional local source documents
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── rag.py
│   └── vector_store.py
├── requirements.txt
└── README.md
```