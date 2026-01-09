# Chat with PDF 📄

An intelligent PDF chatbot powered by Google Gemini AI, built with Streamlit.

## Features

- 📄 Upload and analyze PDF documents
- 💬 Natural language Q&A interface
- 🔍 Context-based answers with source citations
- 🎨 Modern, animated UI with Gemini-inspired design
- 🧠 RAG (Retrieval-Augmented Generation) pipeline
- 💾 ChromaDB vector storage
- 🤖 Powered by Google Gemini

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Get Google API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy your API key

## Deployment

This app can be deployed for free on [Streamlit Community Cloud](https://streamlit.io/cloud).

## Built by

**Shahadat Sagor**

## Tech Stack

- **Frontend:** Streamlit
- **LLM:** Google Gemini (gemini-flash-latest)
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **Vector Store:** ChromaDB
- **Framework:** LangChain
