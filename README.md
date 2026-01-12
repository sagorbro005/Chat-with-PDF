# 📄 Chat with PDF | Professional RAG Pipeline

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)
![LangChain](https://img.shields.io/badge/Framework-LangChain_0.3-green)
![License](https://img.shields.io/badge/License-MIT-purple)

**Chat with PDF** is a production-ready **Retrieval-Augmented Generation (RAG)** application that allows users to converse with their PDF documents using state-of-the-art Cloud LLMs. It features a modern, glassmorphism-inspired UI and supports a multi-provider AI backend (Google, Groq, HuggingFace).

---

## � Features

### 🧠 Multi-Model Intelligence
Choose your preferred AI brain. Zero local computation required.
- **✨ Google Gemini 1.5 Flash**: Native, multimodal, and highly efficient.
- **⚡ Llama 3.3 70B (via Groq)**: Blazing fast inference speeds (~500 tokens/s).
- **💻 Qwen 2.5 Coder 32B (via HuggingFace)**: Top-tier open-source coding model.

### 🎨 Premium User Experience
- **Glassmorphism UI**: A sleek, modern interface with animated backgrounds and gradients.
- **Interactive Chat**: Natural language conversation with history awareness.
- **Source Transparency**: Every answer includes expandable citations from the original PDF.

### 🛠️ Technical Architecture
- **Framework**: LangChain 0.3 (Modern Ecosystem).
- **Vector Database**: ChromaDB (v0.5.3) for persistent embeddings.
- **Embeddings**: `all-MiniLM-L6-v2` via HuggingFace (Serverless).
- **Chunking**: Recursive Character Splitter for optimal context retrieval.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/chat-with-pdf.git
cd chat-with-pdf
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
Create a `.env` file in the root directory and add your API keys.
**Note:** You only need the keys for the models you intend to use.

```ini
# .env file
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token
```

#### 🔑 Get Your FREE Keys:
- **Google Gemini**: [AI Studio](https://makersuite.google.com/app/apikey)
- **Groq Cloud**: [Groq Console](https://console.groq.com/keys)
- **HuggingFace**: [Settings > Tokens](https://huggingface.co/settings/tokens)
  - *Important:* Ensure the HF Token has **"Make calls to serverless inference API"** permission enabled.

---

## 🏃‍♂️ Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

1.  **Upload**: Drag and drop your PDF in the sidebar.
2.  **Select Model**: Choose between Gemini, Llama, or Qwen from the dropdown.
3.  **Chat**: Type your question. The AI will analyze the document and respond with citations.

---

## 📂 Project Structure

```bash
chat-with-pdf/
├── app.py                 # Main Streamlit Frontend Application
├── logic.py               # RAG Pipeline & LangChain Logic
├── requirements.txt       # Project Dependencies
├── .env                   # Environment Variables (Not committed)
└── db/                    # ChromaDB Persistence Directory
```

---

## 🤝 Contributing

Contributions are welcome!
1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes.
4.  Open a Pull Request.

---

## 👨‍💻 Author

**Shahadat Sagor**
*Software Engineer & AI Enthusiast*

---

*Powered by LangChain 0.3, Streamlit, and Coffee ☕*
