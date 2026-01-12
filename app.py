import os
import tempfile
import streamlit as st

# Import Backend Logic
from logic import RAGPipeline

# --- Page Config ---
st.set_page_config(page_title="Chat with PDF", layout="wide", page_icon="📄")

# --- Enhanced Custom CSS (Loaded from assets) ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("assets/style.css")

# --- Logic Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource(show_spinner=False)
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📄 Document")
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf", label_visibility="collapsed")
    
    # --- Model Selector ---
    st.markdown("### 🧠 AI Model")
    model_options = [
        "gemini-flash-latest",
        "Llama 3.3 70B (Groq)",
        "Llama 3.1 8B (Groq)",
        "Qwen 2.5 Coder (HuggingFace)"
    ]
    selected_model = st.selectbox(
        "Choose Model", 
        model_options, 
        index=0, 
        label_visibility="collapsed"
    )
    
    # Cloud Badge
    st.markdown("""
        <div style='background-color: rgba(30, 41, 59, 0.8); border: 1px solid rgba(99, 102, 241, 0.4); border-radius: 8px; padding: 10px; margin-top: 5px; margin-bottom: 20px;'>
            <p style='margin: 0; font-size: 0.8rem; color: #a5b4fc; text-align: center;'>
                ☁️ <b>All models run in Cloud</b><br>
                (No Local CPU usage)
            </p>
        </div>
    """, unsafe_allow_html=True)

    # File Processing
    if uploaded_file and "processed_file" not in st.session_state:
        st.session_state.processed_file = None

    if uploaded_file:
        if st.session_state.get("processed_file") != uploaded_file.name:
            with st.spinner("🔄 Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                success, message = pipeline.ingest_pdf(tmp_file_path)
                os.remove(tmp_file_path)

                if success:
                    st.session_state.processed_file = uploaded_file.name
                    st.success(f"✅ {message}")
                else:
                    st.error(f"❌ {message}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built by Shahadat Sagor")

# --- Main Area ---
st.markdown('<h1 class="hero-header">Chat with PDF 📄</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Intelligent document analysis powered by Advanced Cloud AI</p>', unsafe_allow_html=True)

# Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📚 Sources"):
                for source in message["sources"]:
                    st.caption(f"> {source[:200]}...")

# Chat Input
user_input = st.chat_input("Ask anything about your document...")

if user_input:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response, error = pipeline.query(user_input, model_name=selected_model)
            
            if error:
                st.error(f"⚠️ {error}")
            elif response:  # Check if response exists before accessing
                answer = response["answer"]
                sources = [doc.page_content for doc in response["context"]]
                
                st.markdown(answer)
                
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.caption(f"> {source[:200]}...")
                        
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
