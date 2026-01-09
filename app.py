import os
import tempfile
import streamlit as st
import google.generativeai as genai     

# Python 3.9 compatibility patch
import importlib.metadata
try:
    importlib.metadata.packages_distributions()
except AttributeError:
    import importlib_metadata
    importlib.metadata.packages_distributions = importlib_metadata.packages_distributions

# Import Backend Logic
from logic import RAGPipeline

# --- Page Config ---
st.set_page_config(page_title="Chat with PDF", layout="wide", page_icon="📄")

# --- Enhanced Custom CSS (Gemini-Inspired) ---
st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500&display=swap');
        
        /* Global Font & Smoothing */
        * {
            font-family: 'Google Sans', 'Roboto', sans-serif;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }

        /* Hide Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        /* header {visibility: hidden;}  <-- Removed to show sidebar toggle */
        header {background-color: transparent !important;}
        
        /* Animated Colorful Gradient Background */
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* Sidebar - Match Main Content Styling */
        [data-testid="stSidebar"] {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        
        [data-testid="stSidebar"] h3 {
            color: #fff;
            font-weight: 500;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 1.5rem;
        }

        /* Vibrant Header with Multi-Color Gradient */
        .hero-header {
            background: linear-gradient(135deg, #FF6B6B 0%, #4ECDC4 20%, #45B7D1 40%, #F7DC6F 60%, #BB8FCE 80%, #F8B500 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-weight: 700;
            font-size: 3.5rem;
            text-align: center;
            margin: 2rem 0 0.5rem 0;
            animation: fadeInDown 0.8s ease-out, colorShift 8s ease infinite;
            background-size: 200% 200%;
        }
        
        @keyframes colorShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .hero-subtitle {
            text-align: center;
            color: #ffffff;
            font-size: 1rem;
            font-weight: 400;
            margin-bottom: 3rem;
            animation: fadeInUp 0.8s ease-out 0.2s backwards;
        }
        
        @keyframes fadeInDown {
            from {
                opacity: 0;
                transform: translateY(-20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Modern Buttons with Gradient */
        .stButton > button {
            background: linear-gradient(135deg, #FA8BFF 0%, #2BD2FF 50%, #2BFF88 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            font-size: 0.95rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 4px 15px rgba(250, 139, 255, 0.4);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) scale(1.02);
            box-shadow: 0 8px 25px rgba(250, 139, 255, 0.6);
            background: linear-gradient(135deg, #2BFF88 0%, #2BD2FF 50%, #FA8BFF 100%);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }

        /* File Uploader Styling */
        [data-testid="stFileUploader"] {
            background: rgba(30, 41, 59, 0.5);
            border: 2px dashed rgba(99, 102, 241, 0.4);
            border-radius: 12px;
            padding: 1.5rem;
            transition: all 0.3s ease;
        }
        
        [data-testid="stFileUploader"]:hover {
            border-color: rgba(99, 102, 241, 0.8);
            background: rgba(30, 41, 59, 0.7);
        }

        /* Selectbox Styling */
        .stSelectbox > div > div {
            background-color: rgba(30, 41, 59, 0.8);
            border-radius: 10px;
            border: 1px solid rgba(99, 102, 241, 0.3);
            color: white;
        }

        /* Chat Messages - Gemini Style */
        [data-testid="stChatMessage"] {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(148, 163, 184, 0.1);
            animation: slideIn 0.4s ease-out;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateX(-10px);
            }
            to {
                opacity: 1;
                transform: translateX(0);
            }
        }
        
        /* User Message Accent */
        [data-testid="stChatMessage"][data-testid*="user"] {
            border-left: 4px solid #4facfe;
            background: rgba(79, 172, 254, 0.05);
        }
        
        /* Assistant Message Accent */
        [data-testid="stChatMessage"][data-testid*="assistant"] {
            border-left: 4px solid #667eea;
            background: rgba(102, 126, 234, 0.05);
        }

        /* Chat Input - Colorful Gradient Border */
        .stChatInput > div {
            background: rgba(30, 41, 59, 0.6);
            border-radius: 24px;
            border: 3px solid;
            border-image: linear-gradient(90deg, #FA8BFF, #2BD2FF, #2BFF88, #FA8BFF) 1;
            transition: all 0.3s ease;
            position: relative;
        }
        
        .stChatInput > div::before {
            content: '';
            position: absolute;
            inset: -3px;
            border-radius: 24px;
            padding: 3px;
            background: linear-gradient(90deg, #FA8BFF, #2BD2FF, #2BFF88);
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            animation: borderRotate 3s linear infinite;
            opacity: 0.8;
        }
        
        @keyframes borderRotate {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
        
        .stChatInput > div:focus-within {
            background: rgba(30, 41, 59, 0.8);
            box-shadow: 0 0 30px rgba(250, 139, 255, 0.6), 
                        0 0 60px rgba(43, 210, 255, 0.4),
                        0 0 90px rgba(43, 255, 136, 0.3);
        }

        /* Expander Styling */
        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.3) !important;
            border-radius: 10px !important;
            border: 1px solid rgba(99, 102, 241, 0.2) !important;
            color: #cbd5e1 !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        .streamlit-expanderHeader:hover {
            background: rgba(30, 41, 59, 0.5) !important;
            border-color: rgba(99, 102, 241, 0.4) !important;
        }

        /* Success/Error Messages */
        .stSuccess {
            background: rgba(34, 197, 94, 0.1);
            border-left: 4px solid #22c55e;
            border-radius: 8px;
        }
        
        .stError {
            background: rgba(239, 68, 68, 0.1);
            border-left: 4px solid #ef4444;
            border-radius: 8px;
        }

        /* Spinner */
        .stSpinner > div {
            border-top-color: #667eea !important;
        }

        /* Markdown Content in Chat */
        [data-testid="stChatMessage"] p {
            line-height: 1.7;
            color: #e2e8f0;
        }
        
        /* Code Blocks */
        [data-testid="stChatMessage"] code {
            background: rgba(15, 23, 42, 0.6);
            padding: 0.2rem 0.4rem;
            border-radius: 4px;
            color: #a5b4fc;
        }
    </style>
""", unsafe_allow_html=True)

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
    
    selected_model = "gemini-flash-latest"

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
st.markdown('<p class="hero-subtitle">Intelligent document analysis powered by Google Gemini</p>', unsafe_allow_html=True)

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
            else:
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
