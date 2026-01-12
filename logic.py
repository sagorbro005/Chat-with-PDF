import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

class RAGPipeline:
    """
    Handles the Retrieval-Augmented Generation (RAG) pipeline operations:
    - Ingesting PDF documents
    - Creating and storing embeddings
    - Querying the vector database with an LLM
    """
    
    def __init__(self, persist_directory="db"):
        self.persist_directory = persist_directory
        # Initialize embeddings once
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.api_key = os.getenv("GOOGLE_API_KEY")

    def ingest_pdf(self, file_path):
        """
        Loads a PDF, splits it into chunks, and stores embeddings in ChromaDB.
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            tuple: (success (bool), message (str))
        """
        try:
            # Clear existing data using Chroma API (Avoids [WinError 32] file lock)
            vectorstore = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            try:
                vectorstore.delete_collection()
            except Exception:
                pass # Collection might not exist yet or error in deletion

            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split Text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Create/Update Vector Store
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            return True, "PDF Processed Successfully!"
        except Exception as e:
            return False, str(e)

    def query(self, question, model_name="gemini-flash-latest", k=20):
        """
        Queries the RAG pipeline with a user question.
        
        Args:
            question (str): User's question.
            model_name (str): Gemini model to use.
            k (int): Number of context chunks to retrieve.
            
        Returns:
            tuple: (response (dict), error (str))
                - response contains 'answer' and 'context'
        """
        if not os.path.exists(self.persist_directory):
            return None, "Database not found. Please upload a PDF first."

        try:
            # Load Vector Store
            db = Chroma(persist_directory=self.persist_directory, embedding_function=self.embeddings)
            retriever = db.as_retriever(search_kwargs={"k": k})

            # --- Model Switching Logic ---
            if "gemini" in model_name:
                # --- PROVIDER A: GOOGLE (Native) ---
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self.api_key,
                    temperature=0.2
                )
            
            elif model_name in ["Llama 3.3 70B (Groq)", "Llama 3.1 8B (Groq)"]:
                # --- PROVIDER B: GROQ (The Speed Demon) ---
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    return None, "Groq API Key not found in environment variables."
                
                model_map = {
                    "Llama 3.3 70B (Groq)": "llama-3.3-70b-versatile",
                    "Llama 3.1 8B (Groq)": "llama-3.1-8b-instant"
                }
                
                llm = ChatGroq(
                    model_name=model_map[model_name],
                    temperature=0,
                    api_key=groq_api_key
                )
                
            elif model_name == "Qwen 2.5 Coder (HuggingFace)":
                # --- PROVIDER C: HUGGINGFACE (Serverless Inference) ---
                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
                if not hf_token:
                    return None, "HuggingFace API Token not found in environment variables."

                # Initialize Endpoint (Wrapper around InferenceClient)
                endpoint = HuggingFaceEndpoint(
                    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
                    task="text-generation",
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.03,
                    huggingfacehub_api_token=hf_token
                )
                
                # Wrap in ChatHuggingFace to use the correct 'conservational' API task
                llm = ChatHuggingFace(llm=endpoint)

            else:
                return None, "Invalid model selection."

            # Professional & Detailed "Expert Consultant" Prompt
            prompt = ChatPromptTemplate.from_template("""
            You are a world-class AI Analyst and domain expert. Your goal is to provide a structured, data-driven, and highly professional answer based **exclusively** on the provided context.

            ### Instructions for Excellence:
            1.  **Direct & Concise**: Start immediately with the answer. Avoid filler phrases like "Based on the context" or "The document mentions".
            2.  **Structure**:
                -   **Executive Summary**: A 2-3 sentence high-level overview of the answer.
                -   **Detailed Analysis**: Break down the answer into logical sections with clear **Markdown Headers**.
                -   **Key Facts/Evidence**: Use **bullet points** to present data, lists, or steps for readability.
            3.  **Tone**: Professional, authoritative, yet accessible. Avoid robotic language.
            4.  **Formatting**: Use **BOLD** for key terms/concepts to make the text skimmable.
            5.  **Strict Constraint**: If the answer is NOT in the context, state clearly: *"The provided documents do not contain specific information regarding [topic]."* Do not hallucinate.

            ### Context Data:
            {context}

            ### User Question:
            {input}
            """)

            # Build Chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Execute
            response = retrieval_chain.invoke({"input": question})
            return response, None

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            
            # Check for 403 Permission Error
            if "403 Forbidden" in str(e) or "sufficient permissions" in str(e):
                error_msg = (
                    "**Permission Denied (403):** Your HuggingFace Token is Read-Only.\n"
                    "1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)\n"
                    "2. Create/Edit Token -> Select 'Fine-grained' or 'Write' permissions.\n"
                    "3. Enable **'Make calls to the serverless inference API'** under Inference."
                )
            else:
                error_msg = f"Error: {str(e)}\nInclude this in your report."
                
            print(f"DEBUG EXCEPTION:\n{error_details}") 
            return None, error_msg
