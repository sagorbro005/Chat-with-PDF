import os
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

            # Initialize LLM
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=self.api_key,
                temperature=0.2
            )

            # Professional Prompt
            prompt = ChatPromptTemplate.from_template("""
            You are an expert AI analyst. Your goal is to provide a comprehensive, structured, and accurate answer based on the provided context.
            
            Instructions:
            1.  **Analyze** the provided context thoroughly.
            2.  **Structure** your answer:
                -   **Introduction**: Briefly state what the context says about the topic.
                -   **Key Details**: Use bullet points to list important facts, figures, or arguments found in the text.
                -   **Conclusion**: Summarize the findings.
            3.  **Tone**: Professional, objective, and detailed.
            4.  **Constraints**:
                -   Answer ONLY based on the Context.
                -   If the answer is missing, say: "I cannot find specific information about [topic] in the provided document."
            
            Context:
            {context}
            
            Question:
            {input}
            """)

            # Build Chains
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Execute
            response = retrieval_chain.invoke({"input": question})
            return response, None

        except Exception as e:
            return None, str(e)
