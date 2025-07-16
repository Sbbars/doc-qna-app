
import streamlit as st
import os

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import pipeline

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="📄 Fast Document Q&A", layout="wide")
st.title("📄 AI-Powered Document Q&A")

st.markdown("""
Upload a document (PDF, DOCX, or TXT)
""")

# -------------------------------
# Load lightweight LLM (FLAN-T5)
# -------------------------------
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-small"
    )

llm = load_llm()

# -------------------------------
# File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload your document", type=["pdf", "docx", "txt"])
index_path = "index/faiss_index"

if uploaded_file:
    with st.spinner("Processing..."):
        os.makedirs("data", exist_ok=True)
        file_path = os.path.join("data", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load Document
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)

        docs = loader.load()
        st.info(f"✅ Loaded {len(docs)} document sections.")

        # Split & Embed
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embed_model)

        os.makedirs("index", exist_ok=True)
        vector_store.save_local(index_path)

        st.success(f"✅ Indexed {len(chunks)} chunks.")

# -------------------------------
# Q&A Section
# -------------------------------
st.header("💬 Ask a Question")

if os.path.exists(index_path):
    user_question = st.text_input("Type your question and press Enter:")
    
    if user_question:
        with st.spinner("Retrieving context..."):
            embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.load_local(index_path, embed_model, allow_dangerous_deserialization=True)
            top_chunks = vector_store.similarity_search(user_question, k=3)

        context = "\n".join([doc.page_content for doc in top_chunks])

        st.subheader("📚 Context")
        for i, doc in enumerate(top_chunks):
            with st.expander(f"Chunk {i+1}"):
                st.write(doc.page_content)

        # Generate Answer
        st.subheader("🤖 Answer")
        prompt = f"Answer the question based on the context.\n\nContext:\n{context}\n\nQuestion: {user_question}\nAnswer:"
        
        with st.spinner("Generating..."):
            output = llm(prompt, max_new_tokens=256)[0]["generated_text"]

        st.success("✅ Answer ready!")
        st.markdown(f"**Answer:** {output.strip()}")
else:
    st.warning("⚠️ Please upload a document to build the knowledge base.")
