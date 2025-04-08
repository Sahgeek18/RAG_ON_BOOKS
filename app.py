import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import tempfile

# --------------------
# Set your Gemini API Key
# --------------------
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
 # Replace with your actual key

# --------------------
# Streamlit App UI
# --------------------
st.set_page_config(page_title="📚 PDF Q&A with Gemini", layout="wide")
st.title("📄 Ask your PDF using Gemini 1.5 Flash")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

query = st.text_input("Ask a question about your PDF:", placeholder="e.g. What is the main concept in this document?")

# --------------------
# Process PDF and Answer Questions
# --------------------
if uploaded_file and query:

    with st.spinner("Processing PDF..."):

        # 1. Read PDF text
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""

        # 2. Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=250
        )
        chunks = splitter.create_documents([text])

        # 3. Embed with HuggingFace
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

        # 4. Create FAISS vectorstore (in memory)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # 5. Setup Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            max_output_tokens = 4092
        )


        # adding a prompt 
        prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        Use the following context to answer the question in great detail and depth.
        If the answer is not found in the context, respond with "I don't know."

        Context:
        {context}

        Question: {question}
        """
        )

        # 6. Setup RetrievalQA chain
        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs = {"prompt" : prompt_template},
            return_source_documents=True
        )

        # 7. Ask the question
        response = qa_chain(query)

        # 8. Display Answer
        st.subheader("📘 Answer")
        st.write(response["result"])

        # 9. Optional: Display Source Chunks
        with st.expander("📝 Source Document Chunks"):
            for i, doc in enumerate(response["source_documents"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)

