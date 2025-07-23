import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
from langchain.embeddings import OllamaEmbeddings

import os
import hashlib

load_dotenv()

# Generate a unique hash based on uploaded PDF filenames
def get_file_hash(pdf_docs):
    combined_names = ''.join(sorted([pdf.name for pdf in pdf_docs]))
    return hashlib.md5(combined_names.encode()).hexdigest()


# PDF Reader
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


# Divide the text into chunks
def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks


# Convert chunks into vector store
def get_vector_store(text_chunks, index_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(index_path)


# Load QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in 
    the context, just say "answer is not available in the context". Do not make up an answer.

    Context:\n{context}

    Question: {question}

    Answer:
    """
    model = OllamaLLM(model="llama3:8b")  # Adjust to your actual Ollama model name
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


# Handle user input and question-answering
def user_input(user_question, index_path):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    if not os.path.exists(index_path):
        st.warning("Please upload and process the PDFs first.")
        return

    new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response['output_text'])


# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat With Multiple PDFs")
    st.header("📄 Chat with Multiple PDFs using Ollama")

    if "index_path" not in st.session_state:
        st.session_state.index_path = None

    user_question = st.text_input("💬 Ask a question from the PDF files:")

    if user_question and st.session_state.index_path:
        user_input(user_question, st.session_state.index_path)

    with st.sidebar:
        st.title("📂 Menu")
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

        if st.button("📥 Submit & Process"):
            if pdf_docs:
                with st.spinner("🔄 Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunk(raw_text)
                    file_hash = get_file_hash(pdf_docs)
                    index_path = f"faiss_index_{file_hash}"
                    get_vector_store(text_chunks, index_path)
                    st.session_state.index_path = index_path
                    st.success("✅ PDFs processed and indexed.")
            else:
                st.warning("Please upload at least one PDF.")

if __name__ == "__main__":
    main()
