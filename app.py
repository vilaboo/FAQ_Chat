import pdfplumber
from bs4 import BeautifulSoup
import requests
import tempfile
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

import os 
from dotenv import load_dotenv
load_dotenv()

db_file_path = 'FAISS_Index'
embeddings = HuggingFaceEmbeddings()

def pdf_loader(tmp_file_path):
    with pdfplumber.open(tmp_file_path) as pdf_file:
        page_contents = []
        for page in pdf_file.pages:
            page_text = page.extract_text()
            soup = BeautifulSoup(page_text, 'html.parser')
            text = soup.get_text()
            page_number = page.page_number
            page_contents.append({"page_content": text, "metadata": {"page_number": page_number}})
    return page_contents

def creation_of_vectorDB_in_local(page_contents):
    texts = [page["page_content"] for page in page_contents]
    db = FAISS.from_texts(texts, embeddings)
    db.save_local(db_file_path)

def creation_FAQ_chain(page_contents, user_question):
    db = FAISS.load_local(db_file_path, embeddings)
    retriever = db.as_retriever(score_threshold=0.7)
    
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
    
    prompt_temp = """
    Given the following context and a question, generate an answer based on the content of the uploaded PDF file.
    CONTEXT:{pdf_content}
    QUESTION:{user_question}
    """
    
    PROMPT = PromptTemplate(template=prompt_temp, input_variables=["pdf_content", "user_question"])
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                        retriever=retriever, 
                                        input_key="query", 
                                        return_source_documents=False,
                                        chain_type_kwargs={"prompt": PROMPT})
    
    # Execute the chain to get the answer
    result = chain({"pdf_content": page_contents, "user_question": user_question})
    return result

def main():
    st.set_page_config(page_title="FAQ Chatbot", layout="wide")
    st.title("FAQ ChatBot with your PDF file")
    
    page_contents = None
    
    with st.sidebar:
        st.title("Settings")
        st.markdown('---')
        st.subheader('Upload Your PDF File')
        doc = st.file_uploader("Upload your PDF file and Click Process", 'pdf')

        if st.button("Process"):
            with st.spinner("Processing"):
                if doc is not None:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(doc.read())  # Use doc.read() instead of doc.getvalue()
                        tmp_file_path = tmp_file.name
                        st.success(f'File {doc.name} is successfully saved!')

                    page_contents = pdf_loader(tmp_file_path)
                    creation_of_vectorDB_in_local(page_contents)
                    st.success("Process Done")
                else:
                    st.error("Please Upload Your File!")
        
    if page_contents is None:
        st.error("Please upload a PDF file and click Process.")
        return
    
    query = st.text_input("Ask the Question")
    if st.button("Submit") and query:
        ans = creation_FAQ_chain(page_contents, query)
        a = ans["result"]
        st.markdown(f"**User Question:** {query}")
        st.markdown(f"**Assistant Answer:** {a}")

if __name__ == '__main__':
    main()
