from langchain.document_loaders import PyPDFLoader
import pdfplumber
from bs4 import BeautifulSoup
import requests
import tempfile
import streamlit as st
from Base import creation_FAQ_chain,creation_of_vectorDB_in_local

def pdf_loader(tmp_file_path):
    with pdfplumber.open(tmp_file_path) as pdf_file:
        page_contents = []
        for page in pdf_file.pages:
            page_text = page.extract_text()
            soup = BeautifulSoup(page_text, 'html.parser')
            text = soup.get_text()
            page_contents.append(text)
    return page_contents

def main():
    st.set_page_config(page_title="FAQ Chatbot", layout="wide")
    st.title("FAQ ChatBot with your PDF file")

    with st.sidebar:
        st.title("Settings")
        st.markdown('---')
        st.subheader('Upload Your PDF File')
        doc = st.file_uploader("Upload your PDF file and Click Process", 'pdf')

        if st.button("Process"):
            with st.spinner("Processing"):
                if doc is not None:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(doc.getvalue())
                        tmp_file_path = tmp_file.name
                        st.success(f'File {doc.name} is successfully saved!')

                    page_contents = pdf_loader(tmp_file_path)
                    creation_of_vectorDB_in_local(page_contents)
                    st.success("Process Done")
                else:
                    st.error("Please Upload Your File!")
        
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask the Question")
    if query:
        ans = creation_FAQ_chain(page_contents, query)
        result = ans(query)
        a = result["result"]
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            st.markdown(a)
            st.session_state.messages.append({"role": "assistant", "content": a})

if __name__ == '__main__':
    main()
