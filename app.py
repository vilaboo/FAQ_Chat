from langchain.document_loaders import PyPDFLoader
import tempfile
import streamlit as st
from Base import creation_FAQ_chain,creation_of_vectorDB_in_local

def pdf_loader(tmp_file_path):
    loader = PyPDFLoader(tmp_file_path)
    return loader

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
                    
                    load = pdf_loader(tmp_file_path)
                    creation_of_vectorDB_in_local(load)
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
        ans = creation_FAQ_chain()
        result = ans(query)
        a = result["result"]
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("assistant"):
            st.markdown(a)
            st.session_state.messages.append({"role": "assistant", "content": a})
       
if __name__ == '__main__':
    main()
