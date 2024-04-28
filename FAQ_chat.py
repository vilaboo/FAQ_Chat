import streamlit as st
import os
import tempfile
import google.generativeai as genai
import time
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import DeepLake
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PIL import Image

genai.configure(api_key="AIzaSyC2ngziHY2mFK7_epi4U-gzNqq0BQ1pW4s")

# Set up the model
generation_config = {
  "temperature": 0,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
}

llm = genai.GenerativeModel(model_name="gemini-pro", generation_config=generation_config)

def loading_file(uploaded_file,text_splitter,embeddings):
    file_name = uploaded_file.name
    db = DeepLake(
        dataset_path="./chatbot/deeplake", embedding_function=embeddings, overwrite=True
    )
    with st.spinner("Loading {} ...".format(file_name)):
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name,file_name)
        with open(temp_filepath,'wb') as f:
            f.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        doc = loader.load()  
        texts = text_splitter.split_documents(doc)
        db.add_documents(texts)
    return db

def display_conversation(messages):
    for convo in messages:
        with st.chat_message("user"):
            st.write(convo[0])
        with st.chat_message("assistant"):
            st.write(convo[1])

def submit():
    st.session_state.question = st.session_state.widget
    st.session_state.widget = ''

image = Image.open('FAQ.png')

def main():
    st.set_page_config(page_title="FAQ Chat", layout="wide")    
    faq_logo, title, faq_logo2 = st.columns(3, gap="large")
 
    with faq_logo:
        st.image(image, width=228, use_column_width=False)
    with title:
        st.title("FAQ Chat")
    with faq_logo2:
        st.image(image, width=228, use_column_width=False)

    uploaded_file = st.file_uploader(label='Upload a PDF Document')

    col1, col2 = st.columns(2)

    with col1:
        st.header("Ask a Question")
        question = st.text_input("Type your question here:", key='query')
        if question:
            st.write("Processing your question...")

    with col2:
        st.header("Answer")
        if question:
            try:
                with st.spinner("Fetching response..."):
                    # Assuming 'db' and other necessary components are properly initialized and available
                    convo_qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=db.as_retriever())
                    result = conva_qa({'query': question}, return_only_outputs=True)
                    st.write(result['result'])
            except Exception as e:
                st.error("Failed to process the question. Error: {}".format(e))

if __name__ == "__main__":
    main()
