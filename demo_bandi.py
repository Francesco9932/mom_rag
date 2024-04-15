import streamlit as st

from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SpacyTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import os
import json

# Use CUDA if available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_openai_api_key():
    with open('preset.json') as f:
        data = json.load(f)
    return data["OPENAI_API_KEY"]


def get_embedding_model():
    # Define the path to the pre-trained model you want to use
    modelPath = "nickprock/sentence-bert-base-italian-uncased"

    # Create a dictionary with model configuration options, specifying to use the CPU for computations
    # use cpu or gpu
    model_kwargs = {'device': device}

    # Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
    encode_kwargs = {'normalize_embeddings': False}

    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters
    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,     # Provide the pre-trained model's path
        model_kwargs=model_kwargs, # Pass the model configuration options
        encode_kwargs=encode_kwargs # Pass the encoding options
    )
    return embeddings


def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = PyMuPDFLoader(file_path=tmp_file_path)
        data = loader.load()
        print(data[:2])

        # Split documents into chunks
        text_splitter = SpacyTextSplitter(pipeline="it_core_news_sm")
        docs = text_splitter.split_documents(data)
        # Select embeddings
        embeddings = get_embedding_model()
        # Create a vectorstore from documents
        db = FAISS.from_documents(docs, embeddings)
        # Create retriever interface
        retriever = db.from_documents(docs, embeddings)
        # LLM model
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key, model_name="gpt-4")
        # Create prompt template
        prompt_template = """
        Sei un esperto di bandi e concorsi pubblici. Hai memorizzato tutti i bandi mai creati.
        Utilizza il contesto nella tua memoria per rispondere alle domande sul bando.
        Caratteristiche:
        - Tono: formale.
        - Non fare utilizzo di bullet points.
        Se hai poco contesto rispondi: "Non so aiutarti, scusami"
        Se conosci il contesto riformula la risposta invece
        ----------------
        Contesto: {context}

        Domanda: {question}

        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        # Create QA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever.as_retriever(search_kwargs={"k": 1}),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
        )

        return chain.run(query_text)

# Page Title
st.set_page_config(page_title='🦜🔗 QA Bandi Gara')
st.title('🦜🔗 QA Bandi Gara')

with st.sidebar:
    # st.image("./CONEXO_LOGO.jpg")
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")

# File upload
uploaded_file = st.file_uploader('Upload a PDF', type='pdf')
# Query text
query_text = st.text_input('Fai una domanda:', placeholder = "Qual' è l'oggetto della fornitura del bando autocad?", disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = load_openai_api_key()
    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)