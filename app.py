import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os

# Update imports for LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from datetime import datetime


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text, model_name):
    if model_name == "Google AI":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks, model_name, api_key=None):
    if model_name == "Google AI":
        # Replace Gemini with HuggingFace all-MiniLM
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# creat conversational chain using langchain
def get_conversational_chain(model_name,vector_store=None,api_key=None):
    if model_name=="Google AI":
        prompt_template="""
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
         # Initialize the AI model, temperature: balanced creativity and consistency
        model=ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=api_key)
        # Create prompt object
        prompt=PromptTemplate(template=prompt_template,input_variable=["context", "question"])
        # Build and return the QA chain
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain



