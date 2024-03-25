# Standard Library Imports
import json
import os
import random
import requests
import string
import sys
import uuid
import hashlib
from typing import List

# Third-party Imports
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from dotenv import dotenv_values

# Project-specific Imports
#from langchain.callbacks import get_openai_callback
from langchain_community.callbacks import get_openai_callback
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.prompts import PromptTemplate
# from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from langchain.schema import HumanMessage
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.azure_cosmos_db import (
    AzureCosmosDBVectorSearch,
    CosmosDBSimilarityType,
)
from langchain_community.vectorstores.azuresearch import AzureSearch
#from langchain_openai import AzureChatOpenAI
from langchain_openai import  AzureOpenAIEmbeddings
from langchain_community.chat_models import AzureChatOpenAI

from tempfile import NamedTemporaryFile
import streamlit as st
from openai import OpenAI
import PyPDF2
import os

from utils import fetch_previous_history, get_or_generate_session_id, initialize_and_process, save_to_history, save_uploaded_file

import logging
import warnings
from datetime import datetime

# Create a log folder if it doesn't exist
log_folder = "logs"
os.makedirs(log_folder, exist_ok=True)

# Configure the logging settings
log_filename = os.path.join(log_folder, f"log_{datetime.now().strftime('%Y-%m-%d')}.txt")
logging.basicConfig(
    level=logging.INFO,  # Set the desired log level
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),  # Add this if you also want to log to the console
    ],
)


# Streamlit app layout
def main():
    if 'last_processed_file' not in st.session_state:
        st.session_state.last_processed_file = None
    
    if 'qa_chain_instance' not in st.session_state:
        st.session_state.qa_chain_instance = None

    if 'execution_count' not in st.session_state:        
        st.session_state.execution_count = 0
        
    st.title("AI Powered Hiring Assistant")
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file:
        
        # Initialize conversation history as an empty list in session state
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Read the uploaded PDF file
        document_id = uploaded_file.name
        user_id = "ranga"
        # Get or generate the session ID
        session_id = get_or_generate_session_id()
        
        # Check if the uploaded file is different from the last processed file
        if uploaded_file.name != st.session_state.last_processed_file:
            save_uploaded_file(uploaded_file)
            
            # Save the uploaded PDF to a temporary file
            with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())

            # Get the file path of the temporary file
            temp_file_path = temp_file.name

            _, qa_chain_instance, _ = initialize_and_process(
                    temp_file_path, document_id, user_id, session_id
            )
            
            st.session_state.qa_chain_instance = qa_chain_instance
            st.session_state.execution_count += 1
            print(f"execution count: {st.session_state.execution_count}")
            # Close the temporary file
            temp_file.close()

            st.session_state.last_processed_file = uploaded_file.name
            
        # Remove the temporary file after processing
        #st.file_uploader.cleanup()
        # User question input
        user_question = st.text_input("Enter your question:")

        # Fetch previous conversation history
        history = fetch_previous_history()
        #import pdb; pdb.set_trace()
        # Constructing the conversation history string from the list of dictionaries
        history_str = "\n".join([f"Human: {question}\nBot: {response}" for question, response in history])

        # Ask question to OpenAI when the question is submitted
        if st.button("Ask"):
            if not user_question:
                st.warning("Please enter a question.")
            else:
                # Ask the question to OpenAI with conversation history in the prompt
                try:
                    result = st.session_state.qa_chain_instance({"query": user_question, "chat_history": history})
                    response = result['result']
                    
                    # Display OpenAI's response
                    st.subheader("OpenAI's Response:")
                    st.text_area(label="", value=response, height=100)  # Increased height for OpenAI response
                    
                    # Display previous conversation history with a scrollable text area
                    
                    history = fetch_previous_history()
                    if history:
                        st.subheader("Previous Conversation History:")
                        history_str = "\n\n".join([f"User: {question}\nBot: {response}" for question, response in history])
                        st.text_area(label="", value=history_str, height=200)
                               
                    # Save the current conversation entry to history
                    save_to_history(user_question, response)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
