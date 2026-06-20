# provides answers only through data which is used to train AI.

import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

OPENAI_API_KEY="......................................."

# 1. Code to get streamlit UI to upload the PDF File.

st.header("'AI Chat Bot' -Chandan Yadav") # heading

with st.sidebar:
    st.title("Your Documents.")
    file = st.file_uploader("Upload your PDF File Here and start asking questions to Chat Bot!", type='pdf') # to upload a file and store datd of the file.

# 2. Code to extract data/texts from uploaded files.

try:
    if file is not None:
        # Extract text from file.
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        #st.write(text)

        # 3. Split texts into small small chunks, instead of passing all the data at once.

        split_texts = RecursiveCharacterTextSplitter(
            separators = ["\n\n", "\n", ". ", " ", ""],
            chunk_size = 1000,
            chunk_overlap = 200 # since we have chunk size as 1000, it might break and store incomplete sentence, 
            # so over to overcome on that we use overlap to add some of the previous characters,
            # so that sentence in chunck stays in connect.
        )

        # 4. write the chunks

        chunks = split_texts.split_text(text)
        st.write(chunks)

        # 5. Embedd the generated chunks.

        embedding = OpenAIEmbeddings(
            model = "text-embedding-3-small",
            openai_api_key = OPENAI_API_KEY,
        )

        # 6. Store embeddings in vector db (kind of db).

        vector_store = FAISS.from_texts(chunks, embedding)
        

except Exception as e:
    print(e)