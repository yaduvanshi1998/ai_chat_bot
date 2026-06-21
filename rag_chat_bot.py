# provides answers only through data which is used to train AI.

import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

OPENAI_API_KEY="............................................."

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
        # st.write(chunks)

        # 5. Embedd the generated chunks.

        embedding = OpenAIEmbeddings(
            model = "text-embedding-3-small",
            openai_api_key = OPENAI_API_KEY,
        )

        # 6. Store embeddings in vector db (kind of db).

        vector_store = FAISS.from_texts(chunks, embedding)

        # 7. Get user inputs

        user_input = st.text_input("Enter you questions here!")

        # 8. Generate Answers
        # Get question > embedding > similarity search > ranked results (to LLM) > show response --- this process is achieved by using chain.

        # Similarity search happens in retriever
        retriever = vector_store.as_retriever(
            search_type="mmr", # serach technique
            search_kwargs={"k":4} # to return 4 closest matches
        )

        # Format the  retriever

        def format_docs(docs):
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Define the LLM and prompts.

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3, # contrls randomness of answers
            max_tokens=1000,
            openai_api_key = OPENAI_API_KEY
        )

        # Provide prompts.

        prompt = ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a helpful assistant answering questions about a PDF document. In\n"
                    "Guidelines: \n"
                    "1. Provide complete, well-explained answers usin the context below. In"
                    "2. Include relevant details, numbers, and explanations to give a thorough response. \n"
                    "3. If the context mentions related information, include it to give fuller picture. \n"
                    "4. Only use information from the provided context - do not use outside knowledge. \n"
                    "5. Summarize long information, ideally in bullets where needed\n"
                    "5. If the information is not in the context, say so politely. In\n"
                    "Context: \n{context) "
                ), 
                ("human", "{question}")
                ])

        chain = (
            {"context": retriever | format_docs, "questions": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Invoke the chain is user input is present and write the answer on streamlit UI.

        if user_input:
            response = chain.invoke(user_input)
            st.write(response)
        

except Exception as e:
    print(e)