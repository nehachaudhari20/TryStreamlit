import os
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough

os.environ["GOOGLE_API_KEY"] = "AIzaSyChWDkjxDRyfjdG3H2Q2Wa5SgxzeR3Zgio"

st.title("Try Streamlit with langchain")
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()

    #split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    #embed and create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant. Use the following context to answer the question.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}"
    )

    #chain with LCEL : langchain expression language
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )


    query = st.text_input("Ask a question related to the pdf:")

    if query:
        response = chain.invoke(query)
        st.write("### Answer")
        st.write(response.content)
