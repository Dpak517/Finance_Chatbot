import streamlit as st
import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
#from torchvision.transforms import functional as tvF
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from bs4 import BeautifulSoup
#from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter

#load_dotenv()
st.header('**Finance Chatbot** 📈')
#st.header(':rainbow[**Finance Chatbot**]')


st.sidebar.title('Enter Groq API key')
groq_api_key=st.sidebar.text_input(
    'Enter Groq API key here',
    type="password",
    #placeholder="gsk_.."
)

if groq_api_key!='':
    if not groq_api_key.startswith('gsk_'):
        st.sidebar.warning("Enter valid Groq API key")
        if not groq_api_key:
            st.sidebar.warning("First enter Groq API key")


# if not groq_api_key:
#     st.sidebar.warning("First enter Groq API key")




st.sidebar.title('NEWS Article links')
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
processor=st.empty()
urls=[]
for i in range(3):
    url=st.sidebar.text_input(f'URL {i + 1}')
    if url.startswith('http://') or url.startswith('https://'):
        urls.append(url)


st.sidebar.markdown('Upload PDF')
uploaded_file=st.sidebar.file_uploader(label='Upload PDF',type='pdf')

if uploaded_file:
    save_path=uploaded_file.name
    with open(save_path,'wb') as f:
        f.write(uploaded_file.read())

    pdf_loader=PyPDFLoader(save_path)
    pdf_data=pdf_loader.load()
        #all_docs.extend(pdf_data)

    processor.text('data loading ....')
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_documents(pdf_data)
    embeddings = embeddings
    processor.text('embedding text ....')
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local('faiss_index')
    processor.text('')

processed_url=st.sidebar.button(' Start Processing ')
if processed_url:
    loader=WebBaseLoader(urls)
    url_data=loader.load()
    processor.text('data loading ....')
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.split_documents(url_data)
    embeddings = embeddings
    processor.text('embedding urls ....')
    vector_store = FAISS.from_documents(docs, embeddings)
    vector_store.save_local('faiss_index')
    processor.text('')

if not groq_api_key:
    st.sidebar.warning("First enter Groq API key")

query = st.text_input('Question: ')
if groq_api_key:
    model=ChatGroq(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        temperature=0.3)
    

prompt = PromptTemplate(
    template="""
Answer the question using only the provided context and elaborate the answer like proper sentence.
Instructions:
- Give detailed and well-structured answers.
- Explain concepts clearly.
- If information exists in the context, prioritize it.
- If the answer is not available in the context, clearly state that the information was not found in the provided documents.
- Do not make up facts.


Context:
{context}

Question:
{input}

Answer:
""",
    input_variables=["context", "input"]
)

if query:
    vector_store = FAISS.load_local(
                "faiss_index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5,"fetch_k":15})
    question_answer_chain = create_stuff_documents_chain(llm=model, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    result = rag_chain.invoke({'input':query})

    st.header('Answer', divider='rainbow')
    with st.spinner('Wait for answer',show_time=True):
        st.write(result['answer'])
        # ✅ Show sources
        st.subheader("Sources")
        sources = set()  # use set to avoid duplicates
        for doc in result['context']:
            source = doc.metadata.get('source', None)
            if source:
                sources.add(source)

        if sources:
            for src in sources:
                st.write(f"🔗 {src}")
        else:
            st.write("No sources found")
    #st.write(result['answer'])

    #st.write(result['answer'])

