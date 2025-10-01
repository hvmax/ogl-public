import os
import torch
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from FlagEmbedding import FlagReranker

# --- Page Configuration ---
st.set_page_config(
    page_title="Document Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Chat with Your Documents")
st.write("This chatbot uses a local knowledge base and a reranker to provide high-quality answers. Powered by Groq for speed.")

# --- Configuration ---
@st.cache_resource
def get_config():
    """Returns all static configuration."""
    return {
        "embedding_model_name": "BAAI/bge-base-en-v1.5",
        "reranker_id": 'BAAI/bge-reranker-large',
        "groq_llm_id": "openai/gpt-oss-120b",
        "data_path": './articlesv3/',
        "chroma_db_path": "./chroma_db_bge"
    }

# --- Caching and Loading Components ---
@st.cache_resource
def load_retriever(_config):
    """Loads and caches the retriever component."""
    if not os.path.exists(_config["chroma_db_path"]):
        st.info("One-time setup: Creating new vector store... This may take a minute.")
        if not os.path.exists(_config["data_path"]) or not os.listdir(_config["data_path"]):
            st.error(f"The directory '{_config['data_path']}' is empty. Please add documents.")
            st.stop()
        
        # Determine device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        embeddings = HuggingFaceEmbeddings(
            model_name=_config["embedding_model_name"],
            model_kwargs={'device': device}
        )
        
        loader = DirectoryLoader(_config["data_path"], glob="**/*.txt", show_progress=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=_config["chroma_db_path"]
        )
        st.success("Vector store created successfully!")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embeddings = HuggingFaceEmbeddings(
            model_name=_config["embedding_model_name"],
            model_kwargs={'device': device}
        )
        vector_store = Chroma(
            persist_directory=_config["chroma_db_path"],
            embedding_function=embeddings
        )

    return vector_store.as_retriever(search_kwargs={'k': 10})

@st.cache_resource
def load_reranker(_config):
    """Loads and caches the reranker model."""
    return FlagReranker(_config["reranker_id"], use_fp16=True)

@st.cache_resource
def load_llm(_config):
    """Loads and caches the LLM."""
    try:
        # Use st.secrets for the API key
        groq_api_key = st.secrets["GROQ_API_KEY"]
        return ChatGroq(
            temperature=0.1,
            model_name=_config["groq_llm_id"],
            api_key=groq_api_key
        )
    except (KeyError, FileNotFoundError):
        st.error("GROQ_API_KEY not found in Streamlit secrets. Please add it.")
        st.stop()

# --- Main Application Logic ---
config = get_config()
retriever = load_retriever(config)
reranker = load_reranker(config)
llm = load_llm(config)

# --- Chat History Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input and Chat Logic ---
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # 1. Retrieve
            retrieved_docs = retriever.get_relevant_documents(prompt)

            # 2. Rerank
            reranker_input = [[prompt, doc.page_content] for doc in retrieved_docs]
            scores = reranker.compute_score(reranker_input)
            scored_docs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
            top_docs = [doc for score, doc in scored_docs[:3]]
            
            # 3. Build Context and Prompt
            context = "\n\n".join([doc.page_content for doc in top_docs])
            prompt_template = """
            Use the context to answer the question. If you don't know, say you don't know.
            Context: {context}
            Question: {question}
            Helpful Answer:
            """
            final_prompt = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            ).format(context=context, question=prompt)

            # 4. Generate
            response = llm.invoke(final_prompt)
            response_content = response.content
            st.markdown(response_content)
            
    st.session_state.messages.append({"role": "assistant", "content": response_content})