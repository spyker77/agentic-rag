from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .agents import create_agent_workflow
from .chains import create_rag_chain
from .config import EMBEDDINGS_MODEL, LLM_MODEL
from .tools import create_resume_tool


def create_app():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    documents = DirectoryLoader("data/").load()
    texts = RecursiveCharacterTextSplitter().split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)

    rag_chain = create_rag_chain(llm, vector_store)
    resume_tool = create_resume_tool(rag_chain)

    return create_agent_workflow(llm, [resume_tool])
