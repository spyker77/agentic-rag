from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .agents import create_agent_workflow, create_enhanced_workflow
from .chains import create_chain_suite, create_enhanced_rag_chain, create_rag_chain
from .config import EMBEDDINGS_MODEL, LLM_MODEL
from .tools import create_enhanced_toolset, create_resume_tool


def create_app():
    """Create the original simple application."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    documents = DirectoryLoader("data/").load()
    texts = RecursiveCharacterTextSplitter().split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)

    rag_chain = create_rag_chain(llm, vector_store)
    resume_tool = create_resume_tool(rag_chain)

    return create_agent_workflow(llm, [resume_tool])


def create_enhanced_app():
    """Create the enhanced application with sophisticated capabilities."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    documents = DirectoryLoader("data/").load()
    texts = RecursiveCharacterTextSplitter().split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)

    rag_chain = create_enhanced_rag_chain(llm, vector_store)

    tools, routing_chain, _ = create_enhanced_toolset(
        llm,
        rag_chain,
        vector_store=vector_store,
        documents=documents,
        routing_approach="simple",
    )

    return create_enhanced_workflow(llm, tools, routing_chain)


def create_demo_app():
    """Create demo application showcasing all capabilities."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    documents = DirectoryLoader("data/").load()
    texts = RecursiveCharacterTextSplitter().split_documents(documents)
    vector_store = FAISS.from_documents(texts, embeddings)

    chains = create_chain_suite(llm, vector_store)

    tools, routing_chain, memory_chain = create_enhanced_toolset(
        llm,
        chains["enhanced_rag"],
        vector_store=vector_store,
        documents=documents,
        routing_approach="vector_store",
    )

    workflow = create_enhanced_workflow(llm, tools, routing_chain)

    return {
        "workflow": workflow,
        "chains": chains,
        "tools": tools,
        "routing_chain": routing_chain,
        "memory_chain": memory_chain,
        "llm": llm,
        "vector_store": vector_store,
    }


__all__ = [
    "create_app",
    "create_enhanced_app",
    "create_demo_app",
    "EMBEDDINGS_MODEL",
    "LLM_MODEL",
]
