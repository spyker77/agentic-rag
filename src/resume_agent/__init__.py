from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .agents import create_agent_workflow, create_enhanced_workflow
from .chains import create_chain_suite, create_enhanced_rag_chain, create_rag_chain
from .config import EMBEDDINGS, LLM
from .tools import create_enhanced_toolset, create_resume_tool


def create_app():
    """Create the original simple application."""
    documents = DirectoryLoader("data/").load()
    texts = RecursiveCharacterTextSplitter().split_documents(documents)
    vector_store = FAISS.from_documents(texts, EMBEDDINGS)

    rag_chain = create_rag_chain(LLM, vector_store)
    resume_tool = create_resume_tool(rag_chain)

    return create_agent_workflow(LLM, [resume_tool])


def create_enhanced_app():
    """Create the enhanced application with sophisticated capabilities."""
    documents = DirectoryLoader("data/").load()
    texts = RecursiveCharacterTextSplitter().split_documents(documents)
    vector_store = FAISS.from_documents(texts, EMBEDDINGS)

    rag_chain = create_enhanced_rag_chain(LLM, vector_store)

    tools, routing_chain, _ = create_enhanced_toolset(
        LLM,
        EMBEDDINGS,
        rag_chain,
        vector_store=vector_store,
        documents=documents,
        routing_approach="simple",
    )

    return create_enhanced_workflow(LLM, tools, routing_chain)


def create_demo_app():
    """Create demo application showcasing all capabilities."""
    documents = DirectoryLoader("data/").load()
    texts = RecursiveCharacterTextSplitter().split_documents(documents)
    vector_store = FAISS.from_documents(texts, EMBEDDINGS)

    chains = create_chain_suite(LLM, vector_store)

    tools, routing_chain, memory_chain = create_enhanced_toolset(
        LLM,
        EMBEDDINGS,
        chains["enhanced_rag"],
        vector_store=vector_store,
        documents=documents,
        routing_approach="vector_store",
    )

    workflow = create_enhanced_workflow(LLM, tools, routing_chain)

    return {
        "workflow": workflow,
        "chains": chains,
        "tools": tools,
        "routing_chain": routing_chain,
        "memory_chain": memory_chain,
        "llm": LLM,
        "vector_store": vector_store,
    }


__all__ = [
    "create_app",
    "create_enhanced_app",
    "create_demo_app",
    "EMBEDDINGS",
    "LLM",
]
