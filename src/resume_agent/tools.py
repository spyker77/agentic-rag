import numpy as np
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool


def create_resume_tool(rag_chain):
    """Create the original resume search tool."""
    return Tool(
        name="search_resume",
        description="Search through resume for specific information. Use this for finding details about work experience, skills, and background.",
        func=lambda q: rag_chain.invoke({"input": q})["answer"],
    )


def create_document_search_tool(rag_chain):
    """Create specialized document search tool for resume-specific queries."""
    return Tool(
        name="document_search",
        description="Search through resume and documents for specific information about work experience, skills, education, and professional background. Use this for questions that need specific document context.",
        func=lambda q: rag_chain.invoke({"input": q})["answer"],
    )


def create_general_knowledge_tool(llm):
    """Create general knowledge tool for questions that don't need document context."""
    general_prompt = PromptTemplate(
        template="You are a helpful assistant with general knowledge. Answer this question concisely and accurately: {question}",
        input_variables=["question"],
    )

    general_chain = general_prompt | llm | StrOutputParser()

    return Tool(
        name="general_knowledge",
        description="Answer general knowledge questions like math problems, geography, science facts, and common knowledge. Use this for questions that don't require document context.",
        func=lambda q: general_chain.invoke({"question": q}),
    )


def create_simple_embedding_router(embeddings):
    """Create simple embedding router with minimal, clear examples."""

    category_examples = {
        "document": [
            "What companies has someone worked for?",
            "What are their skills?",
            "What is their work experience?",
            "What is their educational background?",
            "What projects have they worked on?",
            "Tell me about their career",
            "What is their professional background?",
        ],
        "general": [
            "What is 5 + 3?",
            "What is the capital of France?",
            "How does gravity work?",
            "What is machine learning?",
            "When was World War 2?",
            "What is the largest planet?",
            "How do you cook pasta?",
        ],
    }

    category_embeddings = {}
    for category, examples in category_examples.items():
        example_embeddings = embeddings.embed_documents(examples)
        category_embeddings[category] = np.mean(example_embeddings, axis=0)

    def classify_question(question: str) -> str:
        """Classify question using simple embedding similarity."""
        question_embedding = embeddings.embed_query(question)

        similarities = {}
        for category, category_embedding in category_embeddings.items():
            similarity = np.dot(question_embedding, category_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(category_embedding)
            )
            similarities[category] = similarity

        return max(similarities, key=lambda x: similarities[x])

    return classify_question


def create_vector_store_router(embeddings, vector_store, similarity_threshold=0.5):
    """Create router that uses vector store to determine if question needs document context."""

    fallback_router = create_simple_embedding_router(embeddings)

    def classify_with_vector_store(question: str) -> str:
        """Use vector store to determine if question relates to documents."""
        if not vector_store:
            return fallback_router(question)

        try:
            # Search for relevant documents.
            docs = vector_store.similarity_search(question, k=3)

            # If we find relevant documents, it's likely a document question.
            if docs and len(docs) > 0:
                # Simple heuristic: if we found documents, it's probably a document question.
                return "document"
            else:
                # No relevant documents found, probably general knowledge.
                return "general"
        except Exception:
            # If anything goes wrong, fall back to embedding classification.
            return fallback_router(question)

    return classify_with_vector_store


def create_embedding_based_router(embeddings):
    """Create embedding-based router - much faster and more reliable than LLM-based."""

    category_examples = {
        "document": [
            "What companies has John worked for?",
            "What are his skills?",
            "What is his work experience?",
            "Where did he go to school?",
            "What projects has he worked on?",
            "What technologies does he know?",
            "What is his background?",
            "Tell me about his career",
            "What jobs has he had?",
            "What is his education?",
        ],
        "general": [
            "What is 5 + 3?",
            "What is the capital of France?",
            "Who invented the telephone?",
            "What is Python programming language?",
            "How does gravity work?",
            "What is the largest planet?",
            "When was World War 2?",
            "What is machine learning?",
            "How to cook pasta?",
            "What is the speed of light?",
        ],
    }

    category_embeddings = {}
    for category, examples in category_examples.items():
        example_embeddings = embeddings.embed_documents(examples)
        category_embeddings[category] = np.mean(example_embeddings, axis=0)

    def classify_question(question: str) -> str:
        """Classify question using embedding similarity."""
        question_embedding = embeddings.embed_query(question)

        similarities = {}
        for category, category_embedding in category_embeddings.items():
            # Calculate cosine similarity.
            similarity = np.dot(question_embedding, category_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(category_embedding)
            )
            similarities[category] = similarity

        # Return category with highest similarity.
        return max(similarities, key=lambda x: similarities[x])

    return classify_question


def create_routing_chain(llm):
    """Create routing chain to classify questions."""
    routing_prompt = PromptTemplate(
        template="""Analyze this question and determine the best approach:

        Question: {question}

        - Choose "document" if the question asks about:
        * Specific people (names, backgrounds, work history)
        * Companies someone worked for
        * Skills, experience, or qualifications of individuals
        * Any personal or professional information about someone
        * Resume or CV content
        
        - Choose "general" if the question asks about:
        * Math problems (like 15 + 3)
        * Geography (like capital cities)
        * General knowledge facts
        * Science concepts
        * Common knowledge questions

        For the question "{question}", I choose: """,
        input_variables=["question"],
    )

    return routing_prompt | llm | StrOutputParser()


def create_memory_chain(llm):
    """Create memory chain for conversation context."""
    memory_prompt = PromptTemplate(
        template="""Based on the conversation history, provide context for the current question:

        Conversation History:
        {history}

        Current Question: {question}

        Relevant context from history: """,
        input_variables=["history", "question"],
    )

    return memory_prompt | llm | StrOutputParser()


def create_enhanced_toolset(llm, rag_chain, vector_store=None, documents=None, routing_approach="simple"):
    """Create the complete enhanced toolset with clean, maintainable routing."""
    tools = [
        create_document_search_tool(rag_chain),
        create_general_knowledge_tool(llm),
        create_resume_tool(rag_chain),
    ]

    from langchain_huggingface import HuggingFaceEmbeddings

    from .config import EMBEDDINGS_MODEL

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)

    if routing_approach == "simple":
        # Simple embedding approach - clean and maintainable.
        routing_chain = create_simple_embedding_router(embeddings)
    elif routing_approach == "vector_store":
        # Vector store approach - uses actual document similarity.
        routing_chain = create_vector_store_router(embeddings, vector_store)
    elif routing_approach == "basic":
        # Basic embedding approach (original).
        routing_chain = create_embedding_based_router(embeddings)
    else:
        # Fallback to LLM-based routing.
        routing_chain = create_routing_chain(llm)

    memory_chain = create_memory_chain(llm)

    return tools, routing_chain, memory_chain
