from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough


def create_rag_chain(llm, vector_store):
    """Create the original RAG chain."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based on the context provided."),
            ("human", "Context: {context}\nQuestion: {input}"),
        ]
    )

    docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_store.as_retriever(), docs_chain)


def create_enhanced_rag_chain(llm, vector_store):
    """Create enhanced RAG chain with better context handling."""
    enhanced_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant that answers questions based on provided context.
                If you cannot find the answer in the context, say so clearly.
                Be concise but thorough in your responses.""",
            ),
            (
                "human",
                "Context: {context}\nQuestion: {input}",
            ),
        ]
    )
    docs_chain = create_stuff_documents_chain(llm, enhanced_prompt)
    return create_retrieval_chain(vector_store.as_retriever(search_kwargs={"k": 5}), docs_chain)


def create_direct_llm_chain(llm):
    """Create direct LLM chain for general knowledge questions."""
    direct_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer questions accurately and concisely."),
            ("human", "{input}"),
        ]
    )
    return direct_prompt | llm | StrOutputParser()


def create_routing_chain(llm):
    """Create intelligent routing chain to classify questions."""
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
        * Math problems (like 5 + 3)
        * Geography (like capital cities)
        * General knowledge facts
        * Science concepts
        * Common knowledge questions

        For the question "{question}", I choose: """,
        input_variables=["question"],
    )
    return routing_prompt | llm | StrOutputParser()


def create_memory_enhanced_chain(llm, vector_store):
    """Create memory-enhanced chain for conversation context."""
    memory_prompt = PromptTemplate(
        template="""Based on the conversation history and current question, provide a comprehensive answer.

        Conversation History:
        {history}

        Current Question: {question}

        If the question relates to previous conversation, use that context. Otherwise, answer the current question directly.

        Answer: """,
        input_variables=["history", "question"],
    )
    return memory_prompt | llm | StrOutputParser()


def create_sequential_chain(llm, vector_store):
    """Create sequential processing chain like the LangChain demo."""

    # Step 1: Extract key information.
    extract_prompt = PromptTemplate(
        input_variables=["question"],
        template="""
        Analyze this question and extract the key information needed:
        Question: {question}
        
        Key information to search for: """,
    )

    # Step 2: Search documents with actual retrieval.
    search_prompt = PromptTemplate(
        input_variables=["question", "key_info", "context"],
        template="""
        Original question: {question}
        Key information to find: {key_info}
        Retrieved context: {context}
        
        Based on the retrieved documents, provide relevant information: """,
    )

    # Step 3: Synthesize final answer.
    synthesis_prompt = PromptTemplate(
        input_variables=["question", "key_info", "search_results"],
        template="""
        Original question: {question}
        Key information: {key_info}
        Search results: {search_results}
        
        Provide a comprehensive, well-structured answer: """,
    )

    def extract_step(inputs):
        """Extract key information from question."""
        result = (extract_prompt | llm | StrOutputParser()).invoke(inputs)
        return {**inputs, "key_info": result}

    def search_step(inputs):
        """Search documents using vector retrieval."""
        docs = vector_store.similarity_search(inputs["question"], k=3)
        context = "\n".join([doc.page_content for doc in docs])

        result = (search_prompt | llm | StrOutputParser()).invoke({**inputs, "context": context})
        return {**inputs, "search_results": result}

    def synthesis_step(inputs):
        """Synthesize final answer."""
        result = (synthesis_prompt | llm | StrOutputParser()).invoke(inputs)
        return {**inputs, "final_answer": result}

    return RunnablePassthrough() | extract_step | search_step | synthesis_step


def create_chain_suite(llm, vector_store):
    """Create complete suite of chains for different use cases."""
    return {
        "rag": create_rag_chain(llm, vector_store),
        "enhanced_rag": create_enhanced_rag_chain(llm, vector_store),
        "direct_llm": create_direct_llm_chain(llm),
        "routing": create_routing_chain(llm),
        "memory": create_memory_enhanced_chain(llm, vector_store),
        "sequential": create_sequential_chain(llm, vector_store),
    }
