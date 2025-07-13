from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
llm = ChatOllama(model="gemma3:27b", temperature=0)

documents = DirectoryLoader("data/").load()
texts = RecursiveCharacterTextSplitter().split_documents(documents)
vector_store = FAISS.from_documents(texts, embeddings)

# 1. Basic RAG with LCEL (LangChain Expression Language).
print(f"\n{'=' * 50}")
print("🧱 1. BASIC RAG WITH LCEL SYNTAX")
print("=" * 50)

rag_prompt = ChatPromptTemplate.from_template("""
    Answer the question based on the context provided.
    If you cannot find the answer in the context, say so clearly.

    Context: {context}

    Question: {question}

    Answer: """)

rag_chain = (
    {"context": vector_store.as_retriever(), "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()
)

question = "What companies has Evgeni worked for?"
print(f"❓ Question: {question}")
response = rag_chain.invoke(question)
print(f"💬 Answer: {response.strip()}")

# 2. Sequential Chain Composition.
print(f"\n{'=' * 50}")
print("🔄 2. SEQUENTIAL CHAIN COMPOSITION")
print("=" * 50)

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
    """Extract key information from question"""
    result = (extract_prompt | llm | StrOutputParser()).invoke(inputs)
    return {**inputs, "key_info": result}


def search_step(inputs):
    """Search documents using vector retrieval"""
    docs = vector_store.similarity_search(inputs["question"], k=3)
    context = "\n".join([doc.page_content for doc in docs])

    result = (search_prompt | llm | StrOutputParser()).invoke({**inputs, "context": context})
    return {**inputs, "search_results": result}


def synthesis_step(inputs):
    """Synthesize final answer"""
    result = (synthesis_prompt | llm | StrOutputParser()).invoke(inputs)
    return {**inputs, "final_answer": result}


# Create sequential chain using LCEL.
sequential_chain = RunnablePassthrough() | extract_step | search_step | synthesis_step

print(f"❓ Question: {question}")
print("🔄 Running Sequential Chain...")
result = sequential_chain.invoke({"question": question})
print(f"💬 Answer: {result['final_answer']}")

# 3. Simple Smart Routing.
print(f"\n{'=' * 50}")
print("🎯 3. SIMPLE ROUTING")
print("=" * 50)

routing_prompt = PromptTemplate(
    template="""Analyze this question and determine the best approach. You must respond with EXACTLY one word: either "document" or "general".

    Question: {question}

    - Choose "document" if the question asks about:
    * Specific people (names, backgrounds, work history)
    * Companies someone worked for
    * Skills, experience, or qualifications of individuals
    * Any personal or professional information about someone
    
    - Choose "general" if the question asks about:
    * Math problems (like 5 + 3)
    * Geography (like capital cities)
    * General knowledge facts
    * Science concepts

    Answer with one word only: """,
    input_variables=["question"],
)

# Create routing chain.
routing_chain = routing_prompt | llm | StrOutputParser()

# Create specialized chains with proper document access.
document_prompt = PromptTemplate(
    template="""You are a document expert. Use the provided context to answer the question.

    Context from documents:
    {context}

    Question: {input}

    Answer based on the context: """,
    input_variables=["input", "context"],
)

general_prompt = PromptTemplate(
    template="You are a general knowledge assistant. Answer this question: {input}",
    input_variables=["input"],
)


def document_chain_with_retrieval(question):
    """Document chain that actually retrieves context"""
    docs = vector_store.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return (document_prompt | llm | StrOutputParser()).invoke({"input": question, "context": context})


general_chain = general_prompt | llm | StrOutputParser()


def route_and_answer(question: str):
    """General routing function using LLM"""
    route_decision = routing_chain.invoke({"question": question}).strip().lower()

    if "document" in route_decision:
        route = "document"
        response = document_chain_with_retrieval(question)
    else:
        route = "general"
        response = general_chain.invoke({"input": question})

    return route, response


test_questions = ["What companies has Evgeni worked for?", "What is the capital of France?", "What is 5 + 3?"]

for q in test_questions:
    print(f"\n❓ Question: {q}")
    route, response = route_and_answer(q)
    print(f"🎯 LLM routing decision: {route} chain")
    print(f"💬 Answer: {response.strip()}")

# 4. Memory-Enabled Conversation Chain.
print(f"\n{'=' * 50}")
print("🧠 4. CONVERSATION CHAIN WITH MEMORY")
print("=" * 50)

# Memory approach using simple state management.
conversation_history = []

# Conversation prompt with memory and document access.
conversation_prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant with access to document information. Use both the conversation history and retrieved documents to provide contextual responses.

    Retrieved context:
    {context}

    Previous conversation:
    {history}

    Current question: {input}

    Answer using the retrieved context and conversation history: """)

# Memory-enabled chain.
memory_chain = conversation_prompt | llm | StrOutputParser()


def memory_chain_with_retrieval(question, history):
    """Memory chain that uses document retrieval"""
    route_decision = routing_chain.invoke({"question": question}).strip().lower()

    if "document" in route_decision:
        docs = vector_store.similarity_search(question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
    else:
        context = "No specific document context needed for this question."

    return memory_chain.invoke({"input": question, "history": history, "context": context})


# Simulate conversation.
conversation_questions = [
    "What companies has Evgeni worked for?",
    "Which of these companies is the most recent?",
    "What skills does he have for that role?",
]

for q in conversation_questions:
    print(f"\n❓ Question: {q}")
    history_text = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in conversation_history])
    response = memory_chain_with_retrieval(q, history_text)
    print(f"💬 Answer: {response.strip()}")
    conversation_history.append({"question": q, "answer": response})
