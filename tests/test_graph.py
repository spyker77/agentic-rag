import operator
from enum import StrEnum
from typing import Annotated, Literal, TypedDict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph, add_messages

# Initialize models and load documents
embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# Create direct LLM chain for simple questions
direct_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. If you cannot answer the question with high confidence, respond with 'I NEED CONTEXT'.",
        ),
        ("human", "{input}"),
    ]
)
direct_chain = direct_prompt | llm

# Set up RAG components
documents = DirectoryLoader("data/").load()
texts = RecursiveCharacterTextSplitter().split_documents(documents)
vector_store = FAISS.from_documents(texts, embeddings)
retriever = vector_store.as_retriever()

# Create RAG chain
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer the question based on the context provided."),
        ("human", "Context: {context}\nQuestion: {input}"),
    ]
)
docs_chain = create_stuff_documents_chain(llm, rag_prompt)
rag_chain = create_retrieval_chain(retriever, docs_chain)


class ProcessingStatus(StrEnum):
    RUNNING = "RUNNING"
    NEED_CONTEXT = "NEED_CONTEXT"
    ERROR = "ERROR"
    COMPLETE = "COMPLETE"


class GraphState(TypedDict):
    """State of the graph."""

    messages: Annotated[list, add_messages]
    status: ProcessingStatus
    error_count: int
    loop_step: Annotated[int, operator.add]


def classify_query(state: GraphState):
    """Determine if query needs context."""
    try:
        question = state["messages"][-1].content
        response = direct_chain.invoke({"input": question})

        if "I NEED CONTEXT" in response.content:
            return {"status": ProcessingStatus.NEED_CONTEXT, "messages": state["messages"]}

        return {"status": ProcessingStatus.RUNNING, "messages": [AIMessage(content=response.content)]}
    except Exception as e:
        return {
            "status": ProcessingStatus.ERROR,
            "error_count": state.get("error_count", 0) + 1,
            "messages": [AIMessage(content=f"Error in classification: {str(e)}")],
        }


def process_with_rag(state: GraphState):
    """Process query using RAG."""
    try:
        question = state["messages"][-1].content
        response = rag_chain.invoke({"input": question})

        return {
            "status": ProcessingStatus.COMPLETE,
            "messages": [AIMessage(content=response["answer"])],
            "loop_step": 1,
        }
    except Exception as e:
        return {
            "status": ProcessingStatus.ERROR,
            "error_count": state.get("error_count", 0) + 1,
            "messages": [AIMessage(content=f"Error in RAG processing: {str(e)}")],
        }


def process_with_llm(state: GraphState):
    """Process query using direct LLM."""
    try:
        return {
            "status": ProcessingStatus.COMPLETE,
            "messages": state["messages"][-1:],  # use last message which contains LLM response
            "loop_step": 1,
        }
    except Exception as e:
        return {
            "status": ProcessingStatus.ERROR,
            "error_count": state.get("error_count", 0) + 1,
            "messages": [AIMessage(content=f"Error in LLM processing: {str(e)}")],
        }


def handle_error(state: GraphState):
    """Handle error states."""
    error_count = state.get("error_count", 0)
    if error_count >= 3:
        return {
            "status": ProcessingStatus.COMPLETE,
            "messages": [AIMessage(content="Maximum retry attempts reached. Please try again later.")],
            "loop_step": 1,
        }
    return {"status": ProcessingStatus.RUNNING, "messages": state["messages"]}


def route_by_status(state: GraphState) -> Literal["classify", "rag", "llm", "error", "end"]:
    """Route to next node based on status."""
    match state["status"]:
        case ProcessingStatus.ERROR:
            return "error"
        case ProcessingStatus.NEED_CONTEXT:
            return "rag"
        case ProcessingStatus.RUNNING:
            return "llm"
        case ProcessingStatus.COMPLETE:
            return "end"
        case _:
            return "classify"


# Create graph
workflow = StateGraph(GraphState)

# Add nodes
workflow.add_node("classify", classify_query)
workflow.add_node("rag", process_with_rag)
workflow.add_node("llm", process_with_llm)
workflow.add_node("error", handle_error)

# Set entry point
workflow.set_entry_point("classify")

# Add conditional edges
workflow.add_conditional_edges("classify", route_by_status, {"rag": "rag", "llm": "llm", "error": "error", "end": END})
workflow.add_conditional_edges("rag", route_by_status, {"error": "error", "end": END})
workflow.add_conditional_edges("llm", route_by_status, {"error": "error", "end": END})
workflow.add_conditional_edges("error", route_by_status, {"classify": "classify", "end": END})

# Compile graph
chain = workflow.compile()

# Test questions
questions = [
    "What companies Evgeni Sautin has worked for?",
    "What is the capital of France?",
    "What is Evgeni's favorite color?",
]

# Run chain
for question in questions:
    print(f"\nQuestion: {question}")
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "status": ProcessingStatus.RUNNING,
            "error_count": 0,
            "loop_step": 0,
        }
    )
    print(f"Answer: {response['messages'][-1].content}\n")


# import operator
# from enum import StrEnum
# from typing import Annotated, Literal, TypedDict

# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain.chains.retrieval import create_retrieval_chain
# from langchain_community.document_loaders import DirectoryLoader
# from langchain_community.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_core.messages import AIMessage, HumanMessage
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langgraph.graph import END, StateGraph, add_messages

# # Initialize models
# embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
# llm = ChatOllama(model="llama3.1:8b", temperature=0)

# # Load and process documents
# documents = DirectoryLoader("data/").load()
# texts = RecursiveCharacterTextSplitter().split_documents(documents)
# vector_store = FAISS.from_documents(texts, embeddings)
# retriever = vector_store.as_retriever()

# # Create direct LLM chain for simple questions
# direct_prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "You are a helpful assistant. If you cannot answer the question with high confidence, respond with 'I NEED CONTEXT'.",
#         ),
#         ("human", "{input}"),
#     ]
# )
# direct_chain = direct_prompt | llm

# # Create RAG chain
# rag_prompt = ChatPromptTemplate(
#     [
#         ("system", "Answer the question based on the context provided. Be specific and detailed."),
#         ("human", "Context: {context}\nQuestion: {input}"),
#     ]
# )
# docs_chain = create_stuff_documents_chain(llm, rag_prompt)
# rag_chain = create_retrieval_chain(retriever, docs_chain)

# # Create grading prompts
# relevance_prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "Grade if the retrieved documents are relevant to the question. Respond with 'RELEVANT' or 'IRRELEVANT'.",
#         ),
#         ("human", "Question: {question}\nDocuments: {documents}"),
#     ]
# )
# relevance_chain = relevance_prompt | llm

# reflection_prompt = ChatPromptTemplate(
#     [
#         (
#             "system",
#             "Check if the answer is fully supported by the documents and relevant to the question. Respond with 'SUPPORTED' or 'NEEDS_IMPROVEMENT'.",
#         ),
#         ("human", "Question: {question}\nDocuments: {documents}\nAnswer: {answer}"),
#     ]
# )
# reflection_chain = reflection_prompt | llm


# class ProcessingStatus(StrEnum):
#     RUNNING = "RUNNING"
#     NEED_CONTEXT = "NEED_CONTEXT"
#     NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
#     ERROR = "ERROR"
#     COMPLETE = "COMPLETE"


# class GraphState(TypedDict):
#     """Enhanced state tracking."""

#     messages: Annotated[list, add_messages]
#     status: ProcessingStatus
#     error_count: int
#     loop_step: Annotated[int, operator.add]
#     context: str | None
#     documents: list[Document] | None
#     iteration_count: Annotated[int, operator.add]


# def get_retrieval_strategy(question: str) -> dict:
#     """Choose appropriate retrieval strategy based on query type."""
#     strategies = {
#         "semantic": {"search_type": "similarity", "k": 3},
#         "mmr": {"search_type": "mmr", "k": 3, "fetch_k": 5},  # Maximal Marginal Relevance
#         "hybrid": {"search_type": "similarity_score_threshold", "score_threshold": 0.8},
#     }

#     # Simple heuristic - could be enhanced with more sophisticated classification.
#     if any(word in question.lower() for word in ["what", "who", "when"]):
#         return strategies["semantic"]
#     elif any(word in question.lower() for word in ["compare", "difference", "similar"]):
#         return strategies["mmr"]
#     return strategies["hybrid"]


# def classify_query(state: GraphState):
#     """Determine if query needs context."""
#     try:
#         question = state["messages"][-1].content
#         response = direct_chain.invoke({"input": question})

#         if "I NEED CONTEXT" in response.content:
#             strategy = get_retrieval_strategy(question)
#             retriever = vector_store.as_retriever(**strategy)
#             docs = retriever.invoke(question)

#             return {
#                 "status": ProcessingStatus.NEED_CONTEXT,
#                 "messages": state["messages"],
#                 "documents": docs,
#                 "context": "\n\n".join(doc.page_content for doc in docs),
#             }

#         return {"status": ProcessingStatus.RUNNING, "messages": [AIMessage(content=response.content)]}
#     except Exception as e:
#         return {
#             "status": ProcessingStatus.ERROR,
#             "error_count": state.get("error_count", 0) + 1,
#             "messages": [AIMessage(content=f"Error in classification: {str(e)}")],
#         }


# def process_with_rag(state: GraphState):
#     """Process query using RAG with retrieved documents."""
#     try:
#         question = state["messages"][-1].content
#         context = state.get("context", "")

#         docs_relevance = relevance_chain.invoke({"question": question, "documents": context})

#         if "IRRELEVANT" in docs_relevance.content:
#             return {
#                 "status": ProcessingStatus.NEEDS_IMPROVEMENT,
#                 "messages": [
#                     AIMessage(content="Retrieved documents are not relevant. Attempting to improve retrieval.")
#                 ],
#                 "iteration_count": state.get("iteration_count", 0) + 1,
#             }

#         response = rag_chain.invoke({"input": question, "context": context})

#         return {
#             "status": ProcessingStatus.COMPLETE,
#             "messages": [AIMessage(content=response["answer"])],
#             "loop_step": 1,
#         }
#     except Exception as e:
#         return {
#             "status": ProcessingStatus.ERROR,
#             "error_count": state.get("error_count", 0) + 1,
#             "messages": [AIMessage(content=f"Error in RAG processing: {str(e)}")],
#         }


# def process_with_llm(state: GraphState):
#     """Process query using direct LLM."""
#     try:
#         return {"status": ProcessingStatus.COMPLETE, "messages": state["messages"][-1:], "loop_step": 1}
#     except Exception as e:
#         return {
#             "status": ProcessingStatus.ERROR,
#             "error_count": state.get("error_count", 0) + 1,
#             "messages": [AIMessage(content=f"Error in LLM processing: {str(e)}")],
#         }


# def evaluate_response(state: GraphState):
#     """Evaluate response quality and determine if refinement needed."""
#     try:
#         question = state["messages"][0].content
#         answer = state["messages"][-1].content
#         context = state.get("context", "")

#         reflection = reflection_chain.invoke({"question": question, "documents": context, "answer": answer})

#         if "NEEDS_IMPROVEMENT" in reflection.content and state.get("iteration_count", 0) < 3:
#             return {
#                 "status": ProcessingStatus.NEEDS_IMPROVEMENT,
#                 "messages": state["messages"],
#                 "iteration_count": state.get("iteration_count", 0) + 1,
#             }

#         # Always set loop_step to ensure termination.
#         return {"status": ProcessingStatus.COMPLETE, "messages": state["messages"], "loop_step": 1}
#     except Exception as e:
#         return {
#             "status": ProcessingStatus.ERROR,
#             "error_count": state.get("error_count", 0) + 1,
#             "messages": [AIMessage(content=f"Error in evaluation: {str(e)}")],
#         }


# def handle_error(state: GraphState):
#     """Handle error states."""
#     error_count = state.get("error_count", 0)
#     if error_count >= 3:
#         return {
#             "status": ProcessingStatus.COMPLETE,
#             "messages": [AIMessage(content="Maximum retry attempts reached. Please try again later.")],
#             "loop_step": 1,
#         }
#     return {"status": ProcessingStatus.RUNNING, "messages": state["messages"]}


# def route_by_status(state: GraphState) -> Literal["classify", "rag", "llm", "evaluate", "error", "end"]:
#     """Enhanced routing with evaluation and refinement paths."""
#     match state["status"]:
#         case ProcessingStatus.ERROR:
#             return "error"
#         case ProcessingStatus.NEED_CONTEXT:
#             return "rag"
#         case ProcessingStatus.NEEDS_IMPROVEMENT:
#             if state.get("iteration_count", 0) >= 3:
#                 return "end"
#             return "rag"
#         case ProcessingStatus.RUNNING:
#             return "llm"
#         case ProcessingStatus.COMPLETE:
#             # Check loop_step to ensure termination.
#             if state.get("loop_step", 0) >= 1:
#                 return "end"
#             return "evaluate"
#         case _:
#             return "classify"


# # Create graph
# workflow = StateGraph(GraphState)

# # Add nodes
# workflow.add_node("classify", classify_query)
# workflow.add_node("rag", process_with_rag)
# workflow.add_node("llm", process_with_llm)
# workflow.add_node("evaluate", evaluate_response)
# workflow.add_node("error", handle_error)

# # Set entry point
# workflow.set_entry_point("classify")

# # Add conditional edges
# workflow.add_conditional_edges(
#     "classify",
#     route_by_status,
#     {"rag": "rag", "llm": "llm", "error": "error", "evaluate": "evaluate", "end": END},
# )
# workflow.add_conditional_edges(
#     "rag",
#     route_by_status,
#     {"rag": "rag", "llm": "llm", "error": "error", "evaluate": "evaluate", "end": END},
# )
# workflow.add_conditional_edges(
#     "llm",
#     route_by_status,
#     {"rag": "rag", "error": "error", "evaluate": "evaluate", "end": END},
# )
# workflow.add_conditional_edges(
#     "evaluate",
#     route_by_status,
#     {"rag": "rag", "error": "error", "evaluate": "evaluate", "end": END},
# )
# workflow.add_conditional_edges(
#     "error",
#     route_by_status,
#     {"classify": "classify", "end": END},
# )

# # Compile graph
# chain = workflow.compile()

# # Test questions with enhanced logging
# questions = [
#     "What companies Evgeni Sautin has worked for?",
#     "What is the capital of France?",
#     "What is Evgeni's favorite color?",
# ]

# # Run chain with detailed logging
# for question in questions:
#     print(f"\nProcessing Question: {question}")
#     print("-" * 50)

#     response = chain.invoke(
#         {
#             "messages": [HumanMessage(content=question)],
#             "status": ProcessingStatus.RUNNING,
#             "error_count": 0,
#             "loop_step": 0,
#             "iteration_count": 0,
#             "context": None,
#             "documents": None,
#         }
#     )

#     print(f"Final Answer: {response['messages'][-1].content}")
#     print(f"Status: {response.get('status', 'Unknown')}")
#     print(f"Iterations: {response.get('iteration_count', 0)}")
#     print("-" * 50)
