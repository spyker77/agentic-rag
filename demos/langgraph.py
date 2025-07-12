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

EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
llm = ChatOllama(model="llama3.1:8b", temperature=0)

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

documents = DirectoryLoader("data/").load()
texts = RecursiveCharacterTextSplitter().split_documents(documents)
vector_store = FAISS.from_documents(texts, embeddings)
retriever = vector_store.as_retriever()

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


workflow = StateGraph(GraphState)

workflow.add_node("classify", classify_query)
workflow.add_node("rag", process_with_rag)
workflow.add_node("llm", process_with_llm)
workflow.add_node("error", handle_error)

workflow.set_entry_point("classify")

workflow.add_conditional_edges("classify", route_by_status, {"rag": "rag", "llm": "llm", "error": "error", "end": END})
workflow.add_conditional_edges("rag", route_by_status, {"error": "error", "end": END})
workflow.add_conditional_edges("llm", route_by_status, {"error": "error", "end": END})
workflow.add_conditional_edges("error", route_by_status, {"classify": "classify", "end": END})

chain = workflow.compile()

questions = [
    "What companies Evgeni Sautin has worked for?",
    "What is the capital of France?",
    "What is Evgeni's favorite color?",
    "What is 5 + 3?",
]

for question in questions:
    print(f"\n❓ Question: {question}")
    response = chain.invoke(
        {
            "messages": [HumanMessage(content=question)],
            "status": ProcessingStatus.RUNNING,
            "error_count": 0,
            "loop_step": 0,
        }
    )
    print(f"💬 Answer: {response['messages'][-1].content}")
