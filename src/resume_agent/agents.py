import operator
from enum import StrEnum
from functools import partial
from typing import Annotated, Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent


class ProcessingStatus(StrEnum):
    RUNNING = "RUNNING"
    NEED_CONTEXT = "NEED_CONTEXT"
    GENERAL_KNOWLEDGE = "GENERAL_KNOWLEDGE"
    ERROR = "ERROR"
    COMPLETE = "COMPLETE"


class AgentState(MessagesState):
    """Agent state with sophisticated tracking capabilities."""

    remaining_steps: Annotated[int, lambda current, _: max(0, current - 1)]
    processing_status: ProcessingStatus
    error_count: int
    routing_decision: str
    conversation_history: list[str]
    loop_step: Annotated[int, operator.add]
    last_error: str


def create_agent_workflow(llm, tools):
    """Create a basic ReAct agent workflow."""
    agent = create_react_agent(llm, tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", lambda state: "agent" if state["remaining_steps"] > 0 else END)

    return workflow.compile()


def _classify_query(state: AgentState, routing_chain):
    """Classify the query and determine routing."""
    try:
        question = state["messages"][-1].content

        # Use improved routing (embedding-based or keyword-based).
        if callable(routing_chain):
            # New embedding/keyword-based routing.
            routing_decision = routing_chain(question)
        else:
            # Fallback to LLM-based routing.
            routing_decision = routing_chain.invoke({"question": question}).strip().lower()

        # Normalize routing decision to handle both new and old formats.
        routing_decision_str = str(routing_decision).lower()

        if "document" in routing_decision_str:
            status = ProcessingStatus.NEED_CONTEXT
        elif "general" in routing_decision_str:
            status = ProcessingStatus.GENERAL_KNOWLEDGE
        else:
            status = ProcessingStatus.RUNNING

        return {
            "processing_status": status,
            "routing_decision": routing_decision,
            "loop_step": 1,
        }
    except Exception as e:
        return {
            "processing_status": ProcessingStatus.ERROR,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": str(e),
            "loop_step": 1,
        }


def _process_with_agent(state: AgentState, llm, tools):
    """Process query using ReAct agent with appropriate tools."""
    try:
        # Create filtered tools based on routing decision.
        filtered_tools = tools
        if state.get("routing_decision") == "document":
            filtered_tools = [t for t in tools if "document" in t.name or "resume" in t.name]
        elif state.get("routing_decision") == "general":
            filtered_tools = [t for t in tools if "general" in t.name]

        agent = create_react_agent(llm, filtered_tools)

        # Add conversation history for context.
        enhanced_messages = state["messages"]
        if state.get("conversation_history"):
            history_context = "\n".join(state["conversation_history"][-3:])  # last 3 exchanges
            enhanced_messages = enhanced_messages + [
                AIMessage(content=f"Previous conversation context: {history_context}")
            ]

        response = agent.invoke({"messages": enhanced_messages, "remaining_steps": state.get("remaining_steps", 5)})

        # Update conversation history.
        question = state["messages"][-1].content
        answer = response["messages"][-1].content
        updated_history = state.get("conversation_history", []) + [f"Q: {question}", f"A: {answer}"]

        return {
            "messages": response["messages"],
            "processing_status": ProcessingStatus.COMPLETE,
            "conversation_history": updated_history[-10:],  # keep last 10 entries - 5 Q&A pairs
            "loop_step": 1,
        }
    except Exception as e:
        return {
            "processing_status": ProcessingStatus.ERROR,
            "error_count": state.get("error_count", 0) + 1,
            "last_error": str(e),
            "loop_step": 1,
        }


def _handle_error(state: AgentState):
    """Handle error states with retry logic."""
    error_count = state.get("error_count", 0)

    if error_count >= 3:
        return {
            "processing_status": ProcessingStatus.COMPLETE,
            "messages": [
                AIMessage(
                    content=f"Maximum retry attempts reached. Last error: {state.get('last_error', 'Unknown error')}. Please try again later."
                )
            ],
            "loop_step": 1,
        }

    # Reset for retry.
    return {"processing_status": ProcessingStatus.RUNNING, "loop_step": 1}


def _route_by_status(state: AgentState) -> Literal["classify", "agent", "error", "end"]:
    """Route to next node based on processing status."""
    status = state.get("processing_status", ProcessingStatus.RUNNING)

    match status:
        case ProcessingStatus.ERROR:
            return "error"
        case ProcessingStatus.NEED_CONTEXT | ProcessingStatus.GENERAL_KNOWLEDGE | ProcessingStatus.RUNNING:
            return "agent"
        case ProcessingStatus.COMPLETE:
            return "end"
        case _:
            return "classify"


def create_enhanced_workflow(llm, tools, routing_chain):
    """Create workflow with routing, error handling, and memory."""
    workflow = StateGraph(AgentState)

    # Partial-apply the dependencies to the node functions.
    classify_query_node = partial(_classify_query, routing_chain=routing_chain)
    process_with_agent_node = partial(_process_with_agent, llm=llm, tools=tools)

    workflow.add_node("classify", classify_query_node)
    workflow.add_node("agent", process_with_agent_node)
    workflow.add_node("error", _handle_error)

    workflow.set_entry_point("classify")

    workflow.add_conditional_edges("classify", _route_by_status, {"agent": "agent", "error": "error", "end": END})
    workflow.add_conditional_edges("agent", _route_by_status, {"agent": "agent", "error": "error", "end": END})
    workflow.add_conditional_edges("error", _route_by_status, {"classify": "classify", "agent": "agent", "end": END})

    return workflow.compile()
