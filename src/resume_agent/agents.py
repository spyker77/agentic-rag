from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph, add_messages
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import create_react_agent


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    remaining_steps: Annotated[int, RemainingSteps]


def create_agent_workflow(llm, tools):
    agent = create_react_agent(llm, tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        lambda state: state["remaining_steps"] > 1 and "Final Answer:" not in state["messages"][-1].content,
        {True: "agent", False: END},
    )

    return workflow.compile()
