from typing import Annotated

from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent


class AgentState(MessagesState):
    remaining_steps: Annotated[int, lambda current, _: current - 1]


def create_agent_workflow(llm, tools):
    agent = create_react_agent(llm, tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", lambda state: "agent" if state["remaining_steps"] > 0 else END)

    return workflow.compile()
