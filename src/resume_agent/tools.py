from langchain_core.tools import Tool


def create_resume_tool(rag_chain):
    return Tool(
        name="search_resume",
        description="Search through resume for specific information. Use this for finding details about work experience, skills, and background.",
        func=lambda q: rag_chain.invoke({"input": q})["answer"],
    )
