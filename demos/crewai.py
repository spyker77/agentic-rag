import os

from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool

EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


# NOTE: This demo is not working with ollama due to the following error:
# https://github.com/crewAIInc/crewAI/issues/2873#issuecomment-2899272854
#
# For configuration, see:
# https://docs.crewai.com/en/concepts/llms#open-router

llm_config = {
    "model": "openrouter/openai/gpt-4.1-mini",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key": os.environ["OPENROUTER_API_KEY"],
    "temperature": 0.1,
}

llm = LLM(**llm_config)

pdf_tool = PDFSearchTool(
    pdf="data/resume.pdf",
    config=dict(
        llm=dict(provider="openai", config=llm_config),
        embedder=dict(provider="huggingface", config=dict(model=EMBEDDINGS_MODEL)),
    ),
)

research_agent = Agent(
    role="Research Specialist",
    goal="Search and analyze the content of the provided PDF document to answer questions.",
    backstory="Expert at finding and synthesizing information from documents.",
    tools=[pdf_tool],
    llm=llm,
)

knowledge_agent = Agent(
    role="Knowledge Expert",
    goal="Provide accurate answers to general knowledge questions.",
    backstory="Expert at answering general knowledge questions.",
    llm=llm,
)


def process_question(question: str):
    task = Task(
        description=f"Answer the following question: {question}",
        expected_output="A concise and accurate answer.",
    )

    crew = Crew(
        agents=[research_agent, knowledge_agent],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
    )

    return crew.kickoff()


if __name__ == "__main__":
    questions = [
        "What companies Evgeni Sautin has worked for?",
        "What is the capital of France?",
        "What is Evgeni's favorite color?",
        "What is 5 + 3?",
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        result = process_question(question)
        print(f"💬 Answer: {result}")
