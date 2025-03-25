# from crewai import LLM, Agent, Crew, Process, Task
# from crewai_tools import PDFSearchTool

# # Initialize tools and LLM
# pdf_tool = PDFSearchTool(
#     pdf="data/resume.pdf",
#     config=dict(
#         llm=dict(
#             provider="ollama",
#             config=dict(model="llama3.1:8b", temperature=0.1, base_url="http://127.0.0.1:11434"),
#         ),
#         embedder=dict(provider="huggingface", config=dict(model="intfloat/e5-large-v2")),
#     ),
# )

# llm = LLM(model="ollama/llama3.1:8b", temperature=0.1, base_url="http://127.0.0.1:11434")

# # Define manager agent for coordination
# manager_agent = Agent(
#     role="Task Manager",
#     goal="Coordinate question analysis and answer generation",
#     backstory="Expert at analyzing questions and coordinating responses",
#     llm=llm,
#     verbose=True,
# )

# # Define specialized agents
# research_agent = Agent(
#     role="Research Specialist",
#     goal="Search and analyze document content",
#     backstory="Expert at finding and synthesizing information from documents",
#     tools=[pdf_tool],
#     llm=llm,
#     verbose=True,
# )

# knowledge_agent = Agent(
#     role="Knowledge Expert",
#     goal="Provide accurate answers using general knowledge",
#     backstory="Expert at answering general knowledge questions",
#     llm=llm,
#     verbose=True,
# )


# def process_question(question: str):
#     tasks = [
#         Task(
#             description=(
#                 f"Analyze and route this question: {question}\n"
#                 "If the question requires document search, assign to Research Specialist.\n"
#                 "If it's general knowledge, assign to Knowledge Expert."
#             ),
#             agent=manager_agent,
#             expected_output="NEEDS_RESEARCH or GENERAL_KNOWLEDGE",
#         ),
#         Task(
#             description=f"Provide a comprehensive answer to: {question}",
#             agent=None,
#             expected_output="Detailed answer with all relevant information",
#         ),
#     ]

#     # Create crew with hierarchical process.
#     crew = Crew(
#         agents=[research_agent, knowledge_agent],
#         tasks=tasks,
#         process=Process.hierarchical,
#         manager_agent=manager_agent,
#         verbose=True,
#     )

#     return crew.kickoff({"question": question})


# # Test questions
# questions = [
#     "What companies Evgeni Sautin has worked for?",
#     "What is the capital of France?",
#     "What is Evgeni's favorite color?",
# ]

# for question in questions:
#     print(f"\nProcessing Question: {question}")
#     print("-" * 50)
#     result = process_question(question)
#     print(f"Answer: {result}")
#     print("-" * 50)


from crewai import LLM, Agent, Crew, Process, Task
from crewai_tools import PDFSearchTool

# Initialize tools
pdf_tool = PDFSearchTool(
    pdf="data/resume.pdf",
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(model="llama3.1:8b", temperature=0.1, base_url="http://127.0.0.1:11434"),
        ),
        embedder=dict(provider="huggingface", config=dict(model="intfloat/e5-large-v2")),
    ),
)

llm = LLM(model="ollama/llama3.1:8b", temperature=0.1, base_url="http://127.0.0.1:11434")

# Define agents with their capabilities
context_agent = Agent(
    role="Context Expert",
    goal="Find and provide information from documents",
    backstory="Expert at searching and synthesizing information from available documents",
    tools=[pdf_tool],
    llm=llm,
    verbose=True,
)

direct_agent = Agent(
    role="Direct Response Expert",
    goal="Provide accurate answers to general knowledge questions",
    backstory="Expert at answering questions that don't require additional context",
    llm=llm,
    verbose=True,
)


def process_question(question: str):
    task = Task(description=f"Answer this question: {question}", expected_output="Comprehensive answer to the question")

    crew = Crew(
        agents=[context_agent, direct_agent],
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=True,
    )

    return crew.kickoff()


questions = [
    "What companies Evgeni Sautin has worked for?",
    # "What is the capital of France?",
    # "What is Evgeni's favorite color?",
]

for question in questions:
    print(f"\nProcessing Question: {question}")
    print("-" * 50)
    result = process_question(question)
    print(f"Answer: {result}")
    print("-" * 50)
