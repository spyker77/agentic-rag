import asyncio

from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor
from autogen_ext.models.ollama import OllamaChatCompletionClient

model_client = OllamaChatCompletionClient(model="llama3.1:8b")

developer = AssistantAgent(
    name="Developer",
    model_client=model_client,
    system_message="""You are a Python Developer. Write clean, working Python code.
    
    When given a task:
    1. Write complete, executable Python code
    2. Include all necessary imports
    3. Use proper error handling
    4. Write code that produces clear output
    
    Focus on creating working solutions.""",
)

docker_executor = DockerCommandLineCodeExecutor(image="python:3.12", work_dir="coding")

code_executor = CodeExecutorAgent(
    name="CodeExecutor",
    code_executor=docker_executor,
    model_client=model_client,
    system_message="""You execute code and provide feedback on the results.
    
    When executing code:
    1. Run the code safely in a container
    2. Report any errors clearly
    3. Show the actual output
    4. Suggest fixes if needed""",
)

code_reviewer = AssistantAgent(
    name="CodeReviewer",
    model_client=model_client,
    system_message="""You are a Senior Code Reviewer. Review actual code execution results.
    
    1. Analyze the code and its execution output
    2. Check if the results match expectations
    3. Identify any issues or improvements
    4. Validate the solution works correctly
    
    Base your feedback on real execution results.""",
)


async def demonstrate_real_code_execution():
    """Show how agents collaborate with actual code execution"""

    print("=" * 80)
    print("🚀 AutoGen Multi-Agent Team with REAL Code Execution")
    print("👥 Team: Developer, CodeExecutor, Code Reviewer")
    print("=" * 80)

    team = RoundRobinGroupChat(
        participants=[developer, code_executor, code_reviewer],
        max_turns=10,
    )

    task = """
    Create a Python script that:
    1. Generates a dataset of 100 random points
    2. Fits a linear regression model to the data
    3. Plots the data points and regression line
    4. Calculates and displays the R² score
    5. Saves the plot as 'regression_analysis.png'
    
    Make sure the code actually runs and produces real results!
    """

    print(f"📋 Task: {task}")
    print("\n🔄 Team Collaboration with Real Execution:")
    print("-" * 80)

    await docker_executor.start()

    try:
        await Console(team.run_stream(task=task))
    finally:
        await docker_executor.stop()


async def main():
    await demonstrate_real_code_execution()


if __name__ == "__main__":
    asyncio.run(main())
