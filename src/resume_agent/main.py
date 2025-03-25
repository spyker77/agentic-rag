from langchain_core.messages import HumanMessage

from src.resume_agent import create_app


def main():
    chain = create_app()

    questions = ["What companies Evgeni Sautin has worked for?"]
    for question in questions:
        response = chain.invoke({"messages": [HumanMessage(content=question)], "remaining_steps": 10})
        print(f"\nQuestion: {question}")
        print(f"Answer: {response['messages'][-1].content}\n")


if __name__ == "__main__":
    main()
