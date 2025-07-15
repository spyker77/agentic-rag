from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import CodeAgent, LiteLLMModel, tool

EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
llm = LiteLLMModel(model_id="ollama_chat/llama3.3:70b", api_base="http://localhost:11434")

documents = DirectoryLoader("data/").load()
texts = RecursiveCharacterTextSplitter().split_documents(documents)
vector_store = FAISS.from_documents(texts, embeddings)


@tool
def vector_search_tool(query: str) -> str:
    """
    Search for relevant information in the document store.

    Args:
        query: Question to search for in the documents

    Returns:
        Relevant document snippets
    """
    docs = vector_store.similarity_search(query, k=3)
    return "\n".join(f"Document {i + 1}: {doc.page_content}" for i, doc in enumerate(docs))


@tool
def create_visualization(data_type: str, title: str, filename: str) -> str:
    """
    Create and save a data visualization.

    Args:
        data_type: Type of data to visualize ('random', 'sine', 'exponential')
        title: Title for the plot
        filename: Name of file to save (without extension)

    Returns:
        Status message about the visualization
    """
    try:
        plt.figure(figsize=(10, 6))

        if data_type == "random":
            x = np.linspace(0, 10, 100)
            y = np.random.normal(0, 1, 100).cumsum()
            plt.plot(x, y, "b-", linewidth=2)
            plt.ylabel("Cumulative Value")

        elif data_type == "sine":
            x = np.linspace(0, 4 * np.pi, 100)
            y = np.sin(x) * np.exp(-x / 10)
            plt.plot(x, y, "r-", linewidth=2)
            plt.ylabel("Amplitude")

        elif data_type == "exponential":
            x = np.linspace(0, 5, 100)
            y = np.exp(x / 2) * np.cos(2 * x)
            plt.plot(x, y, "g-", linewidth=2)
            plt.ylabel("Value")

        plt.xlabel("X")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = f"coding/{filename}.png"
        Path("coding").mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return f"✅ Visualization saved as {output_path}"

    except Exception as e:
        return f"❌ Error creating visualization: {str(e)}"


@tool
def analyze_data(dataset_name: str, operation: str) -> str:
    """
    Analyze numerical data and return statistics.

    Args:
        dataset_name: Name of dataset to create ('sales', 'temperatures', 'stock_prices')
        operation: Type of analysis ('summary', 'correlation', 'trends')

    Returns:
        Analysis results
    """
    try:
        # Create sample datasets.
        if dataset_name == "sales":
            data = {
                "month": range(1, 13),
                "sales": np.random.normal(50000, 10000, 12),
                "marketing_spend": np.random.normal(5000, 1000, 12),
            }
        elif dataset_name == "temperatures":
            data = {
                "day": range(1, 31),
                "temperature": np.random.normal(22, 5, 30),
                "humidity": np.random.normal(60, 15, 30),
            }
        elif dataset_name == "stock_prices":
            data = {
                "day": range(1, 21),
                "price": np.random.normal(100, 10, 20),
                "volume": np.random.normal(1000000, 200000, 20),
            }
        else:
            return f"❌ Unknown dataset: {dataset_name}"

        df = pd.DataFrame(data)

        if operation == "summary":
            stats = df.describe()
            return f"📊 Summary Statistics for {dataset_name}:\n{stats.to_string()}"

        elif operation == "correlation":
            corr = df.corr()
            return f"📈 Correlation Matrix for {dataset_name}:\n{corr.to_string()}"

        elif operation == "trends":
            # Simple trend analysis.
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            trends = {}
            for col in numeric_cols:
                if col != "day" and col != "month":
                    slope = np.polyfit(df.index, df[col], 1)[0]
                    trends[col] = "increasing" if slope > 0 else "decreasing"

            return f"📊 Trend Analysis for {dataset_name}:\n" + "\n".join(
                f"- {col}: {trend}" for col, trend in trends.items()
            )

        else:
            return f"❌ Unknown operation: {operation}"

    except Exception as e:
        return f"❌ Error analyzing data: {str(e)}"


agent = CodeAgent(
    tools=[vector_search_tool, create_visualization, analyze_data],
    model=llm,
    additional_authorized_imports=["matplotlib", "pandas", "numpy"],
)


tasks = [
    {
        "question": "What companies Evgeni Sautin has worked for?",
        "description": "Document search",
    },
    {
        "question": "Analyze the 'sales' dataset using summary statistics, then create a sine wave visualization titled 'Wave Analysis'.",
        "description": "Statistical analysis + mathematical visualization",
    },
    {
        "question": "Perform trend analysis on stock_prices data, then create an exponential visualization titled 'Market Dynamics'. Explain the results.",
        "description": "Multi-step analysis with explanations",
    },
]

for i, task in enumerate(tasks, 1):
    print(f"\n🔍 Task {i}: {task['description']}")
    print(f"❓ Question: {task['question']}")
    response = agent.run(task["question"])
    print(f"💬 Answer: {response}")
