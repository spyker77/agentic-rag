# AI Agent & RAG Playground

A playground repository exploring different frameworks and approaches for building intelligent agentic systems with Retrieval Augmented Generation (RAG) capabilities.

## Overview

This repository demonstrates various implementations of agentic AI systems that leverage Retrieval Augmented Generation (RAG) for enhanced knowledge access. It serves as a learning playground and reference for AI engineers interested in exploring the landscape of agent-based architectures, multi-agent collaboration, and RAG integration.

The code is intentionally experimental in nature, showcasing different implementation patterns rather than providing a single production-ready solution.

## Frameworks Explored

- **LangChain**: The most mature and comprehensive ecosystem for building LLM applications. Its core strength is LangChain Expression Language (LCEL), which allows for composing chains from modular components, making it ideal for complex, multi-step workflows.
- **LangGraph**: An extension of LangChain for building stateful, agentic systems. It represents workflows as graphs, enabling cycles and conditional routing, which is perfect for creating complex agent behaviors that require state persistence and error recovery.
- **LlamaIndex**: A data-centric framework specializing in document indexing and retrieval for RAG. It offers advanced and highly customizable query engines, making it the go-to choice for document-heavy applications that need sophisticated retrieval strategies.
- **SmolAgents**: A minimalist and lightweight framework designed for creating simple, understandable agents. It excels at custom tool creation and real-time code execution, making it ideal for data analysis and visualization tasks where simplicity is key.
- **AutoGen**: A framework from Microsoft for building applications with multiple, conversing agents. It excels at code generation and execution in sandboxed environments (like Docker) and facilitates complex workflows through automated agent chats and human feedback loops.
- **CrewAI**: A framework designed for orchestrating role-playing, autonomous AI agents. It focuses on creating collaborative agent teams where each agent has a specific role, tools, and tasks, enabling them to work together to solve complex problems.
- **Haystack**: A production-ready, modular framework for building sophisticated search systems. It provides a component-based architecture for creating pipelines that can include hybrid search, re-ranking, and other advanced retrieval techniques, making it an industry standard for enterprise RAG.

## Key Components

### Agent Architectures

- **Single-Agent Systems**: Tool-enabled agents with document retrieval capabilities.
- **Multi-Agent Collaboration**: Specialized agents with role-based coordination (CrewAI).
- **Graph-Based Workflows**: Stateful agent processes with conditional routing and error recovery (LangGraph).
- **Code-Executing Agents**: Docker-containerized agents for safe code execution and testing (AutoGen).

### RAG Implementations

- **Basic RAG**: Implemented with LangChain LCEL syntax and LlamaIndex query engines.
- **Advanced RAG**: Production-grade system with HNSW vector indexing (misc/rag.py).
- **Two-stage Retrieval**: Hybrid search with semantic + keyword retrieval and re-ranking (Haystack).
- **Document Processing**: Chunking strategies, embedding generation, and vector store management.

### Advanced Techniques

- **`misc/dvts.py`**: Diverse Verifier Tree Search using Monte Carlo methods for complex reasoning tasks with local LLM generation and remote verification.
- **`misc/optimizations.py`**: High-performance LLM serving with request batching, two-level caching (Redis + semantic FAISS), and FastAPI endpoints.
- **`misc/perplexity.py`**: Real-time search system with dynamic index management and automatic cleanup after queries.

### Resume Agent

A production-style, modular LangGraph implementation designed to demonstrate a structured approach for building robust agents. Unlike the single-file demos, it separates concerns into distinct modules:

- **Graph Workflow**: Multi-node processing with conditional routing and state management.
- **RAG Integration**: Document retrieval with context-aware question answering.
- **Error Handling**: Robust error recovery with retry mechanisms and fallback strategies.
- **Tool Integration**: Custom tools for document search and information extraction.
- **Conversation Memory**: Persistent conversation history and context tracking.

### Evaluating the RAG System

To ensure the RAG system is both accurate and reliable, this project includes a comprehensive evaluation script based on the [Ragas](https://github.com/explodinggradients/ragas) framework. This script automates the process of testing the RAG chain, providing quantitative metrics to guide development and tuning.

The evaluation script performs the following steps:

1. Loads a document.
2. Splits the document into chunks.
3. Generates a synthetic test set of question/ground-truth answer pairs from the document chunks.
4. Executes the RAG chain against each test question to get an answer and the retrieved context.
5. Compares the generated answers and context against the ground truth using a suite of metrics.

#### How to Run the Evaluation

To run the evaluation, execute the following command from the root of the project:

```bash
python -m misc.evaluation
```

#### Interpreting the Metrics

The evaluation produces a report with the following key metrics, each scored from 0 to 1 (higher is better):

- **`context_precision`**: Measures the signal-to-noise ratio of the retrieved context. A high score means the retrieved context is highly relevant to the question.
  - *Low Score Indicates*: The retriever is pulling in irrelevant information, which can confuse the LLM. Try tuning the chunking strategy or improving the embedding model.
- **`context_recall`**: Measures whether all necessary information to answer the question was retrieved. This is a critical metric.
  - *Low Score Indicates*: The retriever is failing to find all the relevant context. The LLM cannot answer correctly if it doesn't have the information. This is often the first metric to fix.
- **`faithfulness`**: Measures how factually consistent the answer is with the retrieved context. It helps identify hallucinations.
  - *Low Score Indicates*: The LLM is making things up or using its internal knowledge instead of the provided context. Try improving the prompt to be more restrictive or use a different model.
- **`answer_relevancy`**: Measures how relevant the answer is to the *question*. An answer can be faithful to the context but not actually answer the user's query.
  - *Low Score Indicates*: The answer is off-topic. This can be due to poor context precision or a prompt that doesn't properly guide the LLM.
- **`answer_correctness`**: The overall score that measures the accuracy of the answer against the ground truth.
  - *Low Score Indicates*: The system is failing. Use the other four metrics to diagnose whether the problem lies in **retrieval** (`context_recall`) or **generation** (`faithfulness`).

## Getting Started

### Prerequisites

- Python 3.12+
- uv (for dependency management)
- Local LLM server (Ollama recommended)
- Docker (for AutoGen code execution demos)

### Installation

```bash
# Clone the repository
git clone https://github.com/spyker77/agentic-rag.git
cd agentic-rag

# Install dependencies
uv sync

# Set up environment variables (optional)
# Create .env file with API keys for external services:
# OPENAI_API_KEY=your_openai_key
# OPENROUTER_API_KEY=your_openrouter_key
```

### Running Demos

Each demo showcases different framework capabilities:

```bash
# LangChain: Chain composition and routing
python -m demos.langchain

# LangGraph: Stateful agent workflows
python -m demos.langgraph

# LlamaIndex: Document indexing and agents
python -m demos.llamaindex_agent
python -m demos.llamaindex_rag

# SmolAgents: Lightweight tool framework
python -m demos.smolagents

# AutoGen: Code execution agents
python -m demos.autogen

# CrewAI: Multi-agent collaboration
python -m demos.crewai

# Haystack: Two-stage retrieval
python -m demos.haystack

# Resume Agent: Structured, production-style agent
python -m src.resume_agent.main
```

## Project Structure

```bash
.
├── data/                 # Sample data including resume
├── demos/                # Framework comparison demonstrations
│   ├── langchain.py      # Chain composition and routing with LCEL
│   ├── langgraph.py      # Graph-based agent workflows
│   ├── llamaindex_*.py   # Document indexing and agents
│   ├── crewai.py         # Multi-agent collaboration
│   ├── smolagents.py     # Lightweight tool framework
│   ├── autogen.py        # Code execution agents
│   └── haystack.py       # Two-stage retrieval system
├── misc/                 # Advanced techniques and optimizations
│   ├── dvts.py           # Diverse Verifier Tree Search
│   ├── evaluation.py     # RAG evaluation with Ragas
│   ├── optimizations.py  # High-performance LLM serving
│   ├── perplexity.py     # Real-time search system
│   └── rag.py            # Production-grade RAG with HNSW
├── src/                  # Structured agent implementations
│   └── resume_agent/     # Production-style, modular agent implementation
│       ├── agents.py     # Agent definitions and workflows
│       ├── chains.py     # RAG and processing chains
│       ├── tools.py      # Custom tools and utilities
│       └── main.py       # Entrypoint for the agent application
├── pyproject.toml        # Dependencies and configuration
└── README.md             # This file
```

## Disclaimer

This repository is intended as an experimental playground for learning and exploration. The code is not optimized for production use and may contain experimental patterns and approaches.

## License

[MIT](LICENSE.md)
