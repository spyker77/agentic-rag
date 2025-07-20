# AI Agent & RAG Playground

A playground repository exploring different frameworks and approaches for building intelligent agentic systems with Retrieval Augmented Generation (RAG) capabilities.

## Overview

This repository demonstrates various implementations of agentic AI systems that leverage Retrieval Augmented Generation (RAG) for enhanced knowledge access. It serves as a learning playground and reference for AI engineers interested in exploring the landscape of agent-based architectures, multi-agent collaboration, and RAG integration.

The code is intentionally experimental in nature, showcasing different implementation patterns rather than providing a single production-ready solution.

## Frameworks Explored

- **LangChain**: A comprehensive ecosystem for building LLM applications with a focus on composable chains (LCEL).
- **LangGraph**: An extension of LangChain for building stateful, agentic systems using graph-based workflows.
- **LlamaIndex**: A data-centric framework specializing in advanced document indexing and retrieval for RAG.
- **SmolAgents**: A minimalist framework for creating simple, understandable agents with a focus on custom tools and code execution.
- **AutoGen**: A Microsoft framework for building multi-agent systems that excel at code generation in sandboxed environments.
- **CrewAI**: A framework for orchestrating role-playing, autonomous AI agents in collaborative teams.
- **Haystack**: A modular, production-ready framework for building sophisticated search and retrieval pipelines.
- **Pydantic AI**: A type-safe agent framework with a focus on structured, validated responses and production reliability.

## Key Components

### Agent Architectures

- **Single-Agent Systems**: Basic tool-enabled agents with RAG.
- **Multi-Agent Collaboration**: Role-based agent teams (e.g., CrewAI).
- **Graph-Based Workflows**: Stateful, cyclic agent processes (e.g., LangGraph).
- **Code-Executing Agents**: Sandboxed code execution for safety (e.g., AutoGen).

### RAG Implementations

- **Basic RAG**: Simple retrieval with LangChain and LlamaIndex.
- **Advanced RAG**: Production-grade retrieval with HNSW vector indexing.
- **Two-stage Retrieval**: Hybrid search (semantic + keyword) with re-ranking.
- **Document Processing**: Chunking, embeddings, and vector store management.

### Advanced Techniques

- **`misc/dvts.py`**: Diverse Verifier Tree Search for complex reasoning.
- **`misc/optimizations.py`**: High-performance LLM serving with batching and caching.
- **`misc/perplexity.py`**: Real-time search with dynamic index management.

### Resume Agent

A production-style, modular agent built with LangGraph to demonstrate a structured, robust implementation. It features a multi-node graph workflow, RAG integration, error handling, custom tools, and conversation memory.

### Evaluating the RAG System

This project includes evaluation scripts for [Ragas](https://github.com/explodinggradients/ragas) and [DeepEval](https://github.com/confident-ai/deepeval) to measure the performance of the RAG system. These frameworks provide quantitative metrics to guide development and tuning.

The scripts automatically generate a synthetic test set from a document, run the RAG chain, and evaluate the results based on metrics like context precision/recall, faithfulness, and answer relevancy.

#### How to Run Evaluations

To run the evaluation with `ragas`, execute the following command from the root of the project:

```bash
python -m misc.evaluations.ragas
```

To run the evaluation with `deepeval`, run the following command:

```bash
deepeval test run misc/evaluations/test_deepeval.py
```

For detailed explanations of the metrics, please refer to the official documentation for [Ragas](https://docs.ragas.io/en/latest/concepts/metrics/index.html) and [DeepEval](https://docs.confident-ai.com/docs/metrics-introduction).

## Getting Started

### Prerequisites

- Python 3.12+
- uv (for dependency management)
- Local LLM server (Ollama recommended)

  ```bash
  ollama pull gemma3:27b
  ollama pull devstral:24b
  ollama pull llama3.3:70b
  ```

- `OPENROUTER_API_KEY` (or `OPENAI_API_KEY` with changes)
- Docker (for AutoGen code execution demos)

### Installation

```bash
# Clone the repository
git clone https://github.com/spyker77/agentic-rag.git
cd agentic-rag

# Install dependencies
uv sync
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

# Pydantic AI: Type-safe agents with structured responses
python -m demos.pydantic_ai

# Resume Agent: Structured, production-style agent
python -m src.resume_agent.main
```

## Project Structure

```bash
.
├── data/                       # Sample data including resume
├── demos/                      # Framework comparison demonstrations
│   ├── langchain.py            # Chain composition and routing with LCEL
│   ├── langgraph.py            # Graph-based agent workflows
│   ├── llamaindex_*.py         # Document indexing and agents
│   ├── crewai.py               # Multi-agent collaboration
│   ├── smolagents.py           # Lightweight tool framework
│   ├── autogen.py              # Code execution agents
│   ├── haystack.py             # Two-stage retrieval system
│   └── pydantic_ai.py          # Type-safe agents with structured responses
├── misc/                       # Advanced techniques and optimizations
│   └── evaluations/            # Evaluation frameworks
│       ├── ragas.py            # RAG evaluation with Ragas
│       └── test_deepeval.py    # Pytest-based evaluation with DeepEval
│   ├── dvts.py                 # Diverse Verifier Tree Search
│   ├── optimizations.py        # High-performance LLM serving
│   ├── perplexity.py           # Real-time search system
│   └── rag.py                  # Production-grade RAG with HNSW
├── src/                        # Structured agent implementations
│   └── resume_agent/           # Production-style, modular agent implementation
│       ├── agents.py           # Agent definitions and workflows
│       ├── chains.py           # RAG and processing chains
│       ├── tools.py            # Custom tools and utilities
│       └── main.py             # Entrypoint for the agent application
├── pyproject.toml              # Dependencies and configuration
└── README.md                   # This file
```

## Disclaimer

This repository is intended as an experimental playground for learning and exploration. The code is not optimized for production use and may contain experimental patterns and approaches.

## License

[MIT](LICENSE.md)
