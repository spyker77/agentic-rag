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
