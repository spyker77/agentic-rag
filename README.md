# Agentic RAG Playground

A playground repository exploring different frameworks and approaches for building Retrieval Augmented Generation (RAG) systems with agentic capabilities.

## Overview

This repository demonstrates various implementations of RAG systems using different frameworks and approaches. It serves as a learning playground and reference for AI engineers interested in exploring the landscape of RAG and agent-based architectures.

The code is intentionally experimental in nature, showcasing different implementation patterns rather than providing a single production-ready solution.

## Frameworks Explored

- **LangChain**: Implementation of retrieval chains and document processing
- **LangGraph**: Graph-based agent workflows with state management
- **LlamaIndex**: Vector store indexing and query engines
- **CrewAI**: Multi-agent systems with specialized roles and coordination
- **SmolAgents**: Lightweight agent framework with custom tools

## Key Components

### RAG Implementations

- Basic RAG with LangChain and LlamaIndex
- Document chunking and embedding strategies
- Vector store creation and similarity search

### Agent Architectures

- Single-agent RAG with tool use
- Multi-agent systems with coordination
- Graph-based agent workflows

### Advanced Techniques

- `misc/dvts.py`: Diverse Verifier Tree Search for complex reasoning
- `misc/rag.py`: Production-grade RAG with HNSW vector search
- `misc/perplexity.py`: Real-time search with dynamic index management
- `misc/optimizations.py`: High-performance LLM serving optimizations

### Resume Agent

A more structured implementation of a LangGraph-based agent designed to answer questions about a resume:

- Graph-based agent workflow
- RAG chain for document retrieval
- Custom tools for resume search

## Getting Started

### Prerequisites

- Python 3.12+
- Local LLM server (Ollama recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/spyker77/agentic-rag.git
cd agentic-rag

# Install dependencies
uv sync
```

### Running Demos

```bash
python -m demos.crewai

python -m demos.langchain

python -m demos.langgraph

python -m demos.llamaindex_agent

python -m demos.llamaindex_rag

python -m demos.smolagents

python -m src.resume_agent.main
```

## Project Structure

```bash
.
├── data/                 # Sample data including resume
├── misc/                 # Advanced techniques and optimizations
│   ├── dvts.py           # Diverse Verifier Tree Search
│   ├── optimizations.py  # LLM serving optimizations
│   ├── perplexity.py     # Real-time search system
│   └── rag.py            # Advanced RAG implementation
├── src/                  # Source code
│   └── resume_agent/     # Structured LangGraph agent implementation
├── demos/                # Demos with different frameworks
```

## Disclaimer

This repository is intended as an experimental playground for learning and exploration. The code is not optimized for production use and may contain experimental patterns and approaches.

## License

MIT
