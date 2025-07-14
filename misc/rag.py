"""
Advanced RAG System with HNSW Vector Search

This module implements a production-grade RAG system using FAISS HNSW indexing for
efficient similarity search. It handles document chunking, embedding, and retrieval
with a focus on performance and result quality.

Key features:
- Document chunking with configurable overlap
- HNSW indexing for faster approximate nearest neighbor search
- Asynchronous processing for better throughput
- Metadata preservation throughout the retrieval pipeline
- Score thresholding to filter low-quality matches

Designed for applications requiring high-quality retrieval from large document collections.
"""

import asyncio
import logging
from dataclasses import dataclass

import faiss
import numpy as np
from langchain.llms import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Document:
    id: str
    content: str
    metadata: dict
    chunks: list[str]


@dataclass
class SearchResult:
    doc_id: str
    chunk_text: str
    doc_content: str
    metadata: dict
    score: float


class RAGSystem:
    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "embaas/sentence-transformers-e5-large-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 100,
    ):
        # Embedding model
        self.encoder = SentenceTransformer(model_name)
        self.dimension = self.encoder.get_sentence_embedding_dimension()

        # LLM
        self.llm = OpenAI(api_key=openai_api_key, model="gpt-4.1-mini", temperature=0.3)

        # Initialize HNSW index for better performance
        self.index = faiss.IndexHNSWFlat(self.dimension, M=16)
        self.index.hnsw.efConstruction = 512
        self.index.hnsw.efSearch = 64

        # Storage
        self.documents: dict[str, Document] = {}
        self.chunk_to_doc: dict[int, str] = {}

        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    async def add_documents(self, documents: list[Document]) -> None:
        """Process and index documents with their chunks"""
        try:
            all_chunks = []
            current_chunk_id = len(self.chunk_to_doc)

            for doc in documents:
                chunks = self.text_splitter.split_text(doc.content)
                doc.chunks = chunks
                self.documents[doc.id] = doc

                for _ in chunks:
                    self.chunk_to_doc[current_chunk_id] = doc.id
                    current_chunk_id += 1

                all_chunks.extend(chunks)

            # Generate embeddings
            embeddings = self.encoder.encode(
                all_chunks,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )

            # Add to index
            for embedding in embeddings:
                self.index.add(embedding.reshape(1, -1).astype(np.float32))

            logger.info(f"Indexed {len(all_chunks)} chunks from {len(documents)} documents")

        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            raise

    async def search_and_respond(self, query: str, k: int = 5, threshold: float = 0.7) -> dict:
        """Search and generate response"""
        try:
            if not self.documents:
                return {"response": "No documents indexed yet.", "sources": [], "search_results": []}

            # Generate query embedding
            query_embedding = self.encoder.encode(query, normalize_embeddings=True)

            # Search
            scores, indices = self.index.search(query_embedding.reshape(1, -1).astype(np.float32), k)

            # Process results
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or score > threshold:
                    continue

                doc_id = self.chunk_to_doc[idx]
                doc = self.documents[doc_id]
                chunk_idx = idx - (idx // len(doc.chunks) * len(doc.chunks))

                search_results.append(
                    SearchResult(
                        doc_id=doc_id,
                        chunk_text=doc.chunks[chunk_idx],
                        doc_content=doc.content,
                        metadata=doc.metadata,
                        score=float(score),
                    )
                )

            if not search_results:
                return {"response": "No relevant information found.", "sources": [], "search_results": []}

            # Prepare context for LLM
            contexts = [result.chunk_text for result in search_results]
            sources = [result.metadata for result in search_results]

            # Generate response
            prompt = f"""
            Answer the query based on the given contexts.
            Query: {query}
            Contexts: {contexts}

            Instructions:
            1. Use only the information from the provided contexts
            2. Be concise and specific
            3. If the context doesn't contain relevant information, say so

            Response:"""

            response = await asyncio.to_thread(self.llm.predict, prompt)

            return {
                "response": response,
                "sources": sources,
                "search_results": [
                    {"doc_id": r.doc_id, "chunk_text": r.chunk_text, "metadata": r.metadata, "score": r.score}
                    for r in search_results
                ],
            }

        except Exception as e:
            logger.error(f"Error in search and respond: {e}")
            raise

    def clear_index(self):
        """Clear the index and stored data"""
        self.index = faiss.IndexHNSWFlat(self.dimension, M=16)
        self.index.hnsw.efConstruction = 512
        self.index.hnsw.efSearch = 64
        self.documents.clear()
        self.chunk_to_doc.clear()
