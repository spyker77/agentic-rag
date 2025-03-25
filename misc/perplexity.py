"""
Real-Time Search System with Dynamic Index Management

This module implements a search system inspired by Perplexity's approach, with dynamic
index management that refreshes for each query. It uses OpenAI embeddings and GPT-4o
for response generation.

Key features:
- Dynamic indexing of search results
- Automatic index clearing after each query
- Asynchronous embedding generation and search
- Quality filtering based on similarity scores
- Structured response format with source tracking

Ideal for applications where search results change frequently and freshness is critical.
"""

import asyncio
import logging
from dataclasses import dataclass

import faiss
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    text: str
    metadata: dict
    score: float


class RealTimeSearchSystem:
    def __init__(self, openai_api_key: str, dimension: int = 1536):
        self.embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
        self.llm = OpenAI(api_key=openai_api_key, model="gpt-4o", temperature=0.3)
        self.index = faiss.IndexFlatIP(dimension)
        self.texts: list[str] = []
        self.metadata: list[dict] = []

    async def process_search_results(self, texts: list[str], metadata: list[dict]) -> None:
        """Process and index search results"""
        try:
            # Generate embeddings
            embeddings = await asyncio.gather(*[asyncio.to_thread(self.embeddings.embed_query, text) for text in texts])

            # Convert to numpy array
            vectors = np.array(embeddings).astype("float32")

            # Add to FAISS index
            self.index.add(vectors)

            # Store texts and metadata
            self.texts.extend(texts)
            self.metadata.extend(metadata)

            logger.info(f"Indexed {len(texts)} new documents. Total: {len(self.texts)}")

        except Exception as e:
            logger.error(f"Error processing search results: {str(e)}")
            raise

    async def search_and_respond(self, query: str, k: int = 5, score_threshold: float = 0.7) -> dict:
        """Search similar contexts and generate response"""
        try:
            if not self.texts:
                return {"response": "No documents indexed yet.", "sources": [], "search_results": []}

            # Generate query embedding
            query_vector = await asyncio.to_thread(self.embeddings.embed_query, query)

            # Convert to numpy array
            query_vector_np = np.array([query_vector]).astype("float32")

            # Search in FAISS
            scores, indices = self.index.search(query_vector_np, k)

            # Filter and prepare results
            search_results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= score_threshold:
                    search_results.append(
                        SearchResult(text=self.texts[idx], metadata=self.metadata[idx], score=float(score))
                    )

            if not search_results:
                return {"response": "No relevant information found.", "sources": [], "search_results": []}

            # Prepare context for LLM
            contexts = [result.text for result in search_results]
            sources = [result.metadata for result in search_results]

            # Generate response using LLM
            prompt = f"""Based on the following contexts, answer the query.
            Query: {query}
            Contexts: {contexts}
            
            Provide a comprehensive answer using the given context."""

            response = await asyncio.to_thread(self.llm.predict, prompt)

            self.clear_index()

            return {
                "response": response,
                "sources": sources,
                "search_results": [{"text": r.text, "metadata": r.metadata, "score": r.score} for r in search_results],
            }

        except Exception as e:
            logger.error(f"Error in search and respond: {str(e)}")
            raise

    def clear_index(self):
        """Clear the index and stored data"""
        self.index.reset()
        self.texts.clear()
        self.metadata.clear()
