from pathlib import Path
from typing import Any, Dict

from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.components.converters import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.joiners import DocumentJoiner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.rankers import SentenceTransformersSimilarityRanker
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools.component_tool import ComponentTool
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

EMBEDDINGS_MODEL = "intfloat/e5-large-v2"
RERANKER_MODEL = "BAAI/bge-reranker-base"


llm = OllamaChatGenerator(model="llama3.3:70b")


document_store = InMemoryDocumentStore()


indexing_pipeline = Pipeline()
indexing_pipeline.add_component("converter", PyPDFToDocument())
indexing_pipeline.add_component("splitter", DocumentSplitter(split_by="word", split_length=400, split_overlap=100))
indexing_pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(model=EMBEDDINGS_MODEL))
indexing_pipeline.add_component("writer", DocumentWriter(document_store=document_store))

indexing_pipeline.connect("converter", "splitter")
indexing_pipeline.connect("splitter", "embedder")
indexing_pipeline.connect("embedder", "writer")


data_dir = Path(__file__).parent.parent / "data"
pdf_files = list(data_dir.glob("*.pdf"))
indexing_pipeline.run({"converter": {"sources": pdf_files}})


@component
class HaystackTwoStageSearcher:
    def __init__(self, document_store: InMemoryDocumentStore, first_stage_k: int = 50, final_k: int = 10):
        print(f"Initializing Haystack two-stage retriever: {first_stage_k} -> {final_k}")

        # Stage 1: High recall retrieval components.
        self.embedding_retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=first_stage_k)
        self.bm25_retriever = InMemoryBM25Retriever(document_store=document_store, top_k=first_stage_k)
        self.embedder = SentenceTransformersTextEmbedder(model=EMBEDDINGS_MODEL)
        self.joiner = DocumentJoiner(join_mode="reciprocal_rank_fusion")

        # Stage 2: Built-in Haystack ranker for precision.
        print(f"Loading Haystack ranker with model: {RERANKER_MODEL}")
        self.ranker = SentenceTransformersSimilarityRanker(model=RERANKER_MODEL, top_k=final_k)

        self.first_stage_k = first_stage_k
        self.final_k = final_k

    @component.output_types(context=str, metadata=dict)
    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute two-stage retrieval with Haystack's built-in ranker.
        """
        if not query or not query.strip():
            return {
                "context": "No information found, the query was empty.",
                "metadata": {"stage": "input_validation", "candidates": 0, "final_count": 0},
            }

        # Stage 1: Initial retrieval with hybrid search.
        print(f"Stage 1: Retrieving up to {self.first_stage_k} candidates...")

        embedding = self.embedder.run(text=query)["embedding"]
        semantic_docs = self.embedding_retriever.run(query_embedding=embedding)["documents"]
        keyword_docs = self.bm25_retriever.run(query=query)["documents"]

        # Combine using reciprocal rank fusion.
        joined_docs = self.joiner.run(documents=[semantic_docs, keyword_docs])["documents"]

        if not joined_docs:
            return {
                "context": "No information found in the documents for this query.",
                "metadata": {"stage": "first_stage", "candidates": 0, "final_count": 0},
            }

        print(f"Stage 1 complete: Found {len(joined_docs)} candidates")

        # Stage 2: Re-ranking with Haystack's built-in ranker.
        if len(joined_docs) > self.final_k:
            print(f"Stage 2: Re-ranking {len(joined_docs)} candidates with Haystack ranker...")
            ranker_result = self.ranker.run(query=query, documents=joined_docs)
            reranked_docs = ranker_result["documents"]
            print(f"Stage 2 complete: Selected top {len(reranked_docs)} documents")
        else:
            reranked_docs = joined_docs[: self.final_k]
            print(f"Skipped re-ranking: Only {len(joined_docs)} candidates found")

        # Format final context.
        context_string = "\n\n".join([doc.content for doc in reranked_docs if doc.content])

        metadata = {
            "stage": "complete",
            "candidates": len(joined_docs),
            "final_count": len(reranked_docs),
            "reranked": len(joined_docs) > self.final_k,
            "ranker_used": "SentenceTransformersSimilarityRanker",
        }

        return {"context": context_string, "metadata": metadata}


document_searcher = HaystackTwoStageSearcher(document_store, first_stage_k=50, final_k=10)
document_searcher.embedder.warm_up()


document_search_tool = ComponentTool(
    component=document_searcher,
    name="document_searcher",
    description="Search documents using Haystack's two-stage retrieval with SentenceTransformersSimilarityRanker. Use ONLY for questions about the person/people in the documents.",
)


AGENT_PROMPT = """You are a helpful assistant. You MUST follow this routing logic strictly:

🚨 MANDATORY ROUTING RULES - NO EXCEPTIONS:

📚 MATH & GENERAL KNOWLEDGE → ANSWER DIRECTLY (NO TOOLS):
- Math: "What is 15 + 3?" → Answer: "18" 
- Geography: "What is the capital of Germany?" → Answer: "Berlin"
- Science: "What is gravity?" → Answer directly
- History: "Who invented the telephone?" → Answer directly
- NEVER use document_searcher for these!

👤 PERSON/DOCUMENT QUESTIONS → USE TOOL:
- "What companies has the person worked for?" → Use document_searcher
- "What are the person's skills?" → Use document_searcher  
- "What is the person's favorite color?" → Use document_searcher (then say not found if needed)

🔍 DECISION FLOWCHART:
1. Is it math (like 15+3)? → Answer directly
2. Is it world knowledge (capitals, science)? → Answer directly  
3. Is it about the person in documents? → Use tool
4. When in doubt → Use tool

⚠️ CRITICAL: NEVER use document_searcher for basic math or world facts!

Examples:
- "What is 2+2?" → "4" (no tool)
- "Capital of Japan?" → "Tokyo" (no tool)  
- "Person's work history?" → Use document_searcher tool"""


agent = Agent(chat_generator=llm, system_prompt=AGENT_PROMPT, tools=[document_search_tool])


questions = [
    "What companies Evgeni Sautin has worked for?",
    "What is the capital of France?",
    "What is Evgeni's favorite color?",
    "What is 5 + 3?",
]


for question in questions:
    print(f"\n❓ Question: {question}")
    result = agent.run(messages=[ChatMessage.from_user(question)])
    final_answer = result["messages"][-1].text
    print(f"💬 Answer: {final_answer}")
