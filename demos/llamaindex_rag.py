from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDINGS_MODEL)
Settings.llm = Ollama(model="gemma3:27b")

documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

# 1. Basic RAG with source tracking.
print(f"\n{'=' * 50}")
print("🔍 1. BASIC RAG WITH SOURCE TRACKING")
print("=" * 50)

basic_engine = index.as_query_engine(similarity_top_k=3, response_mode=ResponseMode.COMPACT, verbose=True)

query = "What companies has Evgeni worked for?"
print(f"❓ Question: {query}")
response = basic_engine.query(query)
print(f"💬 Answer: {response}")

# Show source documents.
if hasattr(response, "source_nodes") and response.source_nodes:
    print(f"📚 Retrieved {len(response.source_nodes)} source documents:")
    for i, node in enumerate(response.source_nodes):
        score = node.score if hasattr(node, "score") else "N/A"
        # Fix node text access.
        node_text = node.get_content() if hasattr(node, "get_content") else str(node)
        text_preview = node_text[:150] + "..." if len(node_text) > 150 else node_text
        print(f"   📄 Source {i + 1} (Score: {score:.3f}): {text_preview}")

# 2. Advanced synthesis strategy.
print(f"\n{'=' * 50}")
print("🌳 2. TREE SUMMARIZE SYNTHESIS")
print("=" * 50)

detailed_engine = index.as_query_engine(similarity_top_k=8, response_mode=ResponseMode.TREE_SUMMARIZE, verbose=True)

print(f"❓ Question: {query}")
response = detailed_engine.query(query)
print(f"💬 Answer: {response}")
print(f"📊 Used {len(response.source_nodes)} sources for comprehensive analysis")

# 3. Multiple retrieval strategies comparison.
print(f"\n{'=' * 50}")
print("⚖️ 3. RETRIEVAL STRATEGY COMPARISON")
print("=" * 50)

# Compare different similarity thresholds.
focused_engine = index.as_query_engine(similarity_top_k=2, response_mode=ResponseMode.COMPACT)
broad_engine = index.as_query_engine(similarity_top_k=10, response_mode=ResponseMode.REFINE)

print(f"❓ Question: {query}")
print("\n🎯 Focused Retrieval (2 sources):")
focused_response = focused_engine.query(query)
print(f"💬 Answer: {focused_response}")

print("\n🌐 Broad Retrieval (10 sources):")
broad_response = broad_engine.query(query)
print(f"💬 Answer: {broad_response}")

# 4. Intelligent router.
print(f"\n{'=' * 50}")
print("🎯 4. INTELLIGENT QUERY ROUTING")
print("=" * 50)

precise_tool = QueryEngineTool(
    query_engine=basic_engine,
    metadata=ToolMetadata(
        name="precise_search",
        description="Best for specific factual questions requiring precise answers",
    ),
)

comprehensive_tool = QueryEngineTool(
    query_engine=detailed_engine,
    metadata=ToolMetadata(
        name="comprehensive_search",
        description="Best for overview questions requiring detailed analysis",
    ),
)

router_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[precise_tool, comprehensive_tool],
    verbose=True,
)

complex_query = "Give me a comprehensive overview of Evgeni's professional background and skills"
print(f"❓ Question: {complex_query}")
print("🤖 Router analyzing query type...")
response = router_engine.query(complex_query)
print(f"💬 Answer: {response}")
