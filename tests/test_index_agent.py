# from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.query_engine import RouterQueryEngine
# from llama_index.core.selectors import LLMSingleSelector
# from llama_index.core.tools import QueryEngineTool
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama

# # EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
# EMBEDDINGS_MODEL = "intfloat/e5-large-v2"

# # Initialize settings
# Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDINGS_MODEL)
# Settings.llm = Ollama(model="llama3.1:8b")

# # Load and index documents
# documents = SimpleDirectoryReader("data/").load_data()
# index = VectorStoreIndex.from_documents(documents)

# # Create query engine tools
# direct_tool = QueryEngineTool.from_defaults(
#     query_engine=index.as_query_engine(similarity_top_k=0),
#     description="Use for general knowledge questions that don't need specific context",
# )

# rag_tool = QueryEngineTool.from_defaults(
#     query_engine=index.as_query_engine(similarity_top_k=5),
#     description="Use for questions requiring specific document context and detailed information",
# )

# # Create router query engine
# router = RouterQueryEngine(selector=LLMSingleSelector.from_defaults(), query_engine_tools=[direct_tool, rag_tool])


# # Test questions
# questions = [
#     "What companies Evgeni Sautin has worked for?",
#     "What is the capital of France?",
#     "What is Evgeni's favorite color?",
# ]

# # Process questions
# for question in questions:
#     print(f"\nQuestion: {question}")
#     response = router.query(question)
#     print(f"Answer: {response}\n")


# from enum import StrEnum

# from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.evaluation import ResponseEvaluator
# from llama_index.core.query_engine import RouterQueryEngine
# from llama_index.core.selectors import LLMSingleSelector
# from llama_index.core.tools import QueryEngineTool
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama

# # EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
# EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


# Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDINGS_MODEL)
# Settings.llm = Ollama(model="llama3.1:8b")


# documents = SimpleDirectoryReader("data/").load_data()
# index = VectorStoreIndex.from_documents(documents)


# class ProcessingStatus(StrEnum):
#     COMPLETE = "COMPLETE"
#     ERROR = "ERROR"


# class QueryProcessor:
#     def __init__(self, index: VectorStoreIndex):
#         self.index = index
#         self.evaluator = ResponseEvaluator()
#         self.router = self._create_router()

#     def _create_router(self) -> RouterQueryEngine:
#         """Create router with direct and RAG query engines."""
#         tools = [
#             QueryEngineTool.from_defaults(
#                 query_engine=self.index.as_query_engine(similarity_top_k=0),
#                 description="Use for general knowledge questions that don't need specific context",
#             ),
#             QueryEngineTool.from_defaults(
#                 query_engine=self.index.as_query_engine(similarity_top_k=5),
#                 description="Use for questions requiring specific document context",
#             ),
#         ]
#         return RouterQueryEngine(selector=LLMSingleSelector.from_defaults(), query_engine_tools=tools)

#     def evaluate_response(self, question: str, response) -> bool:
#         """Evaluate response quality."""
#         try:
#             eval_result = self.evaluator.evaluate(
#                 query=question,
#                 response=response.response,
#                 contexts=[response.response],
#             )
#             return eval_result.score >= 0.7
#         except Exception as e:
#             print(f"Evaluation error: {str(e)}")
#             return False

#     def process_query(self, question: str) -> tuple[str, ProcessingStatus]:
#         """Process query with evaluation."""
#         try:
#             response = self.router.query(question)
#             if self.evaluate_response(question, response):
#                 return response.response, ProcessingStatus.COMPLETE
#             return f"Low confidence in response: {response.response}", ProcessingStatus.ERROR
#         except Exception as e:
#             return f"Error processing query: {str(e)}", ProcessingStatus.ERROR


# processor = QueryProcessor(index)

# questions = [
#     "What companies Evgeni Sautin has worked for?",
#     "What is the capital of France?",
#     "What is Evgeni's favorite color?",
# ]

# for question in questions:
#     print(f"\nQuestion: {question}")
#     answer, status = processor.process_query(question)
#     print(f"Answer: {answer}")
#     print(f"Status: {status}")

# import asyncio

# from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
# from llama_index.core.agent import ReActAgent
# from llama_index.core.memory import ChatMemoryBuffer
# from llama_index.core.tools import QueryEngineTool
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.ollama import Ollama

# # EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
# EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


# Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDINGS_MODEL)
# Settings.llm = Ollama(model="llama3.1:8b")


# documents = SimpleDirectoryReader("data/").load_data()
# index = VectorStoreIndex.from_documents(documents)


# class AdvancedQueryProcessor:
#     def __init__(self, index: VectorStoreIndex):
#         self.index = index
#         self.memory = ChatMemoryBuffer.from_defaults()
#         self.agent = self._create_agent()

#     def _create_tools(self) -> list[QueryEngineTool]:
#         """Create query tools with different capabilities."""
#         return [
#             QueryEngineTool.from_defaults(
#                 query_engine=self.index.as_query_engine(similarity_top_k=0),
#                 description="For general knowledge questions without needing document context",
#             ),
#             QueryEngineTool.from_defaults(
#                 query_engine=self.index.as_query_engine(similarity_top_k=5),
#                 description="For questions requiring document search and information retrieval",
#             ),
#         ]

#     def _create_agent(self) -> ReActAgent:
#         """Create ReAct agent with tools and memory."""
#         tools = self._create_tools()
#         return ReActAgent.from_tools(tools=tools, memory=self.memory, verbose=False)

#     async def process_query(self, question: str) -> str:
#         """Process query using agent-based reasoning."""
#         try:
#             return await self.agent.aquery(question)
#         except Exception as e:
#             return f"Error processing query: {str(e)}"


# processor = AdvancedQueryProcessor(index)


# questions = [
#     "What companies Evgeni Sautin has worked for?",
#     "What is the capital of France?",
#     "What is Evgeni's favorite color?",
# ]


# async def main():
#     for question in questions:
#         print(f"\nQuestion: {question}")
#         print("-" * 50)
#         answer = await processor.process_query(question)
#         print(f"Answer: {answer}")
#         print("-" * 50)


# asyncio.run(main())


import asyncio

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent import ParallelAgentRunner, ReActAgentWorker
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.tools import QueryEngineTool
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever

# EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDINGS_MODEL)
Settings.llm = Ollama(model="llama3.1:8b")


documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)


class AdvancedQueryProcessor:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.memory = ChatMemoryBuffer.from_defaults()
        self.agent_worker = self._create_agent_worker()
        self.agent = ParallelAgentRunner(agent_worker=self.agent_worker, memory=self.memory)

    def _create_tools(self) -> list[QueryEngineTool]:
        vector_retriever = self.index.as_retriever(similarity_top_k=2)
        bm25_retriever = BM25Retriever.from_defaults(docstore=self.index.docstore, similarity_top_k=2)
        hybrid_retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            mode="reciprocal_rerank",
            similarity_top_k=2,
            num_queries=1,
        )
        hybrid_engine = RetrieverQueryEngine(retriever=hybrid_retriever)
        return [
            QueryEngineTool.from_defaults(
                query_engine=hybrid_engine,
                description="For comprehensive search using both semantic and keyword matching",
            )
        ]

    def _create_agent_worker(self) -> ReActAgentWorker:
        tools = self._create_tools()
        return ReActAgentWorker.from_tools(tools=tools, verbose=False)

    async def process_query(self, question: str) -> str:
        try:
            task = self.agent.create_task(question)

            while True:
                step_output = await self.agent.arun_step(task.task_id, input=question)
                if step_output.is_last:
                    break

            output = self.agent.get_task_output(task.task_id)
            return str(output)
        except Exception as e:
            return f"Error processing query: {str(e)}"


processor = AdvancedQueryProcessor(index)


questions = [
    "What companies Evgeni Sautin has worked for?",
    "What is the capital of France?",
    "What is Evgeni's favorite color?",
]


async def main():
    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)
        answer = await processor.process_query(question)
        print(f"Answer: {answer}")
        print("-" * 50)


asyncio.run(main())
