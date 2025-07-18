import asyncio

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llama_index.core.workflow import Context
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDINGS_MODEL)
Settings.llm = Ollama(model="llama3.3:70b")

documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)


class QueryProcessor:
    def __init__(self, index: VectorStoreIndex):
        self.index = index
        self.agent = self._create_agent()
        self.ctx = Context(self.agent)

    def _create_tools(self) -> list[QueryEngineTool]:
        """Create query tools for different use cases."""
        return [
            QueryEngineTool.from_defaults(
                query_engine=self.index.as_query_engine(similarity_top_k=0),
                name="general_knowledge",
                description="For general knowledge questions without needing document context",
            ),
            QueryEngineTool.from_defaults(
                query_engine=self.index.as_query_engine(similarity_top_k=5),
                name="document_search",
                description="For questions requiring specific document context and retrieval",
            ),
        ]

    def _create_agent(self) -> FunctionAgent:
        """Create FunctionAgent with workflow support."""
        tools = self._create_tools()
        system_prompt = """You are a helpful assistant. Use the available tools to answer questions.
        
        - Use 'general_knowledge' for basic questions that don't need specific documents
        - Use 'document_search' for questions that need information from documents
        - Be concise and accurate in your responses"""

        return FunctionAgent(tools=tools, llm=Settings.llm, system_prompt=system_prompt)

    async def process_query(self, question: str) -> str:
        """Process query using workflow-based agent."""
        try:
            handler = self.agent.run(question, ctx=self.ctx)
            response = await handler
            return str(response)
        except Exception as e:
            return f"Error processing query: {str(e)}"


processor = QueryProcessor(index)

questions = [
    "What companies Evgeni Sautin has worked for?",
    "What is the capital of France?",
    "What is Evgeni's favorite color?",
    "What is 5 + 3?",
]


async def main():
    for question in questions:
        print(f"\n❓ Question: {question}")
        answer = await processor.process_query(question)
        print(f"💬 Answer: {answer}")


asyncio.run(main())
