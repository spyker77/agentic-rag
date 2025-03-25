from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from smolagents import CodeAgent, LiteLLMModel, Tool, tool

# EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_MODEL = "intfloat/e5-large-v2"

embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
llm = LiteLLMModel(model_id="ollama_chat/llama3.1:8b", api_base="http://localhost:11434")

documents = DirectoryLoader("data/").load()
texts = RecursiveCharacterTextSplitter().split_documents(documents)

vector_store = FAISS.from_documents(texts, embeddings)


# class VectorSearchTool(Tool):
#     name = "vector_search"
#     description = "This is a tool that performs semantic search to find relevant information in the document store."
#     inputs = {
#         "query": {
#             "type": "string",
#             "description": "A specific query or question to search for in the documents.",
#         }
#     }
#     output_type = "string"

#     def __init__(self, texts, embeddings, **kwargs):
#         super().__init__(**kwargs)
#         self.vector_store = FAISS.from_documents(texts, embeddings)

#     def forward(self, query: str) -> str:
#         assert isinstance(query, str), "Your search query must be a string"

#         docs = self.vector_store.similarity_search(query)
#         return "\n".join(f"Document {i}: {doc.page_content}" for i, doc in enumerate(docs))


# vector_search_tool = VectorSearchTool(texts, embeddings)


@tool
def vector_search_tool(query: str) -> str:
    """
    This is a tool that performs semantic search to find relevant information in the document store.

    Args:
        query: A specific query or question to search for in the documents.

    Returns:
        A string containing the most relevant document snippets.
    """
    docs = vector_store.similarity_search(query)
    return "\n".join(f"Document {i}: {doc.page_content}" for i, doc in enumerate(docs))


agent = CodeAgent(tools=[vector_search_tool], model=llm)

agent.run("What companies Evgeni Sautin has worked for?")
