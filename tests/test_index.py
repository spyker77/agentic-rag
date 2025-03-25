from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_MODEL = "intfloat/e5-large-v2"

Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDINGS_MODEL)
Settings.llm = Ollama(model="llama3.1:8b")

documents = SimpleDirectoryReader("data/").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(similarity_top_k=5)

questions = ["What companies Evgeni Sautin has worked for?"]
for question in questions:
    response = query_engine.query(question)
    print(f"\nQuestion: {question}")
    print(f"Answer: {response}\n")
