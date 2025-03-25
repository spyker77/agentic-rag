from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# EMBEDDINGS_MODEL = "all-MiniLM-L6-v2"
EMBEDDINGS_MODEL = "intfloat/e5-large-v2"


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
llm = ChatOllama(model="llama3.1:8b", temperature=0)

documents = DirectoryLoader("data/").load()
texts = RecursiveCharacterTextSplitter().split_documents(documents)

vector_store = FAISS.from_documents(texts, embeddings)

prompt = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("human", "Context: {context}\nQuestion: {input}\n\nAnswer:"),
    ]
)

docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), docs_chain)

questions = [
    # "What is the experience of Evgeni Sautin?",
    # "What are the main skills of Evgeni Sautin?",
    "What companies Evgeni Sautin has worked for?",
]

for question in questions:
    response = retrieval_chain.invoke({"input": question})
    print(f"\nQuestion: {question}")
    print(f"Answer: {response['answer']}\n")
