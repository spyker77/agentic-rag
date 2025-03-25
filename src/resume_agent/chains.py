from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate


def create_rag_chain(llm, vector_store):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the question based on the context provided."),
            ("human", "Context: {context}\nQuestion: {input}"),
        ]
    )

    docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(vector_store.as_retriever(), docs_chain)
