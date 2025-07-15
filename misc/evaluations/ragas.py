from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.testset import TestsetGenerator

from src.resume_agent.chains import create_rag_chain
from src.resume_agent.config import EMBEDDINGS, LLM


def load_and_split_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load and split documents into chunks."""
    print(f"Loading document: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    print(f"Splitting document into chunks (size={chunk_size}, overlap={chunk_overlap})")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    print(f"Created {len(docs)} document chunks")

    return docs


def generate_test_set(docs, testset_size: int = 10):
    """Generate test questions and ground truths from documents."""
    print(f"Generating test set with {testset_size} questions...")

    ragas_llm = LangchainLLMWrapper(LLM)
    ragas_embeddings = LangchainEmbeddingsWrapper(EMBEDDINGS)

    generator = TestsetGenerator(llm=ragas_llm, embedding_model=ragas_embeddings)
    testset = generator.generate_with_langchain_docs(docs, testset_size=testset_size)

    test_df = testset.to_pandas()
    test_questions = test_df["user_input"].values.tolist()
    test_groundtruths = test_df["reference"].values.tolist()

    print(f"Generated {len(test_questions)} test questions")
    return test_questions, test_groundtruths


def evaluate_rag_chain(docs, test_questions: list[str]):
    """Evaluate the RAG chain on test questions."""
    print("Creating vector store and RAG chain...")

    vectorstore = Chroma.from_documents(docs, EMBEDDINGS)
    rag_chain = create_rag_chain(LLM, vectorstore)

    print("Running RAG chain on test questions...")
    answers = []
    contexts = []

    for i, question in enumerate(test_questions, 1):
        print(f"Processing question {i}/{len(test_questions)}")
        try:
            response = rag_chain.invoke({"input": question})
            answers.append(response["answer"])
            contexts.append([context.page_content for context in response["context"]])
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            answers.append("Error: Could not generate answer")
            contexts.append(["Error: Could not retrieve context"])

    return answers, contexts


def create_evaluation_dataset(
    test_questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    test_groundtruths: list[str],
) -> EvaluationDataset:
    """Create evaluation dataset from questions, answers, contexts, and ground truths."""
    print("Creating evaluation dataset...")

    samples = []
    for question, answer, context, ground_truth in zip(test_questions, answers, contexts, test_groundtruths):
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=context,
            reference=ground_truth,
        )
        samples.append(sample)

    return EvaluationDataset(samples=samples)


def run_evaluation(dataset: EvaluationDataset):
    """Run ragas evaluation on the dataset."""
    print("Running ragas evaluation...")

    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
        answer_correctness,
    ]

    ragas_llm = LangchainLLMWrapper(LLM)
    ragas_embeddings = LangchainEmbeddingsWrapper(EMBEDDINGS)

    try:
        results = evaluate(dataset, metrics, llm=ragas_llm, embeddings=ragas_embeddings)
        return results
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


def main(document_path: str, testset_size: int):
    """Main function to run the RAG evaluation."""
    try:
        docs = load_and_split_documents(document_path)

        test_questions, test_groundtruths = generate_test_set(docs, testset_size)

        answers, contexts = evaluate_rag_chain(docs, test_questions)

        dataset = create_evaluation_dataset(test_questions, answers, contexts, test_groundtruths)

        results = run_evaluation(dataset)

        results_df = results.to_pandas()

        # pd.set_option("display.max_columns", None)
        # pd.set_option("display.width", None)

        print("\n" + "=" * 50)
        print("RAG EVALUATION RESULTS")
        print("=" * 50)
        print(results_df)

    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main(document_path="data/resume.pdf", testset_size=10)
