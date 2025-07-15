import os
import time

import pytest
from deepeval import evaluate
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case.llm_test_case import LLMTestCase
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from src.resume_agent.chains import create_rag_chain
from src.resume_agent.config import EMBEDDINGS, LLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLLM(DeepEvalBaseLLM):
    """Custom LLM class for using with DeepEval."""

    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        """Generate response using LLM."""
        try:
            model = self.load_model()
            response = model.invoke(prompt)
            content = response.content
            return content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error: Could not generate response"

    async def a_generate(self, prompt: str) -> str:
        """Async version of generate."""
        try:
            model = self.load_model()
            response = await model.ainvoke(prompt)
            content = response.content
            return content.strip()
        except Exception as e:
            print(f"Error generating async response: {e}")
            return "Error: Could not generate response"

    def get_model_name(self):
        return "Custom LLM Model"


@pytest.fixture(scope="session")
def documents():
    loader = PyPDFLoader("data/resume.pdf")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)


@pytest.fixture(scope="session")
def rag_chain(documents):
    vectorstore = Chroma.from_documents(documents, EMBEDDINGS)
    return create_rag_chain(LLM, vectorstore)


@pytest.fixture(scope="session")
def custom_llm():
    return CustomLLM(model=LLM)


def _evaluate_question(question: str, rag_chain):
    try:
        response = rag_chain.invoke({"input": question})
        return {"answer": response["answer"], "context": [context.page_content for context in response["context"]]}
    except Exception as e:
        print(f"Error processing question: {e}")
        return {"answer": "Error: Could not generate answer", "context": ["Error: Could not retrieve context"]}


def _create_test_cases(questions: list[str], rag_chain):
    test_cases = []
    for question in questions:
        result = _evaluate_question(question, rag_chain)
        test_case = LLMTestCase(input=question, actual_output=result["answer"], retrieval_context=result["context"])
        test_cases.append(test_case)
    return test_cases


def test_basic_factual_questions(rag_chain, custom_llm):
    questions = [
        "What is the person's name?",
        "What programming languages does the person know?",
        "What is their current location?",
    ]

    test_cases = _create_test_cases(questions, rag_chain)

    metrics = [
        AnswerRelevancyMetric(threshold=0.7, model=custom_llm),
        FaithfulnessMetric(threshold=0.7, model=custom_llm),
    ]

    results = evaluate(test_cases, metrics)

    for result in results.test_results:
        assert result.success, f"Test failed: {result.input}"


def test_technical_questions(rag_chain, custom_llm):
    questions = [
        "What machine learning experience does the person have?",
        "What databases have they worked with?",
        "What frameworks do they know?",
    ]

    test_cases = _create_test_cases(questions, rag_chain)

    metrics = [
        AnswerRelevancyMetric(threshold=0.6, model=custom_llm),
        FaithfulnessMetric(threshold=0.6, model=custom_llm),
    ]

    results = evaluate(test_cases, metrics)

    for result in results.test_results:
        assert result.success, f"Test failed: {result.input}"


def test_ambiguous_questions(rag_chain, custom_llm):
    questions = [
        "What do they do?",
        "How good are they?",
        "What about projects?",
    ]

    test_cases = _create_test_cases(questions, rag_chain)

    metrics = [AnswerRelevancyMetric(threshold=0.5, model=custom_llm)]

    results = evaluate(test_cases, metrics)  # type: ignore

    for result in results.test_results:
        assert result.success, f"Test failed: {result.input}"


def test_out_of_scope_questions(rag_chain, custom_llm):
    questions = [
        "What is the weather like today?",
        "How do I bake a cake?",
        "What is the capital of France?",
    ]

    test_cases = _create_test_cases(questions, rag_chain)

    metrics = [FaithfulnessMetric(threshold=0.5, model=custom_llm)]

    results = evaluate(test_cases, metrics)  # type: ignore

    assert len(results.test_results) == len(test_cases)


def test_response_time(rag_chain):
    question = "What is the person's current job?"

    start_time = time.time()
    result = _evaluate_question(question, rag_chain)
    end_time = time.time()

    response_time = end_time - start_time

    assert response_time < 10.0, f"Response took {response_time:.2f} seconds"
    assert result["answer"] is not None
    assert len(result["answer"]) > 0


def test_context_retrieval(rag_chain):
    question = "What frameworks does the person know?"
    result = _evaluate_question(question, rag_chain)

    assert len(result["context"]) > 0, "No contexts retrieved"
    assert len(result["context"]) <= 10, f"Too many contexts retrieved: {len(result['context'])}"


def test_professional_tone(rag_chain, custom_llm):
    questions = [
        "What is your experience with Python?",
        "Tell me about your work history?",
        "What are your strengths?",
    ]

    for question in questions:
        result = _evaluate_question(question, rag_chain)

        unprofessional_words = ["sucks", "hate", "terrible", "awful", "stupid"]
        answer_lower = result["answer"].lower()

        for word in unprofessional_words:
            assert word not in answer_lower, f"Unprofessional word '{word}' found in answer"

    test_cases = _create_test_cases(questions, rag_chain)

    metrics = [AnswerRelevancyMetric(threshold=0.5, model=custom_llm)]

    results = evaluate(test_cases, metrics)  # type: ignore

    for result in results.test_results:
        assert result.success, f"Test failed: {result.input}"


def test_empty_input(rag_chain):
    question = ""
    result = _evaluate_question(question, rag_chain)

    assert result["answer"] is not None
    assert len(result["answer"]) > 0
    assert result["context"] is not None
