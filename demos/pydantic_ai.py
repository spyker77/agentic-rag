import os
import random
from dataclasses import dataclass

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# 1. STRUCTURED RESPONSE MODELS
# =============================================================================


class ResumeResponse(BaseModel):
    """Structured response for resume-related queries."""

    answer: str = Field(description="The main answer to the user's question")
    confidence: float = Field(description="Confidence score (0.0-1.0)", ge=0.0, le=1.0)
    sources: list[str] = Field(description="Sources/sections used for the answer")
    needs_clarification: bool = Field(description="Whether the question needs clarification")


class QueryAnalysis(BaseModel):
    """Analysis of user query for routing decisions."""

    query_type: str = Field(description="Type of query: 'resume', 'general', 'math'")
    confidence: float = Field(description="Confidence in classification", ge=0.0, le=1.0)
    requires_documents: bool = Field(description="Whether query needs document context")
    complexity: str = Field(description="Query complexity: 'simple', 'medium', 'complex'")


class FollowUpQuestions(BaseModel):
    """A list of suggested follow-up questions."""

    questions: list[str] = Field(description="A list of 3 relevant follow-up questions based on context.")


class EnhancedResumeResponse(BaseModel):
    """Enhanced structured response with metadata."""

    answer: str = Field(description="Comprehensive answer to the query")
    confidence: float = Field(description="Response confidence", ge=0.0, le=1.0)
    sources: list[str] = Field(description="Document sources used")
    related_topics: list[str] = Field(description="Related topics for follow-up")
    sentiment: str = Field(description="Query sentiment: 'positive', 'neutral', 'negative'")
    follow_up_suggestions: list[str] = Field(description="Suggested follow-up questions")


# =============================================================================
# 2. DEPENDENCIES AND CONTEXT
# =============================================================================


@dataclass
class RAGDependencies:
    """Dependencies for RAG-enabled agents."""

    vector_store: FAISS
    embeddings: HuggingFaceEmbeddings
    documents: list[Document]
    conversation_history: list[dict[str, str]]


@dataclass
class AgentContext:
    """Context for agent operations."""

    model_name: str
    temperature: float = 0.0
    max_tokens: int | None = None
    debug: bool = False


# =============================================================================
# 3. RAG SETUP
# =============================================================================


def setup_rag_system():
    """Set up RAG system with document processing."""
    print("🔧 Setting up RAG system...")

    documents = DirectoryLoader("data/").load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    vector_store = FAISS.from_documents(texts, embeddings)

    print(f"📄 Processed {len(documents)} documents into {len(texts)} chunks")
    return vector_store, embeddings, documents


# =============================================================================
# 4. BASIC PYDANTIC AI AGENT
# =============================================================================


def demo_basic_agent():
    """Demonstrate basic Pydantic AI agent usage."""
    print(f"\n{'=' * 60}")
    print("🤖 1. BASIC PYDANTIC AI AGENT")
    print("=" * 60)

    model = OpenAIModel(
        model_name="openai/gpt-4.1-mini",
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
    )
    agent = Agent(model, instructions="You are a helpful assistant. Be concise and accurate.", retries=3)

    questions = ["What is the capital of France?", "Explain machine learning in one sentence", "What is 15 + 27?"]

    for question in questions:
        print(f"\n❓ Question: {question}")
        try:
            result = agent.run_sync(question)
            print(f"💬 Answer: {result.output}")
        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# 5. STRUCTURED RESPONSE AGENT
# =============================================================================


def demo_structured_agent():
    """Demonstrate structured responses with Pydantic models."""
    print(f"\n{'=' * 60}")
    print("📊 2. STRUCTURED RESPONSE AGENT")
    print("=" * 60)

    model = OpenAIModel(
        model_name="openai/gpt-4.1-mini",
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
    )
    agent = Agent(
        model,
        output_type=QueryAnalysis,
        instructions=(
            "You are a query analysis expert. Analyze user queries and classify them according to their type, complexity, and requirements."
        ),
        retries=3,
    )

    test_queries = [
        "What companies has Evgeni worked for?",
        "What is the capital of France?",
        "Calculate the square root of 144",
        "Tell me about his background and experience",
    ]

    for query in test_queries:
        print(f"\n❓ Query: {query}")
        try:
            result = agent.run_sync(query)
            analysis = result.output
            print("📋 Analysis:")
            print(f"  Type: {analysis.query_type}")
            print(f"  Confidence: {analysis.confidence:.2f}")
            print(f"  Needs Documents: {analysis.requires_documents}")
            print(f"  Complexity: {analysis.complexity}")
        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# 6. RAG-ENABLED AGENT WITH TOOLS
# =============================================================================


def demo_rag_agent():
    """Demonstrate RAG integration with tools."""
    print(f"\n{'=' * 60}")
    print("🔍 3. RAG-ENABLED AGENT WITH TOOLS")
    print("=" * 60)

    vector_store, embeddings, documents = setup_rag_system()

    deps = RAGDependencies(
        vector_store=vector_store,
        embeddings=embeddings,
        documents=documents,
        conversation_history=[],
    )

    model = OpenAIModel(
        model_name="openai/gpt-4.1-mini",
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
    )
    agent = Agent(model, output_type=ResumeResponse, deps_type=RAGDependencies, retries=3)

    @agent.instructions
    def get_rag_instructions(ctx: RunContext[RAGDependencies]) -> str:
        """Dynamic instructions that can access context and dependencies."""
        doc_count = len(ctx.deps.documents)
        conversation_count = len(ctx.deps.conversation_history)

        return (
            f"You are a resume analysis assistant with access to {doc_count} documents and {conversation_count} prior exchanges."
            "Your goal is to provide a structured, factual answer to the user's query based *only* on the information in the documents."
            "Use the `search_documents` tool to find relevant information. You can use `get_conversation_context` for conversational context."
            "Analyze the results from your tools to formulate a concise answer and populate the `ResumeResponse`."
        )

    @agent.tool
    def search_documents(ctx: RunContext[RAGDependencies], query: str) -> str:
        """Search through documents for relevant information."""
        try:
            docs = ctx.deps.vector_store.similarity_search(query, k=5)
            if not docs:
                return "No relevant documents found."

            content = "\n".join([doc.page_content for doc in docs])
            return content
        except Exception as e:
            raise ModelRetry(f"Document search failed: {e}")

    @agent.tool
    def get_conversation_context(ctx: RunContext[RAGDependencies]) -> str:
        """Get recent conversation history for context."""
        history = ctx.deps.conversation_history[-3:]  # last 3 exchanges
        if not history:
            return "No previous conversation history."

        context = "Recent conversation:\n"
        for exchange in history:
            context += f"Q: {exchange['question']}\nA: {exchange['answer']}\n"
        return context

    test_queries = [
        "What companies has Evgeni worked for?",
        "What are his key technical skills?",
        "Tell me about his educational background",
        "What programming languages does he know?",
    ]

    for query in test_queries:
        print(f"\n❓ Question: {query}")
        try:
            result = agent.run_sync(query, deps=deps)
            response = result.output
            print(f"💬 Answer: {response.answer}")
            print(f"📈 Confidence: {response.confidence:.2f}")
            print(f"📄 Sources: {', '.join(response.sources) if response.sources else 'None'}")
            print(f"❓ Needs Clarification: {response.needs_clarification}")

            deps.conversation_history.append({"question": query, "answer": response.answer})

        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# 7. ENHANCED AGENT WITH MEMORY AND ROUTING
# =============================================================================


def demo_enhanced_agent():
    """Demonstrate enhanced agent with memory and smart routing."""
    print(f"\n{'=' * 60}")
    print("🧠 4. ENHANCED AGENT WITH MEMORY & ROUTING")
    print("=" * 60)

    vector_store, embeddings, documents = setup_rag_system()

    model = OpenAIModel(
        model_name="openai/gpt-4.1-mini",
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
    )
    agent = Agent(
        model,
        output_type=EnhancedResumeResponse,
        deps_type=RAGDependencies,
        instructions=(
            "You are an advanced resume analyst. Your goal is to create a detailed, structured JSON response based on a resume document that answers the user's query."
            "You have tools to help you: `smart_document_search` to find information in the resume, and `suggest_follow_ups` to generate relevant next questions."
            "Analyze the user's query and use the tools available to you to gather all the information needed to populate the `EnhancedResumeResponse` schema."
            "Note that `suggest_follow_ups` is most effective when its `search_results` argument is populated from the output of `smart_document_search`."
            "Your final output must be a single, valid JSON object that conforms to the schema."
        ),
        retries=3,
    )

    @agent.tool
    def smart_document_search(ctx: RunContext[RAGDependencies], query: str) -> str:
        """Intelligent document search with relevance scoring."""
        try:
            docs = ctx.deps.vector_store.similarity_search_with_score(query, k=5)

            if not docs:
                return "No relevant information found in documents."

            relevant_docs = [doc for doc, score in docs if score < 0.7]

            if not relevant_docs:
                return "No highly relevant information found."

            content = "Found relevant information:\n"
            for i, doc in enumerate(relevant_docs[:3], 1):
                content += f"{i}. {doc.page_content[:200]}...\n"

            return content
        except Exception as e:
            raise ModelRetry(f"Smart search failed: {e}")

    @agent.tool
    def suggest_follow_ups(ctx: RunContext[RAGDependencies], query: str, search_results: str) -> list[str]:
        """
        Generate intelligent follow-up questions based on the user's query and provided search results.
        This tool is most effective when used with the output of `smart_document_search`.
        """

        model = OpenAIModel(
            model_name="openai/gpt-4.1-nano",  # cheaper model for suggestions
            provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
        )
        generator_agent = Agent(
            model,
            output_type=FollowUpQuestions,
            instructions=(
                "You are an expert at suggesting relevant follow-up questions for a conversation about a person's resume. "
                "Based on the conversation history, the user's last query, and the retrieved document snippets, "
                "provide three insightful questions to help the user learn more. "
                "The questions should be concise and directly related to the provided context."
            ),
        )

        history = ctx.deps.conversation_history
        history_str = "No conversation history yet."
        if history:
            history_str = "Recent conversation:\n"
            for exchange in history[-3:]:
                history_str += f"Q: {exchange['question']}\nA: {exchange['answer']}\n"

        prompt = f"""
        Conversation History:
        {history_str}

        User's Last Query: {query}

        Retrieved Documents:
        {search_results}

        Generate 3 follow-up questions based on this information.
        """
        try:
            result = generator_agent.run_sync(prompt)
            return result.output.questions
        except Exception as e:
            print(f"Follow-up suggestion generation failed: {e}")
            return [
                "Tell me more about his background",
                "What are his key strengths?",
                "What type of role is he looking for?",
            ]

    conversation_history = []
    deps = RAGDependencies(
        vector_store=vector_store,
        embeddings=embeddings,
        documents=documents,
        conversation_history=conversation_history,
    )

    test_queries = [
        "What companies has Evgeni worked for?",
        "What were his main responsibilities?",
        "What technologies is he most experienced with?",
    ]

    for query in test_queries:
        print(f"\n❓ Question: {query}")
        try:
            result = agent.run_sync(query, deps=deps)
            response = result.output

            print(f"💬 Answer: {response.answer}")
            print(f"📈 Confidence: {response.confidence:.2f}")
            print(f"📄 Sources: {', '.join(response.sources) if response.sources else 'None'}")
            print(f"🔗 Related Topics: {', '.join(response.related_topics) if response.related_topics else 'None'}")
            print(f"😊 Sentiment: {response.sentiment}")
            print(
                f"❓ Follow-up Suggestions: {', '.join(response.follow_up_suggestions) if response.follow_up_suggestions else 'None'}"
            )

            conversation_history.append({"question": query, "answer": response.answer})

        except Exception as e:
            print(f"❌ Error: {e}")


# =============================================================================
# 8. PERFORMANCE AND RELIABILITY TEST
# =============================================================================


def demo_reliability():
    """Test error handling and reliability features."""
    print(f"\n{'=' * 60}")
    print("🛡️  6. RELIABILITY AND ERROR HANDLING")
    print("=" * 60)

    model = OpenAIModel(
        model_name="openai/gpt-4.1-mini",
        provider=OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"]),
    )
    agent = Agent(
        model,
        output_type=ResumeResponse,
        instructions="You are a resume assistant. Always provide structured responses.",
        retries=1,
    )

    @agent.tool
    def unreliable_search(ctx: RunContext, query: str) -> str:
        """Tool that might fail to demonstrate retry logic."""
        if random.random() < 0.8:  # 80% chance of failure
            raise ModelRetry("Simulated search failure - retrying...")

        return f"Search results for: {query}"

    print("\n🔧 Testing built-in retry mechanism and error handling:")
    test_queries = ["Test reliability query 1"]

    for query in test_queries:
        print(f"\n❓ Testing: {query}")
        from pydantic_ai import UnexpectedModelBehavior, capture_run_messages

        with capture_run_messages() as messages:
            try:
                result = agent.run_sync(query)
                print(f"✅ Success: {result.output.answer[:50]}...")
            except UnexpectedModelBehavior as e:
                print(f"❌ Final failure after retries: {e}")
                print(f"Cause: {repr(e.__cause__)}")
                print("Captured Messages:")
                for message in messages:
                    print(f"- {message}")


# =============================================================================
# 9. MAIN DEMO FUNCTION
# =============================================================================


def main():
    try:
        demo_basic_agent()
        demo_structured_agent()
        demo_rag_agent()
        demo_enhanced_agent()
        demo_reliability()
    except Exception as e:
        print(f"\n❌ Demo error: {e}")


if __name__ == "__main__":
    main()
