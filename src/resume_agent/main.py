from langchain_core.messages import HumanMessage

from src.resume_agent import create_app, create_demo_app, create_enhanced_app
from src.resume_agent.agents import ProcessingStatus


def create_initial_state(question: str):
    """Create initial state for enhanced workflow."""
    return {
        "messages": [HumanMessage(content=question)],
        "remaining_steps": 10,
        "processing_status": ProcessingStatus.RUNNING,
        "error_count": 0,
        "routing_decision": "",
        "conversation_history": [],
        "loop_step": 0,
        "last_error": "",
    }


def test_original_app():
    """Test the original simple application."""
    print("=" * 60)
    print("🧱 ORIGINAL SIMPLE APPLICATION")
    print("=" * 60)

    chain = create_app()

    questions = ["What companies Evgeni Sautin has worked for?"]
    for question in questions:
        print(f"\n❓ Question: {question}")
        state = create_initial_state(question)
        response = chain.invoke(state)
        print(f"💬 Answer: {response['messages'][-1].content}")


def test_enhanced_app():
    """Test the enhanced application with sophisticated capabilities."""
    print("\n" + "=" * 60)
    print("🚀 ENHANCED APPLICATION WITH SMART ROUTING")
    print("=" * 60)

    workflow = create_enhanced_app()

    questions = [
        "What companies Evgeni Sautin has worked for?",  # document question
        "What is the capital of France?",  # general knowledge
        "What is 5 + 3?",  # math question
        "What is Evgeni's favorite color?",  # document question (likely no answer)
    ]

    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 40)

        state = create_initial_state(question)
        response = workflow.invoke(state)

        print(f"🎯 Processing Status: {response.get('processing_status', 'N/A')}")
        print(f"🔄 Routing Decision: {response.get('routing_decision', 'N/A')}")
        print(f"💬 Answer: {response['messages'][-1].content}")


def test_conversation_memory():
    """Test conversation memory capabilities."""
    print("\n" + "=" * 60)
    print("🧠 CONVERSATION MEMORY TEST")
    print("=" * 60)

    workflow = create_enhanced_app()

    # First question.
    question1 = "What companies Evgeni Sautin has worked for?"
    print(f"\n❓ Question 1: {question1}")

    state1 = create_initial_state(question1)
    response1 = workflow.invoke(state1)

    print(f"💬 Answer 1: {response1['messages'][-1].content}")

    # Follow-up question using memory.
    question2 = "Which of those companies was he at most recently?"
    print(f"\n❓ Question 2: {question2}")

    # Create state with conversation history.
    state2 = create_initial_state(question2)
    state2["conversation_history"] = response1.get("conversation_history", [])

    response2 = workflow.invoke(state2)
    print(f"💬 Answer 2: {response2['messages'][-1].content.strip()}")
    print(f"🧠 Memory Context: {len(response2.get('conversation_history', []))} conversation entries")


def test_error_handling():
    """Test error handling and recovery."""
    print("\n" + "=" * 60)
    print("⚠️  ERROR HANDLING TEST")
    print("=" * 60)

    workflow = create_enhanced_app()

    # Test with empty/invalid question.
    question = ""
    print(f"\n❓ Question: '{question}' (empty)")

    state = create_initial_state(question)
    response = workflow.invoke(state)

    print(f"🎯 Processing Status: {response.get('processing_status', 'N/A')}")
    print(f"⚠️  Error Count: {response.get('error_count', 0)}")
    print(f"💬 Answer: {response['messages'][-1].content}")


def test_demo_app():
    """Test the demo application with all capabilities."""
    print("\n" + "=" * 60)
    print("🎨 DEMO APPLICATION - ALL CAPABILITIES")
    print("=" * 60)

    demo = create_demo_app()
    workflow = demo["workflow"]

    # Test routing chain directly.
    print("\n🔍 Testing Direct Routing Chain:")
    test_questions = [
        "What companies has Evgeni worked for?",
        "What is the capital of France?",
    ]

    for question in test_questions:
        # Handle both function and chain routing.
        if callable(demo["routing_chain"]) and not hasattr(demo["routing_chain"], "invoke"):
            # New function-based routing.
            routing_result = demo["routing_chain"](question)
        else:
            # Old chain-based routing.
            routing_result = demo["routing_chain"].invoke({"question": question})
        print(f"❓ '{question}' → 🎯 {routing_result}")

    # Test individual chains.
    print("\n🧪 Testing Individual Chains:")
    chains = demo["chains"]

    question = "What companies has Evgeni worked for?"
    print(f"\n❓ Question: {question}")

    # Test RAG chain.
    rag_result = chains["rag"].invoke({"input": question})
    print(f"📄 RAG Chain: {rag_result['answer'][:100]}...")

    # Test direct LLM chain.
    llm_result = chains["direct_llm"].invoke({"input": "What is 5 + 3?"})
    print(f"🧠 Direct LLM: {llm_result}")


def test_routing_comparison():
    """Compare different routing approaches."""
    print("\n" + "=" * 60)
    print("🔄 CLEAN ROUTING APPROACHES COMPARISON")
    print("=" * 60)

    demo = create_demo_app()

    # Import routing functions.

    from src.resume_agent.config import EMBEDDINGS
    from src.resume_agent.tools import (
        create_embedding_based_router,
        create_routing_chain,
        create_simple_embedding_router,
        create_vector_store_router,
    )

    # Create different routers.
    simple_router = create_simple_embedding_router(EMBEDDINGS)
    vector_store_router = create_vector_store_router(EMBEDDINGS, demo["vector_store"])
    basic_router = create_embedding_based_router(EMBEDDINGS)
    llm_router = create_routing_chain(demo["llm"])

    test_questions = [
        "What companies has Evgeni worked for?",
        "What is the capital of France?",
        "What is 5 + 3?",
        "Tell me about his background",
        "How does machine learning work?",
        "What projects has he done?",
    ]

    print("\n📊 Clean Routing Comparison Results:")
    print("-" * 80)
    print(f"{'Question':<40} {'Simple':<12} {'Vector':<12} {'Basic':<12} {'LLM':<12}")
    print("-" * 80)

    import time

    timing_results = {"simple": 0.0, "vector_store": 0.0, "basic": 0.0, "llm": 0.0}

    for question in test_questions:
        # Time each approach.
        start_time = time.time()
        simple_result = simple_router(question)
        timing_results["simple"] += time.time() - start_time

        start_time = time.time()
        vector_result = vector_store_router(question)
        timing_results["vector_store"] += time.time() - start_time

        start_time = time.time()
        basic_result = basic_router(question)
        timing_results["basic"] += time.time() - start_time

        start_time = time.time()
        llm_result = llm_router.invoke({"question": question}).strip().lower()
        timing_results["llm"] += time.time() - start_time

        # Normalize LLM result for display.
        llm_display = "document" if "document" in llm_result else "general"

        print(f"{question[:38]:<40} {simple_result:<12} {vector_result:<12} {basic_result:<12} {llm_display:<12}")

    print("-" * 80)
    print("\n⏱️ Performance Comparison (total time for all questions):")
    for approach, total_time in timing_results.items():
        print(f"{approach.capitalize():<15}: {total_time:.4f} seconds")

    print(f"\n🚀 Speed improvements over LLM:")
    for approach, total_time in timing_results.items():
        if approach != "llm" and total_time > 0:
            speedup = timing_results["llm"] / total_time
            print(f"  {approach.capitalize():<15}: {speedup:.1f}x faster")
    print("")


def main():
    """Main function demonstrating all capabilities."""
    print("🎯 RESUME AGENT")

    try:
        # Test original app.
        test_original_app()

        # Test enhanced app.
        test_enhanced_app()

        # Test conversation memory.
        test_conversation_memory()

        # Test error handling.
        test_error_handling()

        # Test demo app.
        test_demo_app()

        # Test routing comparison.
        test_routing_comparison()

    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        print("Make sure you have the required data files and dependencies installed.")


if __name__ == "__main__":
    main()
