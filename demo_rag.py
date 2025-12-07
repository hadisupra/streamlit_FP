"""
Demo script showing RAG agent capabilities
Run this to see a live demonstration of the RAG agent in action
"""

import os
from dotenv import load_dotenv
from agents import create_rag_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def print_result(result, show_sources=True):
    """Print a formatted query result"""
    print(f"📝 ANSWER:\n{result['answer']}\n")
    
    if show_sources and result.get('sources'):
        print("📚 SOURCES USED:")
        for i, source in enumerate(result['sources'], 1):
            print(f"   {i}. {source['tool']}")
        print()


def demo_basic_rag():
    """Demonstrate basic RAG functionality"""
    print_section("DEMO 1: Basic RAG Query")
    
    print("Question: 'What are the top 3 most expensive products and what do customers say about them?'")
    print("Expected: RAG agent will query SQL for products, then search reviews\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    agent = create_rag_agent(
        db_path="olist.db",
        collection_name="olist_reviews",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        llm=llm,
        embeddings=embeddings
    )
    
    result = agent.query(
        "What are the top 3 most expensive products and what do customers say about them?",
        include_sources=True
    )
    
    print_result(result)


def demo_sentiment_analysis():
    """Demonstrate sentiment analysis across data sources"""
    print_section("DEMO 2: Sentiment Analysis with Data Correlation")
    
    print("Question: 'Do expensive products (over $100) get better customer reviews?'")
    print("Expected: RAG agent will combine price data with review sentiment\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    agent = create_rag_agent(
        db_path="olist_small.db",
        collection_name="olist_reviews",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        llm=llm,
        embeddings=embeddings
    )
    
    result = agent.query(
        "Do expensive products (over $100) get better customer reviews?",
        include_sources=True
    )
    
    print_result(result)


def demo_category_analysis():
    """Demonstrate category-level analysis"""
    print_section("DEMO 3: Category Analysis")
    
    print("Question: 'Analyze the electronics category: show top products and customer sentiment'")
    print("Expected: RAG agent will filter products by category and analyze reviews\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    agent = create_rag_agent(
        db_path="olist_small.db",
        collection_name="olist_reviews",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        llm=llm,
        embeddings=embeddings
    )
    
    result = agent.query(
        "Analyze the electronics category: show top products and customer sentiment",
        include_sources=True
    )
    
    print_result(result)


def demo_multi_hop():
    """Demonstrate multi-hop reasoning"""
    print_section("DEMO 4: Multi-Hop Reasoning")
    
    print("Questions:")
    print("1. What are the top 3 product categories by number of products?")
    print("2. For each of these categories, what's the average customer rating?")
    print("3. Which category should we recommend to new sellers?")
    print("\nExpected: Agent will build on each previous answer\n")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    agent = create_rag_agent(
        db_path="olist.db",
        collection_name="olist_reviews",
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        llm=llm,
        embeddings=embeddings
    )
    
    result = agent.multi_hop_query([
        "What are the top 3 product categories by number of products?",
        "For each of these categories, what's the average customer rating?",
        "Based on this data, which category should we recommend to new sellers?"
    ])
    
    print("📝 INDIVIDUAL ANSWERS:")
    for i, answer in enumerate(result['individual_answers'], 1):
        print(f"\n{i}. {answer}")
    
    print("\n" + "="*70)
    print("🎯 FINAL SYNTHESIS:")
    print("="*70)
    print(f"\n{result['synthesis']}\n")


def demo_comparison():
    """Demonstrate the difference between individual and RAG agents"""
    print_section("DEMO 5: Agent Comparison")
    
    question = "Show me product ABC123 details and customer feedback"
    
    print(f"Question: '{question}'\n")
    print("Comparing different approaches...\n")
    
    # This would require individual agents to be created and compared
    print("❌ SQLite Agent alone: Can show product details but NO reviews")
    print("❌ Qdrant Agent alone: Can show reviews but NO product details")
    print("✅ RAG Agent: Shows BOTH product details AND customer reviews\n")
    print("Conclusion: RAG agent provides comprehensive, holistic answers!")


def main():
    """Run all demos"""
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ ERROR: OPENAI_API_KEY not found")
        print("Please set it in your .env file")
        return
    
    if not os.getenv("QDRANT_URL"):
        print("⚠️  WARNING: QDRANT_URL not set")
        print("Some demos will be skipped")
    
    if not os.path.exists("olist_small.db"):
        print("⚠️  WARNING: olist_small.db not found")
        print("Some demos will be skipped")
    
    print("\n" + "="*70)
    print("  RAG AGENT DEMONSTRATION")
    print("  Showcasing Retrieval-Augmented Generation Capabilities")
    print("="*70)
    
    # Run demos
    demos = [
        ("Basic RAG Query", demo_basic_rag),
        ("Sentiment Analysis", demo_sentiment_analysis),
        ("Category Analysis", demo_category_analysis),
        ("Multi-Hop Reasoning", demo_multi_hop),
        ("Agent Comparison", demo_comparison),
    ]
    
    for i, (name, demo_func) in enumerate(demos, 1):
        try:
            if i == 5:  # Comparison demo doesn't need actual queries
                demo_func()
            else:
                if os.getenv("QDRANT_URL") and os.path.exists("olist_small.db"):
                    demo_func()
                else:
                    print_section(f"DEMO {i}: {name}")
                    print(f"⏭️  Skipping (missing requirements)")
        except Exception as e:
            print(f"\n❌ Error in demo: {e}\n")
        
        if i < len(demos):
            input("\n⏸️  Press Enter to continue to next demo...")
    
    # Final summary
    print_section("DEMO COMPLETE!")
    print("Key Takeaways:")
    print("✅ RAG agents combine multiple data sources intelligently")
    print("✅ They automatically choose the right tools for each question")
    print("✅ Multi-hop reasoning enables complex analysis")
    print("✅ Source tracking provides transparency")
    print("✅ More comprehensive than individual agents")
    print("\nFor more info, see RAG_README.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
