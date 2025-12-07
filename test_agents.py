"""
Example usage of SQLite, Qdrant, and RAG agents
Run this script to test the agentic functionality locally
"""

import os
from dotenv import load_dotenv
from agents import create_sqlite_agent, create_qdrant_agent, create_rag_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

def test_sqlite_agent():
    """Test SQLite agent functionality"""
    print("\n" + "="*60)
    print("Testing SQLite Agent")
    print("="*60 + "\n")
    
    # Initialize agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    agent = create_sqlite_agent(db_path="olist_small.db", llm=llm)
    
    # Test queries
    questions = [
        "What tables are available in the database?",
        "Show me the top 5 most expensive products",
        "How many orders were placed in total?",
        "What is the average product price by category?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        result = agent.query(question)
        print(f"A: {result['answer']}\n")


def test_qdrant_agent():
    """Test Qdrant agent functionality"""
    print("\n" + "="*60)
    print("Testing Qdrant Agent")
    print("="*60 + "\n")
    
    # Initialize agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "olist_reviews")
    
    agent = create_qdrant_agent(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        llm=llm,
        embeddings=embeddings
    )
    
    # Test queries
    questions = [
        "What information is available in the collection?",
        "Find reviews about late deliveries",
        "Show me positive reviews with high ratings",
        "What are the most common review ratings?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        result = agent.query(question)
        print(f"A: {result['answer']}\n")


def test_rag_agent():
    """Test RAG agent functionality"""
    print("\n" + "="*60)
    print("Testing RAG Agent (Combined)")
    print("="*60 + "\n")
    
    # Initialize agent
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("QDRANT_COLLECTION", "olist_reviews")
    
    agent = create_rag_agent(
        db_path="olist_small.db",
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        llm=llm,
        embeddings=embeddings
    )
    
    # Test queries that require both structured and unstructured data
    questions = [
        "What are the top 3 most expensive products and what do customers say about them?",
        "Analyze customer sentiment for products in the electronics category",
        "Show me products with prices over $100 and their customer reviews",
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 60)
        result = agent.query(question, include_sources=True)
        print(f"A: {result['answer']}")
        if result.get('sources'):
            print(f"\nSources used: {', '.join([s['tool'] for s in result['sources']])}")
        print()


if __name__ == "__main__":
    # Check if API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file")
        exit(1)
    
    # Test SQLite agent
    try:
        test_sqlite_agent()
    except Exception as e:
        print(f"Error testing SQLite agent: {e}")
    
    # Test Qdrant agent
    try:
        if not os.getenv("QDRANT_URL"):
            print("\nSkipping Qdrant agent test - QDRANT_URL not set")
        else:
            test_qdrant_agent()
    except Exception as e:
        print(f"Error testing Qdrant agent: {e}")
    
    # Test RAG agent
    try:
        if not os.getenv("QDRANT_URL"):
            print("\nSkipping RAG agent test - QDRANT_URL not set")
        elif not os.path.exists("olist_small.db"):
            print("\nSkipping RAG agent test - olist_small.db not found")
        else:
            test_rag_agent()
    except Exception as e:
        print(f"Error testing RAG agent: {e}")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60)
