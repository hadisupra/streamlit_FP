# RAG (Retrieval-Augmented Generation) Agent

The RAG Agent is an intelligent system that combines SQLite and Qdrant agents to provide comprehensive answers by retrieving information from both structured and unstructured data sources.

## 🎯 What is RAG?

**RAG (Retrieval-Augmented Generation)** is a technique that enhances LLM responses by:
1. **Retrieving** relevant information from multiple data sources
2. **Augmenting** the LLM's context with retrieved data
3. **Generating** comprehensive, fact-based answers

Unlike simple agents that query a single source, the RAG Agent intelligently decides which data sources to use and combines their results.

## 🌟 Key Features

### Multi-Source Intelligence
- **Structured Data (SQLite)**: Products, orders, customers, transactions
- **Unstructured Data (Qdrant)**: Customer reviews, feedback, sentiment
- **Cross-Reference**: Combines both sources for holistic insights

### Intelligent Tool Selection
The RAG Agent automatically chooses the right tools:
- `query_structured_data` - For product info, prices, statistics
- `search_reviews` - For customer opinions and sentiment
- `get_product_reviews` - For product-specific feedback
- `analyze_sentiment` - For sentiment analysis on topics
- `cross_reference_data` - For combining SQL and vector results

### Source Tracking
- Tracks which tools were used
- Shows what data sources contributed to the answer
- Provides transparency in the reasoning process

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│         RAG Agent                   │
│  (LangGraph ReAct)                  │
└──────────┬──────────────────────────┘
           │
           ├─── Analyzes Question
           │    Determines required info
           │
           ▼
    ┌──────────────┐
    │ Tool Router  │
    └──────┬───────┘
           │
      ┌────┼────┐
      │    │    │
      ▼    ▼    ▼
   ┌────┐ ┌────┐ ┌─────┐
   │SQL │ │Vec │ │Both │
   │Tool│ │Tool│ │Tools│
   └─┬──┘ └─┬──┘ └──┬──┘
     │      │       │
     ▼      ▼       ▼
┌─────────┐ ┌──────────┐
│ SQLite  │ │ Qdrant   │
│   DB    │ │  Vector  │
└─────────┘ └──────────┘
     │          │
     └────┬─────┘
          ▼
    ┌──────────────┐
    │  Synthesize  │
    │    Answer    │
    └──────────────┘
          │
          ▼
   Comprehensive Response
```

## 📦 Installation

Already included in `requirements.txt`. The RAG agent uses:
- `langgraph>=0.0.20` - Agent orchestration
- `langchain-openai<=1.0.0` - LLM integration
- `qdrant-client>=1.8.0` - Vector search
- `sqlalchemy>=2.0.0` - SQL queries

## 🚀 Quick Start

### In Streamlit App

1. **Enable Agentic Mode** in the sidebar
2. **Check "🎯 Use RAG Agent (Combined)"**
3. **Initialize RAG Agent** with your database and collection
4. **Ask comprehensive questions** that span multiple data sources

Example questions:
```
❓ "What are the top 5 products and what do customers say about them?"
❓ "Analyze sentiment for products over $100"
❓ "Show me popular products with negative reviews"
❓ "Which categories have the best customer satisfaction?"
```

### Programmatic Usage

```python
from agents import create_rag_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

# Initialize
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

rag_agent = create_rag_agent(
    db_path="olist.db",
    collection_name="olist_reviews",
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    llm=llm,
    embeddings=embeddings
)

# Query
result = rag_agent.query(
    "What are the most expensive products and what do customers think about them?",
    include_sources=True
)

print(result["answer"])
print(f"\nSources used: {[s['tool'] for s in result['sources']]}")
```

### Using Existing Agents

```python
from agents import create_sqlite_agent, create_qdrant_agent, create_rag_agent

# Create individual agents
sqlite_agent = create_sqlite_agent("olist.db")
qdrant_agent = create_qdrant_agent("olist_reviews")

# Create RAG agent using existing agents (more efficient)
rag_agent = create_rag_agent(
    sqlite_agent=sqlite_agent,
    qdrant_agent=qdrant_agent,
    llm=llm
)
```

## 🔧 RAG Agent Methods

### `query(question, include_sources=True)`
Execute a single RAG query

**Parameters:**
- `question` (str): Natural language question
- `include_sources` (bool): Whether to track which tools were used

**Returns:**
```python
{
    "question": "original question",
    "answer": "comprehensive answer",
    "sources": [
        {"tool": "query_structured_data", "input": {...}},
        {"tool": "search_reviews", "input": {...}}
    ],
    "full_conversation": [...]  # Full agent reasoning
}
```

### `multi_hop_query(questions)`
Execute multiple related queries in sequence

**Parameters:**
- `questions` (List[str]): List of related questions

**Returns:**
```python
{
    "questions": [...],
    "individual_answers": [...],
    "synthesis": "comprehensive synthesis of all findings",
    "full_conversation": [...]
}
```

**Example:**
```python
result = rag_agent.multi_hop_query([
    "What are the top 3 products by price?",
    "What do customers say about these products?",
    "Based on reviews, which one should I recommend?"
])

print(result["synthesis"])
```

## 🎓 Example Use Cases

### 1. Product Analysis with Customer Feedback
```python
question = "Analyze the electronics category: show top products and customer sentiment"

result = rag_agent.query(question)
# Agent will:
# 1. Query SQLite for top electronics products
# 2. Search Qdrant for reviews in that category
# 3. Analyze sentiment
# 4. Synthesize comprehensive answer
```

### 2. Price-Sentiment Correlation
```python
question = "Do expensive products have better reviews?"

result = rag_agent.query(question)
# Agent will:
# 1. Get price distribution from SQLite
# 2. Get reviews for different price ranges from Qdrant
# 3. Analyze sentiment by price tier
# 4. Provide insights on correlation
```

### 3. Specific Product Deep Dive
```python
question = "Show me all info about product XYZ including customer opinions"

result = rag_agent.query(question)
# Agent will:
# 1. Get product details from SQLite
# 2. Get product-specific reviews from Qdrant
# 3. Summarize both structured and unstructured data
```

### 4. Multi-Hop Reasoning
```python
result = rag_agent.multi_hop_query([
    "What categories have the most products?",
    "For the top 3 categories, what's the average rating?",
    "Which category offers the best value for money?"
])
# Each question builds on previous answers
```

## 🛠️ Available Tools

### For Structured Data (SQL)

#### `query_structured_data(question)`
Query products, orders, customers, prices, categories, etc.

**When to use:** 
- Product information requests
- Price queries
- Order statistics
- Category analysis

**Example inputs:**
- "Get top 10 products by price"
- "How many orders in Q3?"
- "Average product price per category"

### For Unstructured Data (Reviews)

#### `search_reviews(query, k=5)`
Semantic search across customer reviews

**When to use:**
- Finding reviews by topic
- Sentiment queries
- Customer opinion research

**Example inputs:**
- "late delivery"
- "product quality issues"
- "positive feedback"

#### `get_product_reviews(product_id, limit=5)`
Get reviews for a specific product

**When to use:**
- Product-specific feedback
- Per-product sentiment

#### `analyze_sentiment(topic)`
Analyze sentiment on a specific topic

**When to use:**
- Topic-based sentiment analysis
- Categorizing feedback

### For Combined Analysis

#### `cross_reference_data(product_query, review_query)`
Combine SQL and vector data

**When to use:**
- Correlating structured and unstructured data
- Comprehensive product analysis
- Multi-source insights

## 📊 Output Format

### Basic Query Response
```
{
    "question": "What do customers think about expensive products?",
    "answer": "Based on analysis of products over $100 and their reviews:
    
    1. Product Quality: Most expensive products (avg $150+) receive 4.5+ star ratings
    2. Common Praise: Customers frequently mention 'excellent quality', 'worth the price'
    3. Main Complaints: Some mention 'slow delivery' but rarely complain about product itself
    4. Recommendation: Expensive products generally have higher satisfaction rates
    
    This analysis combined data from 50 products and 500+ reviews.",
    
    "sources": [
        {"tool": "query_structured_data", "input": {"question": "products over $100"}},
        {"tool": "search_reviews", "input": {"query": "expensive products", "k": 10}}
    ]
}
```

## 🎯 Best Practices

### 1. **Ask Comprehensive Questions**
❌ "Show me products"
✅ "Show me top products by price and what customers say about them"

### 2. **Be Specific About Context**
❌ "What do people think?"
✅ "What do customers think about delivery speed for electronics?"

### 3. **Use Multi-Hop for Complex Analysis**
For questions requiring multiple steps, use `multi_hop_query()` instead of a single query.

### 4. **Review Sources**
Always check the `sources` field to understand which data contributed to the answer.

### 5. **Reuse Agents**
If you're making multiple queries, initialize agents once and reuse them:
```python
# Initialize once
rag_agent = create_rag_agent(...)

# Query multiple times
for question in questions:
    result = rag_agent.query(question)
```

## 🔍 Debugging

### View Agent Reasoning
```python
result = rag_agent.query("your question")

for msg in result["full_conversation"]:
    print(f"{msg.__class__.__name__}: {msg.content}")
```

### Check Which Tools Were Used
```python
result = rag_agent.query("your question", include_sources=True)

for source in result["sources"]:
    print(f"Used: {source['tool']} with input: {source['input']}")
```

## ⚡ Performance Tips

### 1. **Initialize Once**
Agent initialization is expensive. Create agents once and reuse.

### 2. **Limit Vector Results**
When searching reviews, use reasonable `k` values (5-10).

### 3. **Cache Common Queries**
For repeated questions, cache results in your application.

### 4. **Use Specific Queries**
More specific questions lead to more efficient tool usage.

## 🆚 RAG vs Individual Agents

| Feature | SQLite Agent | Qdrant Agent | RAG Agent |
|---------|--------------|--------------|-----------|
| Data Source | SQL only | Vectors only | Both |
| Use Case | Structured queries | Semantic search | Comprehensive analysis |
| Tool Count | 4 | 4 | 5+ |
| Complexity | Low | Low | High |
| Answer Quality | Factual | Contextual | Holistic |

**When to use each:**
- **SQLite Agent**: Pure data queries (prices, counts, etc.)
- **Qdrant Agent**: Customer opinion research
- **RAG Agent**: Questions requiring both sources

## 🚨 Common Issues

### "RAG agent not initialized"
**Solution:** Initialize in the sidebar before querying

### "Both agents required"
**Solution:** Ensure both SQLite and Qdrant URLs/paths are configured

### Slow responses
**Solution:** 
- RAG agents perform multiple retrieval steps
- First query is always slower (model loading)
- Consider using individual agents for simple queries

### Out of context errors
**Solution:** 
- Reduce `k` parameter in vector searches
- Simplify questions
- Use multi-hop for very complex queries

## 📚 Additional Resources

- **LangGraph Documentation**: https://langchain-ai.github.io/langgraph/
- **ReAct Paper**: https://arxiv.org/abs/2210.03629
- **RAG Survey**: https://arxiv.org/abs/2312.10997

## 🎓 Advanced Usage

### Custom System Prompt
```python
from langchain_core.messages import SystemMessage

# Modify agent's system message for specific domains
custom_prompt = """You are a retail analytics expert.
Focus on business insights and actionable recommendations..."""

# Access after initialization
rag_agent.agent.system_message = SystemMessage(content=custom_prompt)
```

### Async Queries
```python
import asyncio

async def query_multiple():
    questions = [...]
    results = await asyncio.gather(*[
        asyncio.to_thread(rag_agent.query, q)
        for q in questions
    ])
    return results
```

## 📝 Summary

The RAG Agent is your go-to solution for questions that require:
- ✅ Both structured and unstructured data
- ✅ Cross-referencing multiple sources  
- ✅ Comprehensive, fact-based answers
- ✅ Source transparency

It intelligently orchestrates multiple tools to provide holistic insights that single-source agents cannot achieve.
