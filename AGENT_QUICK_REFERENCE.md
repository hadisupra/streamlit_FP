# Quick Reference: Choosing the Right Agent

## Decision Tree

```
Question Type?
│
├─ Pure Data Query (prices, counts, stats)?
│  └─> Use SQLite Agent
│      Examples:
│      • "Top 10 products by price"
│      • "Average order value"
│      • "Products in electronics category"
│
├─ Customer Opinions/Sentiment?
│  └─> Use Qdrant Agent
│      Examples:
│      • "Reviews about delivery"
│      • "Negative feedback"
│      • "Sentiment on product quality"
│
└─ Requires BOTH Data + Reviews?
   └─> Use RAG Agent
       Examples:
       • "Top products and what customers say"
       • "Expensive products with good reviews"
       • "Category analysis with sentiment"
```

## Quick Comparison

| Scenario | Agent | Why? |
|----------|-------|------|
| "Show me all products" | SQLite | Pure database query |
| "Find negative reviews" | Qdrant | Semantic search in reviews |
| "Top 5 products and their reviews" | RAG | Needs both sources |
| "What's the price range?" | SQLite | Statistical query |
| "Sentiment about delivery" | Qdrant | Opinion/sentiment analysis |
| "Do expensive items get better ratings?" | RAG | Correlates price data with reviews |
| "List all tables" | SQLite | Schema exploration |
| "Reviews about late delivery" | Qdrant | Topic-based search |
| "Product X: specs and feedback" | RAG | Combines structured + unstructured |

## Performance Considerations

### SQLite Agent
- **Speed**: ⚡⚡⚡ Fast (direct SQL)
- **Complexity**: Low
- **Use When**: Speed matters, pure data

### Qdrant Agent
- **Speed**: ⚡⚡ Moderate (vector search)
- **Complexity**: Medium
- **Use When**: Semantic search needed

### RAG Agent
- **Speed**: ⚡ Slower (multiple retrievals)
- **Complexity**: High
- **Use When**: Comprehensive analysis needed

## Code Snippets

### SQLite Agent
```python
from agents import create_sqlite_agent

agent = create_sqlite_agent("olist.db")
result = agent.query("Top 10 products by price")
print(result["answer"])
```

### Qdrant Agent
```python
from agents import create_qdrant_agent

agent = create_qdrant_agent(
    collection_name="olist_reviews",
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)
result = agent.query("Find reviews about late delivery")
print(result["answer"])
```

### RAG Agent
```python
from agents import create_rag_agent

agent = create_rag_agent(
    db_path="olist.db",
    collection_name="olist_reviews",
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY
)
result = agent.query(
    "What are the top 5 products and what do customers say?",
    include_sources=True
)
print(result["answer"])
print(f"Sources: {[s['tool'] for s in result['sources']]}")
```

## Common Patterns

### Pattern 1: Explore then Query (SQLite)
```python
# First understand the schema
result1 = sqlite_agent.query("What tables are available?")
# Then query specific data
result2 = sqlite_agent.query("Show products from the electronics table")
```

### Pattern 2: Search then Filter (Qdrant)
```python
# Broad search
result1 = qdrant_agent.query("Find reviews about quality")
# Then filter by rating
result2 = qdrant_agent.query("Show 5-star reviews about quality")
```

### Pattern 3: Multi-Hop Analysis (RAG)
```python
result = rag_agent.multi_hop_query([
    "What are the top 3 categories by product count?",
    "For each category, what's the average rating?",
    "Which category should we focus on?"
])
print(result["synthesis"])
```

## Tips & Tricks

### 💡 Tip 1: Be Specific
❌ "Show me stuff"
✅ "Show me products in the electronics category with prices over $100"

### 💡 Tip 2: Use RAG for "And" Questions
If your question has "and" connecting different data types, use RAG:
- "Products **and** reviews"
- "Prices **and** sentiment"
- "Stats **and** opinions"

### 💡 Tip 3: Reuse Agents
```python
# Initialize once
rag_agent = create_rag_agent(...)

# Use many times
for question in questions:
    result = rag_agent.query(question)
```

### 💡 Tip 4: Check Sources in RAG
```python
result = rag_agent.query(question, include_sources=True)
if "query_structured_data" in [s["tool"] for s in result["sources"]]:
    print("Answer includes database facts")
if "search_reviews" in [s["tool"] for s in result["sources"]]:
    print("Answer includes customer opinions")
```

### 💡 Tip 5: Streamlit UI Shortcuts
1. Enable "Agentic Mode" checkbox
2. For SQL-only questions → Init SQLite Agent only
3. For reviews-only → Init Qdrant Agent only
4. For combined → Check "Use RAG Agent" and init RAG
5. View reasoning in expandable sections

## Error Handling

### "Agent not initialized"
**Fix:** Initialize the agent in Streamlit sidebar or code

### "Both agents required"
**Fix:** For RAG, ensure both db_path and collection_name are set

### Timeout
**Fix:** 
- Simplify your question
- Use individual agents for simple queries
- Increase timeout in request settings

### Empty results
**Fix:**
- Check if data exists (use SQLite agent to verify)
- Rephrase your question
- Check collection name/database path

## When NOT to Use Agents

### Use Direct API Instead:
- Simple health checks
- Fixed queries that don't change
- Batch operations (use direct SQL/Qdrant client)
- Real-time dashboards (too slow)

### Use Agents When:
- Natural language queries
- Exploratory analysis
- User-facing chat interfaces
- Complex multi-step reasoning
- Schema is unknown or complex

## Summary Table

| Agent Type | Data Sources | Response Time | Complexity | Best For |
|------------|--------------|---------------|------------|----------|
| SQLite | SQL DB only | Fast (1-3s) | Low | Data queries |
| Qdrant | Vectors only | Medium (2-5s) | Medium | Semantic search |
| RAG | Both | Slow (5-15s) | High | Comprehensive answers |

## Need More Help?

- **Detailed RAG docs**: See [RAG_README.md](RAG_README.md)
- **Full agent docs**: See [AGENTS_README.md](AGENTS_README.md)
- **Test agents**: Run `python test_agents.py`
- **Examples**: Check `streamlit_app.py` integration

---

**TL;DR:**
- Data only? → SQLite Agent
- Reviews only? → Qdrant Agent
- Both? → RAG Agent
