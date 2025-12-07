# 🚀 Getting Started with RAG Agentic System

## Quick Start (5 minutes)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure Environment
Create a `.env` file:
```env
OPENAI_API_KEY=sk-your-key-here
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=olist_reviews
```

### Step 3: Run Streamlit App
```bash
streamlit run streamlit_app.py
```

### Step 4: Enable RAG Agent
1. In sidebar, check ✅ **"Enable Agentic Execution"**
2. Check ✅ **"🎯 Use RAG Agent (Combined)"**
3. Click **"Initialize RAG Agent"**
4. Wait for ✅ success message

### Step 5: Ask Questions!
```
Try these:
• "What are the top 5 products and what do customers say?"
• "Do expensive products get better reviews?"
• "Analyze electronics category with sentiment"
```

---

## What You Get

### 3 Agent Types

#### 1️⃣ SQLite Agent
- Queries structured database
- Fast (1-3 seconds)
- Use for: product data, prices, statistics

#### 2️⃣ Qdrant Agent
- Searches review vectors
- Medium speed (2-5 seconds)
- Use for: customer opinions, sentiment

#### 3️⃣ RAG Agent (⭐ NEW!)
- Combines both sources
- Slower but comprehensive (5-15 seconds)
- Use for: questions requiring both data types

### What Makes RAG Special?

**Traditional Approach:**
```
Q: "Show me expensive products"
A: [List of products] ← Only data, no context
```

**RAG Approach:**
```
Q: "Show me expensive products and what customers think"
A: Top products:
   1. Product A ($299)
      → Reviews: "Excellent quality", "Worth it"
   2. Product B ($250)
      → Reviews: "Good but slow delivery"
   ...
   ← Data + Context = Complete answer!
```

---

## Usage Patterns

### Pattern 1: Simple Question → Individual Agent
```
Q: "What's the average product price?"
→ Use SQLite Agent (fast, data-only)
```

### Pattern 2: Opinion Question → Vector Agent
```
Q: "Find negative reviews"
→ Use Qdrant Agent (semantic search)
```

### Pattern 3: Combined Question → RAG Agent
```
Q: "Show products and their reviews"
→ Use RAG Agent (comprehensive)
```

---

## Interactive Demo

### Try It Yourself

```bash
# Run demo script
python demo_rag.py
```

This shows 5 demonstrations:
1. Basic RAG query
2. Sentiment analysis
3. Category analysis
4. Multi-hop reasoning
5. Agent comparison

---

## Code Examples

### Example 1: Basic RAG Query
```python
from agents import create_rag_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

rag = create_rag_agent(
    db_path="olist.db",
    collection_name="olist_reviews",
    qdrant_url=os.getenv("QDRANT_URL"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    llm=llm,
    embeddings=embeddings
)

result = rag.query("What are top products and reviews?")
print(result["answer"])
```

### Example 2: Check Sources
```python
result = rag.query(
    "Do expensive products get better reviews?",
    include_sources=True
)

print(result["answer"])
print("\nSources used:")
for source in result["sources"]:
    print(f"  - {source['tool']}")
```

### Example 3: Multi-Hop Reasoning
```python
result = rag.multi_hop_query([
    "What are top 3 categories?",
    "What's the average rating for each?",
    "Which category should we focus on?"
])

print(result["synthesis"])
```

---

## Architecture Diagram

```
Your Question
     │
     ▼
┌──────────────┐
│  RAG Agent   │ ← Smart router
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
   ▼       ▼
[SQL]   [Vector] ← Data sources
   │       │
   └───┬───┘
       ▼
  Answer + Sources
```

---

## Choosing the Right Agent

### Use SQLite Agent when:
- ✅ Question about products, prices, orders
- ✅ Need statistical data
- ✅ Speed is critical
- ✅ Example: "Average product price?"

### Use Qdrant Agent when:
- ✅ Question about opinions, reviews
- ✅ Sentiment analysis needed
- ✅ Semantic search required
- ✅ Example: "Find negative feedback"

### Use RAG Agent when:
- ✅ Question needs both data types
- ✅ "What" + "What do customers think"
- ✅ Cross-reference needed
- ✅ Example: "Top products and reviews?"

---

## Common Questions

### Q: Do I need a database?
**A:** Yes, you need either:
- SQLite database (`olist.db`), OR
- Qdrant collection, OR
- Both (for full RAG)

### Q: How much does it cost?
**A:** Costs are from OpenAI API:
- Embedding: ~$0.0001 per query
- GPT-4o-mini: ~$0.01 per query
- Total: ~$0.02-0.05 per RAG query

### Q: Can I use local LLMs?
**A:** Yes! Replace `ChatOpenAI` with any LangChain-compatible LLM:
```python
from langchain_community.llms import Ollama
llm = Ollama(model="llama2")
```

### Q: Is it production-ready?
**A:** Yes! Features include:
- ✅ Error handling
- ✅ Source tracking
- ✅ Transparent reasoning
- ✅ Modular design

---

## Performance Tips

### 1. Cache Agents
```python
# Bad: Initialize every time
for q in questions:
    agent = create_rag_agent(...)  # Slow!
    agent.query(q)

# Good: Initialize once
agent = create_rag_agent(...)
for q in questions:
    agent.query(q)  # Fast!
```

### 2. Use Specific Queries
```python
# Vague (slow)
"Tell me about stuff"

# Specific (fast)
"Show top 5 products over $100 with reviews"
```

### 3. Choose Right Agent
```python
# Wrong (overkill)
rag_agent.query("What is 2+2?")

# Right (appropriate)
sqlite_agent.query("Average product price")
```

---

## Troubleshooting

### Issue: "Agent not initialized"
**Fix:**
```python
# Ensure environment variables are set
import os
print(os.getenv("OPENAI_API_KEY"))  # Should not be None
```

### Issue: Slow responses
**Fix:**
- First query is always slower (model loading)
- Reduce `k` parameter in searches
- Use individual agents for simple queries

### Issue: Import errors
**Fix:**
```bash
pip install --upgrade langchain langchain-openai langgraph qdrant-client
```

### Issue: "Collection not found"
**Fix:**
- Verify collection name in Qdrant
- Check QDRANT_URL is correct
- Ensure API key has permissions

---

## Next Steps

### Learn More
1. **Quick Reference** → `AGENT_QUICK_REFERENCE.md`
2. **Full RAG Docs** → `RAG_README.md`
3. **All Agents** → `AGENTS_README.md`

### Try These
1. Run tests: `python test_agents.py`
2. Run demos: `python demo_rag.py`
3. Experiment in Streamlit UI

### Customize
1. Add new tools to RAG agent
2. Modify system prompts
3. Integrate with your data

---

## Example Workflow

### Scenario: E-commerce Product Analysis

```python
# Initialize once
rag = create_rag_agent(...)

# Analysis workflow
q1 = rag.query("What are our top 10 products by revenue?")
# → Uses SQLite to get product data

q2 = rag.query("What do customers say about these products?")
# → Uses Qdrant to search reviews

q3 = rag.query("Which products have high sales but low ratings?")
# → Uses both sources to cross-reference

# Multi-hop for recommendations
rec = rag.multi_hop_query([
    "What categories are most profitable?",
    "Which have best customer satisfaction?",
    "What should we stock more of?"
])
print(rec["synthesis"])
```

---

## Resources

### Documentation
- `RAG_IMPLEMENTATION_SUMMARY.md` - This file
- `RAG_README.md` - Deep dive into RAG
- `AGENT_QUICK_REFERENCE.md` - Quick decision guide
- `AGENTS_README.md` - All agents overview

### Code
- `agents.py` - Agent implementations
- `streamlit_app.py` - UI integration
- `test_agents.py` - Unit tests
- `demo_rag.py` - Interactive demos

### External
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [Qdrant Docs](https://qdrant.tech/documentation/)
- [OpenAI API](https://platform.openai.com/docs)

---

## Summary

You now have a complete RAG agentic system that:

✅ **Combines multiple data sources** (SQL + Vector)  
✅ **Intelligently routes queries** (chooses right tools)  
✅ **Provides comprehensive answers** (holistic insights)  
✅ **Tracks sources** (transparent reasoning)  
✅ **Supports multi-hop reasoning** (complex analysis)  
✅ **Integrates with Streamlit** (user-friendly UI)  

**Start asking questions and see the magic! 🎉**

---

## Support

Having issues? Check:
1. Environment variables are set (`.env`)
2. Dependencies installed (`pip install -r requirements.txt`)
3. Database/collection exists and accessible
4. API keys are valid

Still stuck? Review the error message and check relevant docs above.

**Happy querying! 🚀**
