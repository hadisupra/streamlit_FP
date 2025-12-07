# Agentic SQLite, Qdrant, and RAG Integration

This project includes intelligent agents powered by LangGraph that can reason about and execute queries on SQLite databases, Qdrant vector stores, and combine both through RAG (Retrieval-Augmented Generation).

## Features

### SQLite Agent 🗄️
The SQLite agent uses LangGraph's ReAct pattern to intelligently query SQLite databases. It can:

- **Explore Database Schema**: List tables and describe their structure
- **Generate SQL Queries**: Convert natural language to SQL
- **Execute Queries**: Run SQL and return formatted results
- **Sample Data**: Retrieve sample data from tables
- **Reason About Data**: Use multiple tools in sequence to answer complex questions

### Qdrant Agent 🔍
The Qdrant agent provides intelligent vector search and analytics capabilities:

- **Semantic Search**: Find similar documents using vector embeddings
- **Metadata Filtering**: Filter documents by specific fields
- **Collection Analytics**: Get statistics and aggregations
- **Smart Reasoning**: Decide when to use search vs. filtering vs. aggregation

### RAG Agent 🎯 **NEW!**
The RAG agent combines both SQLite and Qdrant for comprehensive answers:

- **Multi-Source Retrieval**: Query both structured and unstructured data
- **Cross-Reference Analysis**: Correlate database records with customer reviews
- **Intelligent Tool Selection**: Automatically choose SQL, vector, or both
- **Source Tracking**: Show which data sources contributed to answers
- **Multi-Hop Reasoning**: Execute complex queries that build on previous results

## Architecture

```
┌─────────────────────────────────────────────┐
│           Streamlit UI                      │
│  (streamlit_app.py)                         │
└─────────────────┬───────────────────────────┘
                  │
                  ├─── Agentic Mode Enabled?
                  │
        ┌─────────┼─────────┬─────────────┐
        │         │         │             │
        ▼         ▼         ▼             ▼
┌──────────┐ ┌────────┐ ┌──────────┐ ┌──────────┐
│ SQLite   │ │ Qdrant │ │   RAG    │ │   API    │
│  Agent   │ │ Agent  │ │  Agent   │ │ (fallback)│
└────┬─────┘ └────┬───┘ └────┬─────┘ └──────────┘
     │            │           │
     │            │           ├─── Uses Both Agents
     │            │           │
     │            │      ┌────┴────┐
     │            │      │         │
     ▼            ▼      ▼         ▼
┌──────────┐  ┌──────────────────────┐
│ SQLite   │  │   Qdrant Vector DB   │
│   DB     │  │   (Reviews)          │
└──────────┘  └──────────────────────┘

Tool Sets:
SQLite Agent:        Qdrant Agent:         RAG Agent:
• list_tables        • search_vectors      • query_structured_data
• describe_table     • filter_by_metadata  • search_reviews
• execute_query      • aggregate_by_field  • get_product_reviews
• get_sample_data    • get_collection_info • analyze_sentiment
                                           • cross_reference_data
```

## Installation

All required dependencies are in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `langchain==0.1.8` - Base LangChain framework
- `langchain-openai<=1.0.0` - OpenAI integration
- `langgraph>=0.0.20` - Graph-based agent orchestration
- `qdrant-client>=1.8.0` - Qdrant vector database client
- `sqlalchemy>=2.0.0` - SQLite database access

## Environment Variables

Create a `.env` file with:

```env
# Required
OPENAI_API_KEY=your-openai-api-key

# For Qdrant agent
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
QDRANT_COLLECTION=olist_reviews

# For API backend
API_URL=https://your-api-url.com
```

## Usage

### In Streamlit App

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. In the sidebar, check **"Enable Agentic Execution"**

3. Initialize the agents:
   - **SQLite Agent**: Expand "SQLite Agent Config" and click "Initialize SQLite Agent"
   - **Qdrant Agent**: Expand "Qdrant Agent Config" and click "Initialize Qdrant Agent"

4. Ask questions naturally:
   - "What are the top 10 most expensive products?"
   - "Show me reviews about late deliveries"
   - "How many orders were placed last month?"

5. View the agent's reasoning by expanding the "🔍 View Agent Reasoning" section

### Programmatic Usage

```python
from agents import create_sqlite_agent, create_qdrant_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# SQLite Agent
sqlite_agent = create_sqlite_agent(db_path="olist.db", llm=llm)
result = sqlite_agent.query("What tables are in the database?")
print(result["answer"])

# Qdrant Agent
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
qdrant_agent = create_qdrant_agent(
    collection_name="olist_reviews",
    qdrant_url="your-url",
    qdrant_api_key="your-key",
    llm=llm,
    embeddings=embeddings
)
result = qdrant_agent.query("Find negative reviews")
print(result["answer"])
```

### Testing

Run the test script to verify agent functionality:

```bash
python test_agents.py
```

## How It Works

### ReAct Pattern
Both agents use the ReAct (Reasoning + Acting) pattern:

1. **Thought**: Agent reasons about what to do
2. **Action**: Agent selects and executes a tool
3. **Observation**: Agent processes the tool's output
4. **Repeat**: Continues until the question is answered

Example SQLite agent reasoning:
```
Thought: I need to find the available tables first
Action: list_tables()
Observation: Available tables: products, orders, customers...
Thought: Now I need to see the products table schema
Action: describe_table("products")
Observation: Schema shows price, category, name columns...
Thought: I can now write a query to get top products by price
Action: execute_query("SELECT * FROM products ORDER BY price DESC LIMIT 10")
Observation: [results]
Final Answer: Here are the top 10 most expensive products...
```

### Tool Selection
Agents intelligently choose which tools to use based on the query:

**SQLite Agent Tools:**
- Schema exploration → `list_tables`, `describe_table`
- Data inspection → `get_sample_data`
- Query execution → `execute_query`

**Qdrant Agent Tools:**
- Semantic questions → `search_vectors`
- Specific field queries → `filter_by_metadata`
- Statistics/counts → `aggregate_by_field`
- General info → `get_collection_info`

## Advantages Over Direct API Calls

### 1. **Intelligent Reasoning**
   - Agents break down complex queries into steps
   - They explore the schema before querying
   - They choose the right tool for the job

### 2. **Error Recovery**
   - If a query fails, agents can reformulate
   - They validate their approach before execution

### 3. **Transparency**
   - Full reasoning trace is available
   - Users can see exactly how the agent arrived at the answer

### 4. **Flexibility**
   - Handles ambiguous or incomplete queries
   - Adapts to different database schemas

## Example Queries

### SQLite Agent
```
✅ "Show me products in the electronics category"
✅ "What's the average order value?"
✅ "List the top 5 sellers by number of orders"
✅ "How many products cost more than $100?"
```

### Qdrant Agent
```
✅ "Find reviews mentioning delivery problems"
✅ "Show me 5-star reviews about product quality"
✅ "What's the distribution of ratings?"
✅ "Get reviews from angry customers"
```

### RAG Agent (Combined)
```
✅ "What are the top 5 products and what do customers say about them?"
✅ "Analyze sentiment for products over $100"
✅ "Show me the electronics category: top products and customer feedback"
✅ "Which products have high prices but low ratings?"
✅ "Do expensive products get better reviews?"
```

## Troubleshooting

### Agent Not Initializing
- Check that `OPENAI_API_KEY` is set in `.env`
- Verify database/collection paths are correct
- Ensure all dependencies are installed

### Slow Response Times
- Agents need to reason through multiple steps
- First queries may take 10-30 seconds
- Consider caching for repeated queries

### Import Errors
```bash
pip install --upgrade langchain langchain-openai langgraph
```

## API Reference

### `SQLiteAgent(db_path, llm)`
Creates an agent for SQLite database operations.

**Parameters:**
- `db_path` (str): Path to SQLite database file
- `llm` (ChatOpenAI): Language model for reasoning

**Methods:**
- `query(question: str) -> Dict`: Execute a natural language query

### `QdrantAgent(collection_name, qdrant_url, qdrant_api_key, llm, embeddings)`
Creates an agent for Qdrant vector operations.

**Parameters:**
- `collection_name` (str): Name of Qdrant collection
- `qdrant_url` (str): Qdrant server URL
- `qdrant_api_key` (str): API key for authentication
- `llm` (ChatOpenAI): Language model for reasoning
- `embeddings` (OpenAIEmbeddings): Embedding model for vector search

**Methods:**
- `query(question: str) -> Dict`: Execute a natural language query

### `RAGAgent(sqlite_agent, qdrant_agent, llm, db_path, collection_name, ...)`
Creates a RAG agent that combines SQLite and Qdrant.

**Parameters:**
- `sqlite_agent` (SQLiteAgent, optional): Existing SQLite agent
- `qdrant_agent` (QdrantAgent, optional): Existing Qdrant agent
- `llm` (ChatOpenAI, optional): Language model for reasoning
- `db_path` (str, optional): Path to SQLite database
- `collection_name` (str, optional): Qdrant collection name
- `qdrant_url` (str, optional): Qdrant server URL
- `qdrant_api_key` (str, optional): API key for authentication
- `embeddings` (OpenAIEmbeddings, optional): Embedding model

**Methods:**
- `query(question: str, include_sources: bool = True) -> Dict`: Execute RAG query
- `multi_hop_query(questions: List[str]) -> Dict`: Execute multi-step reasoning

**See [RAG_README.md](RAG_README.md) for detailed RAG documentation.**

## Contributing

To add new tools to the agents:

1. Define the tool function with `@tool` decorator
2. Add it to the agent's `_create_tools()` method
3. Update the system message to describe when to use it

Example:
```python
@tool
def my_new_tool(param: str) -> str:
    """Description of what this tool does"""
    # Implementation
    return result
```

## License

This project is part of the streamlit_FP repository.
