"""
Streamlit App for LLM Agent
Connects to FastAPI backend (local or GCP Cloud Run)
"""

import os
import json
from datetime import datetime

import requests
import streamlit as st
from agents import create_sqlite_agent, create_qdrant_agent, create_rag_agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Configuration - Set your API base URL (no trailing slash)
DEFAULT_API_URL = "https://llm-agent-api-447949002484.us-central1.run.app"
API_URL = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")

# Page config
st.set_page_config(
    page_title="Olist LLM Agent Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "use_agents" not in st.session_state:
    st.session_state.use_agents = False
if "sqlite_agent" not in st.session_state:
    st.session_state.sqlite_agent = None
if "qdrant_agent" not in st.session_state:
    st.session_state.qdrant_agent = None
if "rag_agent" not in st.session_state:
    st.session_state.rag_agent = None
if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    api_url_input = st.text_input(
        "API URL",
        value=API_URL,
        help="Your FastAPI backend URL (e.g., http://localhost:8080 or Cloud Run URL)",
    )
    if api_url_input:
        API_URL = api_url_input.rstrip("/")

    if st.button("🔌 Test Connection"):
        try:
            response = requests.get(f"{API_URL}/health", timeout=5)
            if response.status_code == 200:
                st.success("✅ Connected to API!")
                st.json(response.json())
            else:
                st.error(f"❌ API returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Connection failed: {str(e)}")

    st.divider()

    st.subheader("📊 Session Info")
    st.text(f"Session ID: {st.session_state.session_id}")
    st.text(f"Messages: {len(st.session_state.messages)}")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    
    st.subheader("🤖 Agentic Mode")
    use_agents = st.checkbox(
        "Enable Agentic Execution",
        value=st.session_state.use_agents,
        help="Use LangGraph agents for intelligent query execution"
    )
    st.session_state.use_agents = use_agents
    
    if use_agents:
        st.info("🧠 Agents will reason about queries and use tools dynamically")
        
        # SQLite agent configuration
        with st.expander("SQLite Agent Config"):
            db_path = st.text_input("Database Path", value="olist.db")
            if st.button("Initialize SQLite Agent"):
                try:
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    if not openai_api_key:
                        st.error("OPENAI_API_KEY not found in environment")
                    else:
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                        st.session_state.sqlite_agent = create_sqlite_agent(db_path, llm)
                        st.success("✅ SQLite Agent initialized!")
                except Exception as e:
                    st.error(f"Error initializing SQLite agent: {e}")
        
        # Qdrant agent configuration
        with st.expander("Qdrant Agent Config"):
            collection_name = st.text_input("Collection Name", value="olist_reviews")
            if st.button("Initialize Qdrant Agent"):
                try:
                    openai_api_key = os.getenv("OPENAI_API_KEY")
                    qdrant_url = os.getenv("QDRANT_URL")
                    qdrant_api_key = os.getenv("QDRANT_API_KEY")
                    
                    if not openai_api_key:
                        st.error("OPENAI_API_KEY not found in environment")
                    elif not qdrant_url:
                        st.error("QDRANT_URL not found in environment")
                    else:
                        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                        st.session_state.qdrant_agent = create_qdrant_agent(
                            collection_name, qdrant_url, qdrant_api_key, llm, embeddings
                        )
                        st.success("✅ Qdrant Agent initialized!")
                except Exception as e:
                    st.error(f"Error initializing Qdrant agent: {e}")
        
        # RAG agent configuration
        st.divider()
        use_rag = st.checkbox(
            "🎯 Use RAG Agent (Combined)",
            value=st.session_state.use_rag,
            help="Use RAG agent that combines SQL and vector search intelligently"
        )
        st.session_state.use_rag = use_rag
        
        if use_rag:
            st.info("🔄 RAG agent combines both SQLite and Qdrant for comprehensive answers")
            with st.expander("RAG Agent Config"):
                rag_db_path = st.text_input("DB Path (RAG)", value="olist.db", key="rag_db")
                rag_collection = st.text_input("Collection (RAG)", value="olist_reviews", key="rag_coll")
                
                if st.button("Initialize RAG Agent"):
                    try:
                        openai_api_key = os.getenv("OPENAI_API_KEY")
                        qdrant_url = os.getenv("QDRANT_URL")
                        qdrant_api_key = os.getenv("QDRANT_API_KEY")
                        
                        if not openai_api_key:
                            st.error("OPENAI_API_KEY not found in environment")
                        else:
                            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                            
                            # Use existing agents if available
                            st.session_state.rag_agent = create_rag_agent(
                                db_path=rag_db_path if os.path.exists(rag_db_path) else None,
                                collection_name=rag_collection,
                                qdrant_url=qdrant_url,
                                qdrant_api_key=qdrant_api_key,
                                llm=llm,
                                embeddings=embeddings,
                                sqlite_agent=st.session_state.sqlite_agent,
                                qdrant_agent=st.session_state.qdrant_agent
                            )
                            st.success("✅ RAG Agent initialized!")
                            st.info("RAG agent can now query both structured data and reviews")
                    except Exception as e:
                        st.error(f"Error initializing RAG agent: {e}")

    st.divider()
    with st.expander("📡 Available Endpoints"):
        st.markdown(
            """
            **API Endpoints:**
            - `/health` - Health check
            - `/sqlite?q=...` - SQL queries over products/orders (GET)
            - `/qdrant/search?q=...` - Vector search over reviews (GET)
            - `/reviews/ask?q=...` - Reviews agent answer (GET)
            - `/chat?message=...` - General chat (POST)
            
            **Agentic Mode:**
            - SQLite Agent - Intelligent SQL query generation and execution
            - Qdrant Agent - Smart vector search with metadata filtering
            - RAG Agent - Combined retrieval and generation (SQL + Vector)
            """
        )

# Main content and diagram
st.title("🤖 LLM Agent Chat Interface")
st.markdown( """
    This interface allows you to interact with the LLM Agent backend.)
                """)
# st.title("LLM Agent Chat Workflow")

# mermaid_code = """
# flowchart TD
#     A[User Query (Natural Language)] --> B[FastAPI + LLM]
#     B --> C[Route Selector]
#     C --> D1[/sqlite → SQL over products/]
#     C --> D2[/qdrant/search → Vector search/]
#     C --> D3[/reviews/ask → Review agent/]
#     D1 --> E[SQLite tables]
#     D2 --> E
#     D3 --> E
#     E --> F[Response Assembly]
#     F --> G[Chat UI Output]
# """

# st.markdown(
#     f"""
#     <div class="mermaid">
#     {mermaid_code}
#     </div>
#     <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
#     <script>mermaid.initialize({{startOnLoad:true}});</script>
#     """,
#     unsafe_allow_html=True,
# )

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Routing keyword heuristics
sql_keywords = [
    "product",
    "products",
    "price",
    "category",
    "seller",
    "sellers",
    "order",
    "orders",
    "sqlite",
    "sql",
    "table",
    "product_id",
    "seller_id",
    "payment",
    "freight",
]
review_keywords = [
    "review",
    "reviews",
    "rating",
    "comment",
    "feedback",
    "delivery",
    "late",
    "customer",
    "complaint",
    "satisfaction",
    "positivo",
    "negativo",
]

# Chat input and handling
if prompt := st.chat_input("Ask a question about your data..."):
    text = prompt.lower()
    route = "chat"
    if any(k in text for k in sql_keywords):
        route = "sqlite"
    elif any(k in text for k in review_keywords):
        route = "qdrant"

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Check if RAG agent should be used (takes precedence)
                if st.session_state.use_rag and st.session_state.rag_agent:
                    # Use RAG agent for comprehensive retrieval and generation
                    result = st.session_state.rag_agent.query(prompt, include_sources=True)
                    answer = result["answer"]
                    st.markdown(answer)
                    
                    # Show sources
                    if result.get("sources"):
                        with st.expander("📚 Information Sources"):
                            for i, source in enumerate(result["sources"], 1):
                                st.write(f"{i}. **Tool:** `{source['tool']}`")
                                st.json(source["input"])
                    
                    # Show reasoning
                    with st.expander("🔍 View RAG Agent Reasoning"):
                        for msg in result["full_conversation"]:
                            st.write(f"**{msg.__class__.__name__}:** {msg.content}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {"route": "rag-agent", "sources": len(result.get("sources", []))},
                    })
                
                # Check if agentic mode is enabled for individual agents
                elif st.session_state.use_agents and route == "sqlite" and st.session_state.sqlite_agent:
                    # Use SQLite agent
                    result = st.session_state.sqlite_agent.query(prompt)
                    answer = result["answer"]
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Agent Reasoning"):
                        for msg in result["full_conversation"]:
                            st.write(f"**{msg.__class__.__name__}:** {msg.content}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {"route": "sqlite-agent"},
                    })
                    
                elif st.session_state.use_agents and route == "qdrant" and st.session_state.qdrant_agent:
                    # Use Qdrant agent
                    result = st.session_state.qdrant_agent.query(prompt)
                    answer = result["answer"]
                    st.markdown(answer)
                    
                    with st.expander("🔍 View Agent Reasoning"):
                        for msg in result["full_conversation"]:
                            st.write(f"**{msg.__class__.__name__}:** {msg.content}")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": {"route": "qdrant-agent"},
                    })
                    
                elif route == "sqlite":
                    resp = requests.get(f"{API_URL}/sqlite", params={"q": prompt}, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        sql_text = data.get("sql", "")
                        if sql_text:
                            st.code(sql_text, language="sql")
                        rows = data.get("rows", [])
                        if rows:
                            st.subheader("SQL Result (All Rows)")
                            st.dataframe(rows)
                        md = data.get("result")
                        if md:
                            st.markdown(md)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "Displayed SQL results.",
                            "metadata": {"route": "sqlite", "rows": len(rows)},
                        })
                    else:
                        st.error(f"Error: API returned status {resp.status_code}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error {resp.status_code}",
                            "metadata": None,
                        })

                elif route == "qdrant":
                    # Try reviews agent first, then vector search, then legacy endpoint
                    resp = requests.get(f"{API_URL}/reviews/ask", params={"q": prompt}, timeout=60)
                    if resp.status_code == 404:
                        resp = requests.get(
                            f"{API_URL}/qdrant/search", params={"q": prompt, "k": 5}, timeout=60
                        )
                    if resp.status_code == 404:
                        resp = requests.get(f"{API_URL}/qdrant", timeout=60)

                    if resp.status_code == 200:
                        data = resp.json()
                        if isinstance(data, dict) and "answer" in data and "question" in data:
                            # /reviews/ask shape
                            answer = data.get("answer", "")
                            st.markdown(answer)
                            st.subheader("Raw JSON")
                            st.json(data)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "metadata": {"route": "reviews/ask"},
                            })
                        else:
                            # /qdrant/search shape or legacy
                            results = data.get("results") if isinstance(data, dict) else None
                            if isinstance(results, list):
                                if not results:
                                    answer = "No relevant reviews found."
                                else:
                                    formatted = []
                                    # Support both old shape (text+metadata) and new shape (score+id+payload)
                                    for i, d in enumerate(results, 1):
                                        payload = d.get("payload") if "payload" in d else d.get("metadata", {})
                                        txt = d.get("text") if "text" in d else (
                                            payload.get("review_comment_message") or payload.get("text") or ""
                                        )
                                        score = d.get("score") if "score" in d else payload.get("review_score", "-")
                                        title = payload.get("review_comment_title", "")
                                        prefix = f"{title} - " if title else ""
                                        formatted.append(f"{i}. Score: {score}\n{prefix}{txt}\n")
                                    answer = "\n".join(formatted)
                                st.markdown(f"```text\n{answer}\n```")
                                if results:
                                    st.subheader("Qdrant Results (Full)")
                                    table_rows = []
                                    for r in results:
                                        payload = r.get("payload") if "payload" in r else r.get("metadata", {})
                                        table_rows.append({
                                            "id": r.get("id"),
                                            "score": r.get("score"),
                                            "text": r.get("text") or payload.get("review_comment_message") or payload.get("text") or "",
                                            **payload,
                                        })
                                    st.dataframe(table_rows)
                                st.subheader("Raw JSON")
                                st.json(data)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": answer,
                                    "metadata": {"route": "qdrant"},
                                })
                            else:
                                # Legacy /qdrant raw JSON
                                st.subheader("Qdrant Raw Response")
                                st.json(data)
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": "Displayed Qdrant response.",
                                    "metadata": {"route": "qdrant-legacy"},
                                })
                    else:
                        st.error(f"Error: API returned status {resp.status_code}")
                        try:
                            st.subheader("Error Body")
                            st.text(resp.text)
                        except Exception:
                            pass
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error {resp.status_code}",
                            "metadata": None,
                        })

                else:
                    resp = requests.post(f"{API_URL}/chat", params={"message": prompt}, timeout=60)
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data.get("agent_response") or json.dumps(data)
                        st.markdown(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "metadata": {"route": "chat"},
                        })
                    else:
                        st.error(f"Error: API returned status {resp.status_code}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error {resp.status_code}",
                            "metadata": None,
                        })

            except requests.exceptions.Timeout:
                error_msg = "⏱️ Request timed out. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": None,
                })
            except requests.exceptions.RequestException as e:
                error_msg = f"❌ Connection error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": None,
                })
            except Exception as e:
                error_msg = f"❌ Unexpected error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": None,
                })

# Footer
st.divider()
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <small>Powered by FastAPI + Qdrant + OpenAI | Deployed on GCP Cloud Run</small>
</div>
""",
    unsafe_allow_html=True,
)
