# importing necessary libraries
### cateatn harus jalan di python 3.13.5 base anaconda

import os
import pandas as pd
import numpy as np
import re
import asyncio
import time
import requests
import logging
from typing import Optional
import streamlit as st
from uuid import uuid4
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import qdrant as QdrantVectorStore
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from qdrant_client.http import models

# load environment variables from .env file
load_dotenv()   

# Streamlit page config MUST be the first Streamlit command
st.set_page_config(
    page_title="Customer Review Analysis",
    layout="wide",
    page_icon="🤖",
)

import asyncio
def has_loop():
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False
# Initialize tools
QDRANT_URL = os.getenv("QDRANT_URL", "https://acb9e0ed-c7e4-4abc-9495-1382817b533e.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
DEFAULT_API_URL = "https://llm-agent-api-447949002484.us-central1.run.app/"
API_URL = os.getenv("API_URL", DEFAULT_API_URL).rstrip("/")
DEFAULT_API_TIMEOUT = int(os.getenv("API_TIMEOUT", "30"))
API_TIMEOUT = DEFAULT_API_TIMEOUT

# Optional base URL to fetch missing CSVs on cloud deploys
DATA_BASE_URL = os.getenv("DATA_BASE_URL", "").rstrip("/")

print(f"QDRANT_URL in use: {QDRANT_URL}")

# -- Qdrant connection check helper
def check_qdrant_connection(url: Optional[str], api_key: Optional[str], timeout: int = 5):
    """Return (connected: bool, message: str)."""
    if not url:
        return False, "QDRANT_URL is not set"
    try:
        # Do not send API key over insecure (http) connection. Only include api_key when URL is HTTPS.
        client_kwargs = {"url": url, "timeout": timeout}
        if api_key and url.lower().startswith("https"):
            client_kwargs["api_key"] = api_key
        else:
            if api_key:
                logging.warning("QDRANT_API_KEY is set but QDRANT_URL is not HTTPS. Not sending API key to avoid insecure usage.")

        qc = QdrantClient(**client_kwargs)
        # simple call to verify connectivity
        qc.get_collections()
        return True, "Connected"
    except Exception as e:
        return False, str(e)

# check immediately so UI and later code can react
qdrant_connected, qdrant_status_msg = check_qdrant_connection(QDRANT_URL, QDRANT_API_KEY)
print(f"Qdrant connected: {qdrant_connected} - {qdrant_status_msg}")

# Note: Remove these print statements before deploying
# print(f"OPENAI_API_KEY: {OPENAI_API_KEY}")
# print(f"SERPAPI_API_KEY: {SERPAPI_API_KEY}")
# print(f"QDRANT_API_KEY: {QDRANT_API_KEY}")
# print(f"QDRANT_URL: {QDRANT_URL}")
"""
Optimization: avoid heavy CSV merging and document creation unless we need
to create the Qdrant collection. Keep the Streamlit app fast when a collection
already exists by skipping upload/ingestion steps.
"""

# Define embedding and LLM models
embeddings = None
llm = None
if OPENAI_API_KEY:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
else:
    logging.error("OPENAI_API_KEY is not set. Embeddings and LLM will not be initialized.")

base_dir = os.path.dirname(__file__)
data_dir = os.path.join(base_dir, "data")

# Validate required CSV files exist and surface clear info in the UI
REQUIRED_CSVS = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_order_reviews_dataset.csv",
]

missing_csvs = [f for f in REQUIRED_CSVS if not os.path.isfile(os.path.join(data_dir, f))]

# Cloud-friendly fallback: try to download missing CSVs if DATA_BASE_URL is set
def _download_missing_csvs():
    downloaded = []
    if not DATA_BASE_URL:
        return downloaded
    os.makedirs(data_dir, exist_ok=True)
    for fname in list(missing_csvs):
        try:
            url = f"{DATA_BASE_URL}/{fname}"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                with open(os.path.join(data_dir, fname), "wb") as f:
                    f.write(resp.content)
                downloaded.append(fname)
        except Exception as e:
            logging.warning(f"Failed to download {fname} from {DATA_BASE_URL}: {e}")
    # Refresh missing list after attempted downloads
    missing_csvs[:] = [f for f in REQUIRED_CSVS if not os.path.isfile(os.path.join(data_dir, f))]
    return downloaded


# 9. Initialize Qdrant Vector Store
collection_name = "olist_reviews"
model_name = "text-embedding-3-small"
vector_store = None

try:
    # Try to create from Qdrant using proper import
    from langchain_community.vectorstores.qdrant import Qdrant
    
    # verify Qdrant reachable before trying to create collection
    if not qdrant_connected:
        raise Exception(f"Qdrant not reachable: {qdrant_status_msg}")
    
    # Check if collection already exists
    try:
        client_kwargs = {"url": QDRANT_URL, "timeout": 5}
        if QDRANT_API_KEY and QDRANT_URL and QDRANT_URL.lower().startswith("https"):
            client_kwargs["api_key"] = QDRANT_API_KEY
        qc = QdrantClient(**client_kwargs)
        collections = qc.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
    except Exception:
        collection_exists = False
    
    if collection_exists:
        print(f"\n📦 Collection '{collection_name}' already exists. Loading existing collection (no re-indexing)...")
        # Load existing collection WITHOUT re-uploading documents
        vector_store = Qdrant(
            client=qc,
            collection_name=collection_name,
            embeddings=embeddings,
        )
        print(f"✅ Loaded existing collection: {collection_name}")
    else:
        print(f"\n🆕 Collection '{collection_name}' does not exist. Creating new collection with documents...")
        if missing_csvs:
            fetched = _download_missing_csvs()
            if fetched:
                print(f"📥 Downloaded CSVs: {', '.join(fetched)}")
            if missing_csvs:
                raise FileNotFoundError(
                    "Missing required CSVs in data folder: "
                    + ", ".join(missing_csvs)
                    + f"\nExpected location: {data_dir}"
                    + (f"\nTried downloading from {DATA_BASE_URL} but some files are still missing." if DATA_BASE_URL else "")
                )
        # Build documents only when needed
        # Load CSVs, preprocess, and create chunks
        orders = pd.read_csv(os.path.join(data_dir, "olist_orders_dataset.csv"))
        items = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
        products = pd.read_csv(os.path.join(data_dir, "olist_products_dataset.csv"))
        customers = pd.read_csv(os.path.join(data_dir, "olist_customers_dataset.csv"))
        reviews = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))

        orders = orders.rename(columns=lambda x: x.lower()).dropna(subset=["order_id"])
        items = items.rename(columns=lambda x: x.lower()).dropna(subset=["order_id"])
        products = products.rename(columns=lambda x: x.lower()).dropna(subset=["product_id"])
        customers = customers.rename(columns=lambda x: x.lower()).dropna(subset=["customer_id"])
        reviews = reviews.rename(columns=lambda x: x.lower()).dropna(subset=["review_id"])

        df = (
            orders.merge(items, on="order_id")
            .merge(products, on="product_id")
            .merge(customers, on="customer_id")
            .merge(reviews, on="order_id", how="left")
        )
        df["review_comment_message"] = df["review_comment_message"].fillna("").str.lower()
        df["review_comment_message"] = df["review_comment_message"].apply(
            lambda x: re.sub(r"[^a-zA-ZÀ-ÿ\s]", "", x)
        )
        df["review_comment_message_en"] = df["review_comment_message"].fillna("")

        sample_texts = df["review_comment_message_en"].head(3).tolist()
        print(f"\n📋 Original Portuguese review texts (first 3):")
        for i, text in enumerate(sample_texts, 1):
            print(f"  [{i}] {text[:100] if text else '(empty)'}")
        print(f"Total reviews: {len(df)}")
        print(f"✅ Ready to process {len(df)} reviews")

        data = df.copy()
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")
        print(f"First few rows:\n{data.head()}")

        documents = []
        for index, row in data.iterrows():
            try:
                text_content = row.get("review_comment_message_en", "")
                if not text_content:
                    continue
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "order_id": str(row.get("order_id", "")),
                        "review_id": str(row.get("review_id", "")),
                        "rating": float(row.get("review_score", 0) or 0),
                        "product_id": str(row.get("product_id", "")),
                    },
                )
                documents.append(doc)
            except Exception as e:
                print(f"Error processing row {index}: {e}")

        print(f"Total documents created: {len(documents)}")
        from langchain_text_splitters import CharacterTextSplitter
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunked_documents = []
        for doc in documents:
            chunks = text_splitter.split_text(doc.page_content)
            for chunk in chunks:
                chunked_documents.append(
                    Document(page_content=chunk, metadata=doc.metadata)
                )
        print(f"Total chunked documents: {len(chunked_documents)}")

        # Only pass api_key if the URL uses HTTPS to avoid insecure API key transmission
        qdrant_kwargs = {
            "documents": chunked_documents,
            "embedding": embeddings,
            "collection_name": collection_name,
            "url": QDRANT_URL,
            "prefer_grpc": False,
            "force_recreate": False,
            "timeout": 60,
        }
        if QDRANT_API_KEY and QDRANT_URL and QDRANT_URL.lower().startswith("https"):
            qdrant_kwargs["api_key"] = QDRANT_API_KEY
        else:
            if QDRANT_API_KEY:
                logging.warning(
                    "QDRANT_API_KEY is set but QDRANT_URL is not HTTPS. Not sending API key to avoid insecure usage."
                )

        vector_store = Qdrant.from_documents(**qdrant_kwargs)
        print(
            f"✅ Successfully stored {len(chunked_documents)} documents to Qdrant collection: {collection_name}"
        )
    
    print(f"✅ Vector store ready for retrieval")
except Exception as e:
    print(f"❌ Error initializing vector store: {e}")
    vector_store = None
#
    # define sql_chain here since llm is defined
sql_chain = None
print("ℹ️ Using FastAPI /sqlite endpoint for SQL queries (no local SQL chain).")

# Define tools for the agent

def sql_query_api(query: str) -> str:
    """Execute an SQL-style question via FastAPI /sqlite and summarize results."""
    def _attempt() -> tuple[bool, str]:
        try:
            resp = requests.get(f"{API_URL}/sqlite", params={"q": query}, timeout=max(10, API_TIMEOUT))
            if resp.status_code != 200:
                try:
                    err = resp.json()
                    return False, f"SQL API error {resp.status_code}: {err}"
                except Exception:
                    return False, f"SQL API error {resp.status_code}: {resp.text}"
            data = resp.json()
            sql_text = data.get("sql", "")
            rows = data.get("rows", [])
            md = data.get("result", "")
            preview = ""
            if isinstance(rows, list) and rows:
                preview_count = min(3, len(rows))
                headers = list(rows[0].keys()) if isinstance(rows[0], dict) else []
                table_md = "| " + " | ".join(headers) + " |\n" + "| " + " | ".join(["---"]*len(headers)) + " |\n" if headers else ""
                for r in rows[:preview_count]:
                    if headers:
                        table_md += "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n"
                preview = f"Returned {len(rows)} rows. Preview (first {preview_count}):\n" + table_md
            parts = []
            if sql_text:
                parts.append("SQL executed:\n```sql\n" + sql_text + "\n```")
            if preview:
                parts.append(preview)
            if md:
                parts.append(md)
            return True, ("\n\n".join(parts) if parts else "No results.")
        except requests.exceptions.Timeout:
            return False, "timeout"
        except Exception as e:
            return False, f"Error calling SQL API: {e}"

    # Retry with simple backoff to handle cold starts or transient latency
    delays = [0, 2, 5]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        ok, res = _attempt()
        if ok:
            return res
        last_err = res
        # If it was a timeout, keep retrying; otherwise break after first non-timeout error
        if res != "timeout" and not str(res).lower().startswith("sql api error 504"):
            break
    return f"Error calling SQL API: {last_err}"

def sql_query_api_json(question: str):
    """Call /sqlite with a natural language question and return parsed JSON or error text.

    Returns dict on success: {question, sql, rows, columns, result}
    Returns str on failure with error message.
    """
    def _attempt():
        try:
            resp = requests.get(f"{API_URL}/sqlite", params={"q": question}, timeout=max(10, API_TIMEOUT))
            if resp.status_code != 200:
                try:
                    return False, resp.json()
                except Exception:
                    return False, resp.text
            return True, resp.json()
        except requests.exceptions.Timeout:
            return False, "timeout"
        except Exception as e:
            return False, f"Error calling SQL API: {e}"

    delays = [0, 2, 5]
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        ok, res = _attempt()
        if ok:
            return res
        last_err = res
        if res != "timeout":
            break
    return f"Error calling SQL API: {last_err}"

@tool
def sql_query_tool_struct(query: str) -> str:
    """Tool wrapper for SQL via FastAPI /sqlite."""
    return sql_query_api(query)
    

@tool
def current_datetime(query: str = "") -> str:
    """Get current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def search_reviews(query: str) -> str:
    """Searches for customer reviews matching the query."""
    if not vector_store:
        return "Vector store not initialized."
    try:
        results = vector_store.similarity_search(query, k=3)
        if not results:
            return "No relevant reviews found."
        response = "\n\n".join([f"Review: {d.page_content}\nMetadata: {d.metadata}" for d in results])
        return response
    except Exception as e:
        return f"Error searching reviews: {e}"

reviews_df_min = None
items_df_min = None

def _lazy_load_reviews_df():
    global reviews_df_min
    if reviews_df_min is None:
        try:
            df = pd.read_csv(os.path.join(data_dir, "olist_order_reviews_dataset.csv"))
            df = df.rename(columns=lambda x: x.lower()).dropna(subset=["review_id"])
            reviews_df_min = df[["order_id", "review_score"]].copy()
        except Exception as e:
            logging.error(
                f"Failed to load reviews CSV: {e}. Expected at: "
                f"{os.path.join(data_dir, 'olist_order_reviews_dataset.csv')}"
            )
            reviews_df_min = pd.DataFrame(columns=["order_id", "review_score"])  # empty fallback

def _lazy_load_items_df():
    global items_df_min
    if items_df_min is None:
        try:
            df = pd.read_csv(os.path.join(data_dir, "olist_order_items_dataset.csv"))
            df = df.rename(columns=lambda x: x.lower()).dropna(subset=["order_id"])
            items_df_min = df[["order_id", "product_id"]].copy()
        except Exception as e:
            logging.error(
                f"Failed to load items CSV: {e}. Expected at: "
                f"{os.path.join(data_dir, 'olist_order_items_dataset.csv')}"
            )
            items_df_min = pd.DataFrame(columns=["order_id", "product_id"])  # empty fallback

@tool
def get_review_statistics(product_id: str = "") -> str:
    """Get review statistics overall or for a specific product_id without heavy ingestion."""
    try:
        # Prefer Qdrant payloads if vector_store is ready, to avoid CSV dependency
        if vector_store and QDRANT_URL:
            try:
                client_kwargs = {"url": QDRANT_URL, "timeout": 10}
                if QDRANT_API_KEY and QDRANT_URL.lower().startswith("https"):
                    client_kwargs["api_key"] = QDRANT_API_KEY
                qc_stats = QdrantClient(**client_kwargs)
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                collected = []
                next_offset = None
                # Scroll through points and collect ratings from payload
                # Limit batches to avoid memory pressure
                while True:
                    filt = None
                    if product_id:
                        # Only include points with matching product_id in payload
                        filt = Filter(must=[FieldCondition(key="product_id", match=MatchValue(value=str(product_id)))])
                    points, next_offset = qc_stats.scroll(
                        collection_name=collection_name,
                        with_payload=True,
                        with_vectors=False,
                        limit=1000,
                        offset=next_offset,
                        filter=filt,
                    )
                    for p in points:
                        payload = getattr(p, "payload", {}) or {}
                        rating = payload.get("rating")
                        if rating is not None:
                            try:
                                collected.append(float(rating))
                            except Exception:
                                pass
                    if not next_offset:
                        break
                if not collected:
                    # Fallback to CSV path if no ratings in payload
                    raise RuntimeError("No ratings in Qdrant payloads")
                series = pd.Series(collected, dtype=float)
                stats = (
                    f"Review Statistics (Qdrant){' for product ' + str(product_id) if product_id else ''}:\n"
                    f"- Total Reviews: {int(series.count())}\n"
                    f"- Average Rating: {series.mean():.2f}\n"
                    f"- Min Rating: {series.min()}\n"
                    f"- Max Rating: {series.max()}"
                )
                return stats
            except Exception:
                # Fall back to CSV-based minimal loaders
                pass

        _lazy_load_reviews_df()
        if reviews_df_min is None or reviews_df_min.empty:
            return "No reviews found."

        if product_id:
            _lazy_load_items_df()
            if items_df_min is None or items_df_min.empty:
                return "Unable to load item mappings for product stats."
            merged = pd.merge(reviews_df_min, items_df_min, on="order_id", how="inner")
            filtered = merged[merged["product_id"] == product_id]
            if filtered.empty:
                return "No reviews found for that product."
            series = filtered["review_score"].astype(float)
        else:
            series = reviews_df_min["review_score"].astype(float)

        stats = (
            f"Review Statistics:\n"
            f"- Total Reviews: {int(series.count())}\n"
            f"- Average Rating: {series.mean():.2f}\n"
            f"- Min Rating: {series.min()}\n"
            f"- Max Rating: {series.max()}"
        )
        return stats
    except Exception as e:
        return f"Error getting statistics: {e}"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper to render agent output and parse simple TSV into a table
def render_agent_output(text: str):
    try:
        st.markdown(text)
        lines = [ln for ln in (text or "").splitlines() if ln.strip()]
        # Try to find lines that look like tab-separated values
        tsv_lines = [ln for ln in lines if "\t" in ln]
        if tsv_lines:
            rows = [ln.split("\t") for ln in tsv_lines]
            # Ensure consistent column count
            ncols = max(len(r) for r in rows)
            rows = [r + [""] * (ncols - len(r)) for r in rows]
            cols = [f"col{i+1}" for i in range(ncols)]
            df = pd.DataFrame(rows, columns=cols)
            st.subheader("Parsed Table")
            st.dataframe(df)
    except Exception:
        # Best-effort render; ignore parsing errors
        pass

# Initialize the agent if API key is available
review_agent = None
if OPENAI_API_KEY and vector_store and llm is not None:
    # Newer create_react_agent signature does not accept 'prompt'
    review_agent = create_react_agent(
        tools=[search_reviews, current_datetime, get_review_statistics, sql_query_tool_struct],
        model=llm,
    )
    
    def get_chat_bot_response(input_text, chat_history):
        """Get response from the chatbot agent."""
        try:
            if review_agent is None:
                return "Agent not initialized"
            # Prepend a system instruction to guide the agent
            sys_msg = SystemMessage(content="You analyze customer reviews and use provided tools. If the question is about products/SQL, prefer the SQL tool; for review insights, prefer the review search tool.")
            result = review_agent.invoke(
                {"messages": [sys_msg] + chat_history + [HumanMessage(content=input_text)]}
            )
            return result["messages"][-1].content
        except Exception as e:
            logger.error(f"Error in chatbot response: {e}")
            return f"Sorry, I encountered an error: {e}"
else:
    if not OPENAI_API_KEY or llm is None:
        logger.error("OPENAI_API_KEY is not set. Please configure your API key.")
    if not vector_store:
        logger.error("Vector store not initialized. Please check your Qdrant connection.")

# 11. Streamlit Web Application
st.title("🤖 Customer Review Analysis Chatbot")
st.markdown("Interact with your customer review data using AI-powered analysis!")

# Show Qdrant connection status in sidebar
with st.sidebar:
    st.header("Connections")
    if qdrant_connected:
        st.success(f"Qdrant: Connected — {qdrant_status_msg}")
    else:
        st.error(f"Qdrant: Not connected — {qdrant_status_msg}")
    st.divider()
    st.subheader("Data Files")
    st.text(f"Data folder: {data_dir}")
    if missing_csvs:
        st.error("Missing CSVs: " + ", ".join(missing_csvs))
        if DATA_BASE_URL:
            st.caption(f"Will try to fetch from {DATA_BASE_URL} during setup.")
        else:
            st.caption("Place the Olist CSVs in the data folder above or set DATA_BASE_URL.")
    else:
        st.success("All required CSVs found")
    st.divider()
    st.subheader("API")
    api_url_input = st.text_input("API URL", value=API_URL, help="FastAPI base URL")
    if api_url_input:
        API_URL = api_url_input.rstrip("/")
    API_TIMEOUT = st.number_input("API timeout (seconds)", min_value=5, max_value=120, value=API_TIMEOUT, step=5)
    if st.button("Test API"):
        try:
            # Use a longer timeout to avoid Cloud Run cold starts
            r = requests.get(f"{API_URL}/health", timeout=max(10, API_TIMEOUT))
            if r.status_code == 200:
                st.success("API reachable")
            else:
                st.error(f"API status {r.status_code}")
        except requests.exceptions.RequestException as e:
            st.error(f"API connection failed: {e}")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Simple routing keywords to improve SQL questions
sql_keywords = [
    "sql", "sqlite", "query", "select", "average", "count", "sum",
    "group", "by", "top", "price", "category", "orders", "products"
]

# Chat input
if review_agent:
    if prompt_U := st.chat_input("Ask me about the reviews or products..."):
        # Get last 20 messages for context
        chat_history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
                       for msg in st.session_state.messages[-20:]]
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt_U)
        st.session_state.messages.append({"role": "user", "content": prompt_U})
        
        # If it looks like a SQL question, call the API directly via the sql tool
        if any(k in prompt_U.lower() for k in sql_keywords):
            with st.chat_message("assistant"):
                with st.spinner("Running SQL via API..."):
                    data = sql_query_api_json(prompt_U)
                    if isinstance(data, dict) and data.get("rows") is not None:
                        sql_text = data.get("sql", "")
                        rows = data.get("rows", [])
                        md = data.get("result", "")
                        if sql_text:
                            st.code(sql_text, language="sql")
                        if isinstance(rows, list) and rows:
                            st.subheader("SQL Rows")
                            st.dataframe(rows)
                        if md:
                            st.markdown(md)
                        # Compose chat-friendly summary
                        headers = list(rows[0].keys()) if (isinstance(rows, list) and rows and isinstance(rows[0], dict)) else []
                        table_md = ""
                        if headers:
                            preview_count = min(3, len(rows))
                            table_md = "| " + " | ".join(headers) + " |\n" + "| " + " | ".join(["---"]*len(headers)) + " |\n"
                            for r in rows[:preview_count]:
                                table_md += "| " + " | ".join(str(r.get(h, "")) for h in headers) + " |\n"
                        parts = []
                        if sql_text:
                            parts.append("SQL executed:\n```sql\n" + sql_text + "\n```")
                        if rows:
                            parts.append(f"Returned {len(rows)} rows. Preview:")
                            if table_md:
                                parts.append(table_md)
                        if md:
                            parts.append(md)
                        assistant_msg = "\n\n".join(parts) if parts else "Displayed SQL results."
                        render_agent_output(assistant_msg)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_msg})
                    else:
                        # Fallback to text summary path
                        answer = sql_query_api(prompt_U)
                        render_agent_output(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            # Get and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing reviews..."):
                    try:
                        answer = get_chat_bot_response(prompt_U, chat_history)
                        render_agent_output(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"Error processing your request: {e}"
                        st.error(error_msg)
                        logger.error(error_msg)
else:
    st.error("❌ Agent not initialized. Please check your API configuration and vector store connection.")