"""
Agentic functionality for SQLite and Qdrant vector operations
Uses LangGraph to create intelligent agents that can reason and execute queries
"""

import os
import sqlite3
from typing import Optional, List, Dict, Any, Union
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()


class SQLiteAgent:
    """
    Agentic interface for SQLite database operations
    Uses LangGraph ReAct agent to intelligently query the database
    """
    
    def __init__(self, db_path: str, llm: Optional[ChatOpenAI] = None):
        self.db_path = db_path
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.tools = self._create_tools()
        self.agent = create_react_agent(self.llm, self.tools)
        
    def _create_tools(self):
        """Create tools for the SQLite agent"""
        db_path = self.db_path
        
        @tool
        def list_tables() -> str:
            """List all tables in the SQLite database"""
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                conn.close()
                return f"Available tables: {', '.join(tables)}"
            except Exception as e:
                return f"Error listing tables: {str(e)}"
        
        @tool
        def describe_table(table_name: str) -> str:
            """Get the schema/structure of a specific table"""
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                conn.close()
                
                schema = f"Schema for table '{table_name}':\n"
                for col in columns:
                    schema += f"  - {col[1]} ({col[2]})\n"
                return schema
            except Exception as e:
                return f"Error describing table: {str(e)}"
        
        @tool
        def execute_query(sql_query: str) -> str:
            """Execute a SQL query and return the results"""
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                # Get column names
                columns = [description[0] for description in cursor.description] if cursor.description else []
                
                # Fetch results
                results = cursor.fetchall()
                conn.close()
                
                if not results:
                    return "Query executed successfully but returned no results."
                
                # Format results
                output = f"Columns: {', '.join(columns)}\n\n"
                output += f"Found {len(results)} rows:\n"
                for i, row in enumerate(results[:10], 1):  # Limit to first 10 rows
                    output += f"{i}. {row}\n"
                
                if len(results) > 10:
                    output += f"... and {len(results) - 10} more rows"
                
                return output
            except Exception as e:
                return f"Error executing query: {str(e)}"
        
        @tool
        def get_sample_data(table_name: str, limit: int = 5) -> str:
            """Get sample data from a table"""
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit};")
                
                columns = [description[0] for description in cursor.description]
                results = cursor.fetchall()
                conn.close()
                
                output = f"Sample data from '{table_name}' (limit {limit}):\n"
                output += f"Columns: {', '.join(columns)}\n\n"
                for i, row in enumerate(results, 1):
                    output += f"{i}. {row}\n"
                
                return output
            except Exception as e:
                return f"Error getting sample data: {str(e)}"
        
        return [list_tables, describe_table, execute_query, get_sample_data]
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the database using natural language
        Returns the agent's response with reasoning steps
        """
        system_message = SystemMessage(content="""You are a helpful SQL database assistant.
Your job is to help users query a SQLite database by:
1. Understanding their natural language question
2. Exploring the database schema if needed
3. Writing and executing appropriate SQL queries
4. Explaining the results in a clear, human-readable way

Always use the provided tools to interact with the database.
When constructing SQL queries, be careful about table and column names.
""")
        
        messages = [system_message, HumanMessage(content=question)]
        
        result = self.agent.invoke({"messages": messages})
        
        return {
            "question": question,
            "answer": result["messages"][-1].content,
            "full_conversation": result["messages"]
        }


class QdrantAgent:
    """
    Agentic interface for Qdrant vector database operations
    Uses LangGraph ReAct agent to intelligently search and analyze vector data
    """
    
    def __init__(
        self,
        collection_name: str,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        llm: Optional[ChatOpenAI] = None,
        embeddings: Optional[OpenAIEmbeddings] = None
    ):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.qdrant_api_key = qdrant_api_key or os.getenv("QDRANT_API_KEY")
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize Qdrant client
        client_kwargs = {"url": self.qdrant_url, "timeout": 30}
        if self.qdrant_api_key and self.qdrant_url.lower().startswith("https"):
            client_kwargs["api_key"] = self.qdrant_api_key
        
        self.client = QdrantClient(**client_kwargs)
        self.vectorstore = None
        self._init_vectorstore()
        
        self.tools = self._create_tools()
        self.agent = create_react_agent(self.llm, self.tools)
    
    def _init_vectorstore(self):
        """Initialize the vector store"""
        try:
            self.vectorstore = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                embeddings=self.embeddings
            )
        except Exception as e:
            print(f"Warning: Could not initialize vector store: {e}")
    
    def _create_tools(self):
        """Create tools for the Qdrant agent"""
        
        @tool
        def search_vectors(query: str, k: int = 5) -> str:
            """Search for similar documents in the vector database"""
            try:
                if not self.vectorstore:
                    return "Vector store not initialized"
                
                results = self.vectorstore.similarity_search_with_score(query, k=k)
                
                if not results:
                    return "No results found for the query."
                
                output = f"Found {len(results)} similar documents:\n\n"
                for i, (doc, score) in enumerate(results, 1):
                    content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    metadata = doc.metadata
                    output += f"{i}. Score: {score:.4f}\n"
                    output += f"   Content: {content}\n"
                    output += f"   Metadata: {metadata}\n\n"
                
                return output
            except Exception as e:
                return f"Error searching vectors: {str(e)}"
        
        @tool
        def get_collection_info() -> str:
            """Get information about the Qdrant collection"""
            try:
                info = self.client.get_collection(self.collection_name)
                return f"""Collection: {self.collection_name}
Status: {info.status}
Vectors count: {info.vectors_count}
Points count: {info.points_count}
Vector size: {info.config.params.vectors.size if hasattr(info.config.params, 'vectors') else 'N/A'}
"""
            except Exception as e:
                return f"Error getting collection info: {str(e)}"
        
        @tool
        def filter_by_metadata(field: str, value: str, limit: int = 10) -> str:
            """Filter documents by metadata field and value"""
            try:
                must_conditions = [
                    models.FieldCondition(
                        key=field,
                        match=models.MatchValue(value=value)
                    )
                ]
                
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=models.Filter(must=must_conditions),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                
                points = results[0]
                
                if not points:
                    return f"No documents found with {field}={value}"
                
                output = f"Found {len(points)} documents with {field}={value}:\n\n"
                for i, point in enumerate(points, 1):
                    payload = point.payload
                    output += f"{i}. ID: {point.id}\n"
                    output += f"   Payload: {payload}\n\n"
                
                return output
            except Exception as e:
                return f"Error filtering by metadata: {str(e)}"
        
        @tool
        def aggregate_by_field(field: str) -> str:
            """Get aggregated statistics for a specific field in the collection"""
            try:
                # Scroll through all points to aggregate
                all_points = []
                next_offset = None
                
                while True:
                    results = self.client.scroll(
                        collection_name=self.collection_name,
                        limit=1000,
                        offset=next_offset,
                        with_payload=True,
                        with_vectors=False
                    )
                    points, next_offset = results
                    all_points.extend(points)
                    
                    if not next_offset:
                        break
                
                # Extract field values
                values = []
                for point in all_points:
                    if field in point.payload:
                        values.append(point.payload[field])
                
                if not values:
                    return f"No values found for field '{field}'"
                
                # Calculate statistics
                from collections import Counter
                counter = Counter(values)
                
                output = f"Aggregation for field '{field}':\n"
                output += f"Total entries: {len(values)}\n"
                output += f"Unique values: {len(counter)}\n\n"
                output += "Top 10 most common values:\n"
                for value, count in counter.most_common(10):
                    output += f"  - {value}: {count} occurrences\n"
                
                return output
            except Exception as e:
                return f"Error aggregating field: {str(e)}"
        
        return [search_vectors, get_collection_info, filter_by_metadata, aggregate_by_field]
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the vector database using natural language
        Returns the agent's response with reasoning steps
        """
        system_message = SystemMessage(content="""You are a helpful vector database assistant.
Your job is to help users search and analyze a Qdrant vector database by:
1. Understanding their natural language question
2. Performing semantic search when appropriate
3. Filtering by metadata fields when specific criteria are mentioned
4. Aggregating and analyzing data to answer questions
5. Explaining the results in a clear, human-readable way

Always use the provided tools to interact with the vector database.
When users ask about reviews, ratings, or sentiment, use the search_vectors tool.
When they ask about statistics or counts, use filter_by_metadata or aggregate_by_field tools.
""")
        
        messages = [system_message, HumanMessage(content=question)]
        
        result = self.agent.invoke({"messages": messages})
        
        return {
            "question": question,
            "answer": result["messages"][-1].content,
            "full_conversation": result["messages"]
        }


class RAGAgent:
    """
    Agentic RAG (Retrieval-Augmented Generation) system
    Combines SQLite and Qdrant agents with intelligent retrieval and synthesis
    """
    
    def __init__(
        self,
        sqlite_agent: Optional[SQLiteAgent] = None,
        qdrant_agent: Optional[QdrantAgent] = None,
        llm: Optional[ChatOpenAI] = None,
        db_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        embeddings: Optional[OpenAIEmbeddings] = None
    ):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Initialize or use provided agents
        self.sqlite_agent = sqlite_agent
        if not self.sqlite_agent and db_path:
            self.sqlite_agent = SQLiteAgent(db_path=db_path, llm=self.llm)
        
        self.qdrant_agent = qdrant_agent
        if not self.qdrant_agent and collection_name:
            self.qdrant_agent = QdrantAgent(
                collection_name=collection_name,
                qdrant_url=qdrant_url,
                qdrant_api_key=qdrant_api_key,
                llm=self.llm,
                embeddings=embeddings or OpenAIEmbeddings(model="text-embedding-3-small")
            )
        
        self.tools = self._create_tools()
        self.agent = create_react_agent(self.llm, self.tools)
    
    def _create_tools(self):
        """Create RAG tools that combine SQL and vector search"""
        
        @tool
        def query_structured_data(question: str) -> str:
            """Query structured data from SQLite database (products, orders, customers, etc.)"""
            if not self.sqlite_agent:
                return "SQLite agent not initialized"
            
            try:
                result = self.sqlite_agent.query(question)
                return result["answer"]
            except Exception as e:
                return f"Error querying structured data: {str(e)}"
        
        @tool
        def search_reviews(query: str, k: int = 5) -> str:
            """Search customer reviews using semantic search"""
            if not self.qdrant_agent:
                return "Qdrant agent not initialized"
            
            try:
                result = self.qdrant_agent.query(f"Search for: {query}. Return top {k} results.")
                return result["answer"]
            except Exception as e:
                return f"Error searching reviews: {str(e)}"
        
        @tool
        def get_product_reviews(product_id: str, limit: int = 5) -> str:
            """Get reviews for a specific product by product_id"""
            if not self.qdrant_agent:
                return "Qdrant agent not initialized"
            
            try:
                result = self.qdrant_agent.query(
                    f"Filter reviews by product_id={product_id} and return {limit} results"
                )
                return result["answer"]
            except Exception as e:
                return f"Error getting product reviews: {str(e)}"
        
        @tool
        def analyze_sentiment(topic: str) -> str:
            """Analyze customer sentiment on a specific topic from reviews"""
            if not self.qdrant_agent:
                return "Qdrant agent not initialized"
            
            try:
                result = self.qdrant_agent.query(
                    f"Search for reviews about {topic} and analyze the sentiment. "
                    f"Categorize as positive, negative, or neutral."
                )
                return result["answer"]
            except Exception as e:
                return f"Error analyzing sentiment: {str(e)}"
        
        @tool
        def cross_reference_data(product_query: str, review_query: str) -> str:
            """Cross-reference product data with customer reviews"""
            if not self.sqlite_agent or not self.qdrant_agent:
                return "Both SQLite and Qdrant agents required"
            
            try:
                # Get product info
                product_result = self.sqlite_agent.query(product_query)
                product_info = product_result["answer"]
                
                # Get related reviews
                review_result = self.qdrant_agent.query(review_query)
                review_info = review_result["answer"]
                
                return f"PRODUCT DATA:\n{product_info}\n\nCUSTOMER REVIEWS:\n{review_info}"
            except Exception as e:
                return f"Error cross-referencing data: {str(e)}"
        
        tools = []
        if self.sqlite_agent:
            tools.append(query_structured_data)
        if self.qdrant_agent:
            tools.extend([search_reviews, get_product_reviews, analyze_sentiment])
        if self.sqlite_agent and self.qdrant_agent:
            tools.append(cross_reference_data)
        
        return tools
    
    def query(self, question: str, include_sources: bool = True) -> Dict[str, Any]:
        """
        Query using RAG - retrieve relevant information and generate comprehensive answer
        
        Args:
            question: Natural language question
            include_sources: Whether to include source information in response
            
        Returns:
            Dictionary with answer, sources, and conversation history
        """
        system_message = SystemMessage(content="""You are an intelligent RAG (Retrieval-Augmented Generation) assistant.

Your job is to answer user questions by:
1. Analyzing what information is needed (structured data, reviews, or both)
2. Using the appropriate tools to retrieve relevant information
3. Synthesizing the information into a comprehensive, well-structured answer
4. Citing sources when providing specific data or reviews

Guidelines:
- For questions about products, prices, categories, orders → use query_structured_data
- For questions about customer opinions, sentiment, feedback → use search_reviews
- For questions about specific products and their reviews → use cross_reference_data
- For sentiment analysis → use analyze_sentiment
- Always provide clear, actionable insights
- When mentioning statistics or specific data, be precise
- Organize your answer in a logical, easy-to-read format

Remember: You have access to both structured database and unstructured review data. Use them together for comprehensive answers.
""")
        
        messages = [system_message, HumanMessage(content=question)]
        
        result = self.agent.invoke({"messages": messages})
        
        # Extract sources from conversation if requested
        sources = []
        if include_sources:
            for msg in result["messages"]:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tool_call in msg.tool_calls:
                        sources.append({
                            "tool": tool_call.get("name", "unknown"),
                            "input": tool_call.get("args", {})
                        })
        
        return {
            "question": question,
            "answer": result["messages"][-1].content,
            "sources": sources,
            "full_conversation": result["messages"]
        }
    
    def multi_hop_query(self, questions: List[str]) -> Dict[str, Any]:
        """
        Execute multiple related queries in sequence (multi-hop reasoning)
        Each answer builds on the previous ones
        """
        conversation_history = []
        answers = []
        
        for i, question in enumerate(questions, 1):
            # Build context from previous answers
            context = ""
            if answers:
                context = "\n\nPrevious findings:\n" + "\n".join(
                    [f"{j}. {ans}" for j, ans in enumerate(answers, 1)]
                )
            
            full_question = f"{question}{context}"
            result = self.query(full_question, include_sources=True)
            
            answers.append(result["answer"])
            conversation_history.extend(result["full_conversation"])
        
        # Generate final synthesis
        synthesis_prompt = f"""Based on the following multi-hop query results, provide a comprehensive summary:

Questions asked:
{chr(10).join([f"{i}. {q}" for i, q in enumerate(questions, 1)])}

Findings:
{chr(10).join([f"{i}. {a}" for i, a in enumerate(answers, 1)])}

Provide a coherent synthesis that connects all findings:"""
        
        final_result = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
        
        return {
            "questions": questions,
            "individual_answers": answers,
            "synthesis": final_result.content,
            "full_conversation": conversation_history
        }


# Convenience functions for quick initialization
def create_sqlite_agent(db_path: str, llm: Optional[ChatOpenAI] = None) -> SQLiteAgent:
    """Create a SQLite agent with default configuration"""
    return SQLiteAgent(db_path=db_path, llm=llm)


def create_qdrant_agent(
    collection_name: str,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    llm: Optional[ChatOpenAI] = None,
    embeddings: Optional[OpenAIEmbeddings] = None
) -> QdrantAgent:
    """Create a Qdrant agent with default configuration"""
    return QdrantAgent(
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        llm=llm,
        embeddings=embeddings
    )


def create_rag_agent(
    db_path: Optional[str] = None,
    collection_name: Optional[str] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
    llm: Optional[ChatOpenAI] = None,
    embeddings: Optional[OpenAIEmbeddings] = None,
    sqlite_agent: Optional[SQLiteAgent] = None,
    qdrant_agent: Optional[QdrantAgent] = None
) -> RAGAgent:
    """Create a RAG agent with default configuration"""
    return RAGAgent(
        sqlite_agent=sqlite_agent,
        qdrant_agent=qdrant_agent,
        llm=llm,
        db_path=db_path,
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_api_key,
        embeddings=embeddings
    )
