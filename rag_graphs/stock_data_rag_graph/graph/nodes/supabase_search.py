from typing import Any, Dict
from dotenv import load_dotenv
from db.supabase_db import SupabaseDBClient
from rag_graphs.stock_data_rag_graph.graph.state import GraphState
import os
import pandas as pd
from utils.logger import logger

load_dotenv()

def initialize_supabase_client():
    """
    Initialize the SupabaseDBClient using .env credentials.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    return SupabaseDBClient(url, key)

def execute_supabase_query(query: str, params: dict = None):
    """
    Execute a query using SupabaseDBClient and return results as a DataFrame.

    Args:
        query (str): The query to execute.
        params (dict, optional): Parameters for the query.

    Returns:
        pd.DataFrame: The query results as a Pandas DataFrame.
    """
    db_client = initialize_supabase_client()
    try:
        if "select" in query.lower():  # For SELECT queries
            results = db_client.read("your_table_name", {"column": "value"})  # Modify with actual table/conditions
            return pd.DataFrame(results)
        else:  # For other queries (INSERT, UPDATE, DELETE)
            # Implement appropriate Supabase operations here
            return pd.DataFrame()  # Return an empty DataFrame for non-SELECT
    except Exception as e:
        logger.error(f"Error executing Supabase query: {e}")
        raise

def supabase_fetch_query(state: GraphState) -> Dict[str, Any]:
    logger.info("---SUPABASE SEARCH---")
    query = state["supabase_query"]  # Assuming similar state structure
    results = execute_supabase_query(query)
    return {"supabase_results": results, "supabase_query": query}