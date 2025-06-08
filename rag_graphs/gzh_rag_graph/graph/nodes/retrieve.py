# Code for retrieval node
from typing import Any, Dict
from rag_graphs.gzh_rag_graph.graph.state import GraphState
from rag_graphs.gzh_rag_graph.ingestion import gzh_articles_retriever
from utils.logger import logger

def retrieve(state:GraphState)->Dict[str, Any]:
    logger.info("---RETRIEVE---")
    question    = state['question']
    documents   = gzh_articles_retriever.invoke(question)

    return {"documents": documents, "question": question}