from dotenv import load_dotenv
from rag_graphs.xhs_rag_graph.graph.graph import app
from utils.logger import logger

load_dotenv()

if __name__=='__main__':
    logger.info("--STOCK XHS GRAPH--")
    res = app.invoke({"question": "Documents related to Apple"})
