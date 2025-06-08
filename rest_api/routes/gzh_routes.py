from fastapi import APIRouter, HTTPException, Query
from rag_graphs.gzh_rag_graph.graph.graph import app
from db.mongo_db import get_db  # 导入MongoDB连接获取函数
from db.models.gzh_data import GZHData  # 导入公众号数据模型类
router = APIRouter()

@router.get("/recent")
def gzh_recent_records():
    """
    Get recent 10 new records directly from database.

    Returns:
        dict: Recent 10 new records.
    """
    try:

        db = get_db()
        # 查询最近10条记录（按创建时间降序）
        # 检查集合是否存在
        if GZHData.__tablename__ not in db.list_collection_names():
            raise HTTPException(status_code=404, detail="GZH data collection not found")
        recent_records = list(db[GZHData.__tablename__].find().sort("publish_time", -1).limit(30))
        # 转换MongoDB的ObjectId为字符串
        for record in recent_records:
            record["_id"] = str(record["_id"])
        return {
            "count": 30,
            "result": recent_records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{ticker}")
def gzh_by_topic(
    ticker: str,
    # Optional query parameter
    topic: str  = Query(None, description="Topic"),
):
    """
    Get news a specific ticker.

    Args:
        ticker (str): Stock ticker symbol.
        topic (str): Topic to fetch news for a specific stock.

    Returns:
        dict: Relevant news for a specific ticker.
    """
    try:
        if topic:
            human_query = f"News related to {topic} for {ticker}"
        else:
            human_query = f"News related to {ticker}"
        res = app.invoke({"question": human_query})
        return {
            "ticker": ticker,
            "topic": topic,
            "result": res["generation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))