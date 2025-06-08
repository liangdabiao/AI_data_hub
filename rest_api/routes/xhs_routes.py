from fastapi import APIRouter, HTTPException, Query
from rag_graphs.xhs_rag_graph.graph.graph import app
from db.mongo_db import get_db  # 导入MongoDB连接获取函数
from db.models.xhs_data import XHSData  # 导入公众号数据模型类
router = APIRouter()



@router.get("/recent")
def xhs_recent_records():
    """
    Get recent 10 new records directly from database.

    Returns:
        dict: Recent 10 new records.
    """
    try:

        db = get_db()
        # 查询最近10条记录（按创建时间降序）
        # 检查集合是否存在
        if XHSData.__tablename__ not in db.list_collection_names():
            raise HTTPException(status_code=404, detail="XHS data collection not found")
        recent_records = list(db[XHSData.__tablename__].find().sort("publish_time", -1).limit(30))
        # 转换MongoDB的ObjectId为字符串并移除image_arr字段
        for record in recent_records:
            record["_id"] = str(record["_id"])
            record.pop("image_arr", None)
        return {
            "count": 30,
            "result": recent_records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{ticker}")
def xhs_by_topic(
    ticker: str,
    # Optional query parameter
    topic: str  = Query(None, description="Topic"),
):
    """
    Get xhs a specific ticker.

    Args:
        ticker (str): Stock ticker symbol.
        topic (str): Topic to fetch xhs for a specific stock.

    Returns:
        dict: Relevant xhs for a speicific ticker.
    """

    try:

        if topic:
            human_query = f"News related to {topic} for {ticker}"
        else:
            human_query = f"News related to {ticker}"

        res         = app.invoke({"question": human_query})
        return {
            "ticker": ticker,
            "topic": topic,
            "result": res["generation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



