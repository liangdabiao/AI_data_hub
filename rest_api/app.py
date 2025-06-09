from flask import Flask, request, jsonify
from twoin1 import FeatureExtractor, MilvusClient
import logging
# 图片多模态处理和上传到向量数据库， 图片以文搜图，图搜搜 功能

app = Flask(__name__)

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/query')
def query():
    search_term = request.args.get('s', 'aaaa')
    
    # 初始化Milvus客户端和特征提取器
    MILVUS_HOST = "https://in03-0910f86a91.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
    MILVUS_PORT = "80"
    MILVUS_TOKEN = "4eca3d22814dfe910cf10d70d3e6715751ab7704e344e1e92ad7b2810f6b6ddf847ac3abfd8bd5d7a24ae"
    COLLECTION_NAME = "multimodal_search2"
    INDEX = "IVF_FLAT"
    DASHSCOPE_API_KEY = "sk-cacaffaa99fbe6702adf4f"
    
    milvus_client = MilvusClient(MILVUS_TOKEN, MILVUS_HOST, MILVUS_PORT, INDEX, COLLECTION_NAME)
    extractor = FeatureExtractor(DASHSCOPE_API_KEY)
    
    # 生成文本embedding并搜索
    text_embedding = extractor(search_term, "text")
    text_results_1 = milvus_client.search(text_embedding, feild = 'image_embedding')
    logger.info(f"以文搜图查询结果: {text_results_1}")
    text_results_2 = milvus_client.search(text_embedding, feild = 'text_embedding')
    logger.info(f"以文搜文查询结果: {text_results_2}")
    
    # 合并结果并返回
    results = {
        'image_search': [{'image': item['origin'], 'image_description': item['image_description']} for item in text_results_1][:2],
        'text_search': [{'image': item['origin'], 'image_description': item['image_description']} for item in text_results_2][:2],
    }
    #return jsonify(results)
    import json
    return json.dumps({'data': results}, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)