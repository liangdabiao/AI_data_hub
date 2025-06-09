import base64
import csv
import dashscope
import os
import pandas as pd
import sys
import time
from tqdm import tqdm
from datetime import datetime, timedelta
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    MilvusException,
    utility,
)
from http import HTTPStatus
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
class FeatureExtractor:
    def __init__(self, DASHSCOPE_API_KEY):
        self._api_key = DASHSCOPE_API_KEY  # 使用环境变量存储API密钥
    def __call__(self, input_data, input_type):
        if input_type not in ("image", "text"):
            raise ValueError("Invalid input type. Must be 'image' or 'text'.")
        try:
            if input_type == "image":
                if input_data.startswith(('http://', 'https://')):
                    # 处理OSS远程URL
                    payload = [{"image": input_data}]
                else:
                    # 处理本地文件路径
                    _, ext = os.path.splitext(input_data)
                    image_format = ext.lstrip(".").lower()
                    with open(input_data, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    input_data = f"data:image/{image_format};base64,{base64_image}"
                    payload = [{"image": input_data}]
            else:
                payload = [{"text": input_data}]
            resp = dashscope.MultiModalEmbedding.call(
                model="multimodal-embedding-v1",
                input=payload,
                api_key=self._api_key,
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output["embeddings"][0]["embedding"]
            else:
                raise RuntimeError(
                    f"API调用失败，状态码: {resp.status_code}, 错误信息: {resp.message}"
                )
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise

class FeatureExtractorVL:
    def __init__(self, DASHSCOPE_API_KEY):
        self._api_key = DASHSCOPE_API_KEY  # 使用环境变量存储API密钥
    def __call__(self, input_data, input_type):
        if input_type not in ("image"):
            raise ValueError("Invalid input type. Must be 'image'.")
        try:
            if input_type == "image":
                if input_data.startswith(('http://', 'https://')):
                    # 处理远程URL
                    image_url = input_data
                else:
                    # 处理本地文件路径
                    _, ext = os.path.splitext(input_data)
                    image_format = ext.lstrip(".").lower()
                    with open(input_data, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                    image_url = f"data:image/{image_format};base64,{base64_image}"
                
                payload=[
                            {
                                "role": "system",
                                "content": [{"type":"text","text": "You are a helpful assistant."}]
                            },
                            {
                                "role": "user",
                                "content": [
                                            {"image": image_url},
                                            {"text": "图片内容是积木，请详细的列出积木拼搭的所有东西，并总结图片描述的10个关键词"}
                                            ],
                            }
                        ]
            resp = dashscope.MultiModalConversation.call(
                model="qwen-vl-plus",
                messages=payload,
                api_key=self._api_key,
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output["choices"][0]["message"].content[0]["text"]
            else:
                raise RuntimeError(
                    f"API调用失败，状态码: {resp.status_code}, 错误信息: {resp.message}"
                )
        except Exception as e:
            logger.error(f"处理失败: {str(e)}")
            raise

class MilvusClient:
    def __init__(self, MILVUS_TOKEN, MILVUS_HOST, MILVUS_PORT, INDEX, COLLECTION_NAME):
        self._token = MILVUS_TOKEN
        self._host = MILVUS_HOST
        self._port = MILVUS_PORT
        self._index = INDEX
        self._collection_name = COLLECTION_NAME
        self._connect()
        self._create_collection_if_not_exists()
    def _connect(self):
        try:
            connections.connect(alias="default", uri=self._host,  token=self._token)
            logger.info("Connected to Milvus successfully.")
        except Exception as e:
            logger.error(f"连接Milvus失败: {str(e)}")
            sys.exit(1)
    def _collection_exists(self):
        return self._collection_name in utility.list_collections()

    def _create_collection_if_not_exists(self):
        try:
            if not self._collection_exists():
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="origin", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="image_description", dtype=DataType.VARCHAR, max_length=30480),
                    FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                    FieldSchema(name="text_embedding", dtype=DataType.FLOAT_VECTOR, dim=1024)
                ]
                schema = CollectionSchema(fields)
                self._collection = Collection(self._collection_name, schema)
                if self._index == 'IVF_FLAT':
                    self._create_ivf_index()
                else:
                    self._create_hnsw_index()   
                logger.info("Collection created successfully.")
            else:
                self._collection = Collection(self._collection_name)
                logger.info("Collection already exists.")
        except Exception as e:
            logger.error(f"创建或加载集合失败: {str(e)}")
            sys.exit(1)
    def _create_ivf_index(self):
        index_params = {
            "index_type": "IVF_FLAT",
            "params": {
                        "nlist": 1024, # Number of clusters for the index
                    },
            "metric_type": "L2",
        }
        self._collection.create_index("image_embedding", index_params)
        self._collection.create_index("text_embedding", index_params)
        logger.info("Index created successfully.")
    def _create_hnsw_index(self):
        index_params = {
            "index_type": "HNSW",
            "params": {
                        "M": 64, # Maximum number of neighbors each node can connect to in the graph
                        "efConstruction": 100, # Number of candidate neighbors considered for connection during index construction
                    },
            "metric_type": "L2",
        }
        self._collection.create_index("image_embedding", index_params)
        self._collection.create_index("text_embedding", index_params)
        logger.info("Index created successfully.")

    def insert(self, data):
        try:
            self._collection.insert(data)
            self._collection.load()
            logger.info("数据插入并加载成功.")
        except MilvusException as e:
            logger.error(f"插入数据失败: {str(e)}")
            raise
    def search(self, query_embedding, feild, limit=3):
        try:
            if self._index == 'IVF_FLAT':
                param={"metric_type": "L2", "params": {"nprobe": 10}}
            else:
                param={"metric_type": "L2", "params": {"ef": 10}}
            result = self._collection.search(
                data=[query_embedding],
                anns_field=feild,
                param=param,
                limit=limit,
                output_fields=["origin", "image_description"],
            )
            return [{"id": hit.id, "distance": hit.distance, "origin": hit.origin, "image_description": hit.image_description} for hit in result[0]]
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return None

def load_image_embeddings(extractor, extractorVL, csv_path):
    # 初始化OSS客户端
    from oss2 import Auth, Bucket
    auth = Auth('LTAI5tR5f73KRxrwL9Q', 'LU6nYTFACA5pGS6qFMy77')
    bucket = Bucket(auth, 'https://oss-cn-shenzhen.aliyuncs.com', 'legoallbricks')
 
    # 读取CSV并检查是否有finish列
    df = pd.read_csv(csv_path)
    if 'finish' not in df.columns:
        df['finish'] = False
    
    # 只处理未完成的记录
    unfinished_df = df[(df['finish'] == False) & (df['image'].notna()) & (df['image'] != '')]
    # print(unfinished_df["image"].tolist()[:5])
    # return false;
    image_embeddings = {}
    for image_path in tqdm(unfinished_df["image"].tolist()[:5], desc="生成图像embedding"):
        try:
            # 增加逻辑，判断如果image_path为空，那么跳过
            if pd.isna(image_path) or image_path.strip() == '':
                df.loc[df['image'] == image_path, 'finish'] = True
                df.to_csv(csv_path, index=False)
                continue
            # 上传图片到OSS
            # 处理包含'!'和不包含'!'的URL
            if '!' in image_path:
                object_name = os.path.basename(image_path.split('!')[0])
            else:
                object_name = os.path.basename(image_path)
            if image_path.startswith(('http://', 'https://')):
                # 处理远程URL
                import requests
                max_retries = 3
                retry_count = 0
                success = False
                
                while retry_count < max_retries and not success:
                    try:
                        # 验证URL有效性
                        if not image_path.startswith(('http://', 'https://')):
                            logger.warning(f"无效的URL格式: {image_path}")
                            retry_count += 1
                            time.sleep(2)
                            continue
                            
                        headers = {
                                'Referer': 'http://brick4.com/',
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                                'Cache-Control': 'max-age=0',
                                'X-Reqid': 'ZjUAAABbRurD-j8Y',
                                'X-Log': 'X-Log',
                                'Accept': 'image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
                                'Accept-Encoding': 'gzip, deflate, br',
                                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8'
                            }
                        try:
                            print(f"正在下载图片: {image_path}")
                            response = requests.get(image_path, headers=headers, timeout=10)
                            print(f"下载完成，状态码: {response.status_code}")
                            # 打印调试
                            #print(f"响应头: {response.headers}")
                            if response.status_code == 200:
                                # 确保临时文件目录存在
                                temp_dir = 'temp_images'
                                os.makedirs(temp_dir, exist_ok=True)
                                print(f"object_name: {object_name}")
                                temp_path = os.path.join(temp_dir, object_name)
                                print(f"temp_path: {temp_path}")
                                
                                with open(temp_path, 'wb') as f:
                                    f.write(response.content)
                                
                                # 验证文件大小
                                if os.path.getsize(temp_path) > 0:
                                    with open(temp_path, 'rb') as fileobj:
                                        #result = bucket.put_object_from_file(object_name, temp_path)
                                        bucket.put_object(object_name, fileobj)
                                        #print(f"上传成功cc: {result}")
                                    os.remove(temp_path)
                                    oss_url = f'https://legoallbricks.oss-cn-shenzhen.aliyuncs.com/{object_name}'
                                    print(f"上传成功: {oss_url}")
                                else:
                                    raise ValueError("下载的文件大小为0")
                            else:
                                raise requests.exceptions.HTTPError(f"HTTP错误: {response.status_code}")

                            try:
                                desc = extractorVL(oss_url, "image")
                                # 获取当前行的title并添加到desc
                                title = df[df["image"] == image_path]["title"].values[0]
                                content = df[df["image"] == image_path]["content"].values[0]
                                brand = df[df["image"] == image_path]["brand"].values[0]
                                category = df[df["image"] == image_path]["category"].values[0]
                                desc = f"{desc}\n\n积木名称: {title}"
                                desc = f"{desc}\n\n积木介绍: {content}"
                                desc = f"{desc}\n\n积木品牌: {brand}"
                                desc = f"{desc}\n\n积木category: {category}"
                                image_embeddings[oss_url] = [desc, extractor(oss_url, "image"), extractor(desc, "text")]
                                success = True
                                print(f"处理成功: {oss_url}")
                            except Exception as e:
                                logger.warning(f"API处理失败: {str(e)}")
                                df.loc[df['image'] == image_path, 'finish'] = 'badass'
                                df.to_csv(csv_path, index=False)
                                retry_count += 1
                                time.sleep(2)
                                continue
                        except requests.exceptions.RequestException as e:
                            logger.warning(f"URL请求失败: {str(e)}")
                            retry_count += 1
                            time.sleep(2)
                            continue
                        
                    except Exception as e:
                        logger.warning(f"下载远程图片 {image_path} 失败，重试 {retry_count + 1}/{max_retries}: {str(e)}")
                        retry_count += 1
                        time.sleep(2)
                
                if not success:
                    logger.warning(f"处理远程图片 {image_path} 失败，已跳过")
                    df.loc[df['image'] == image_path, 'finish'] = True
                    df.to_csv(csv_path, index=False)
                    continue
            else:
                # 处理本地文件
                with open(image_path, 'rb') as fileobj:
                    bucket.put_object(object_name, fileobj)
                oss_url = f'https://legoallbricks.oss-cn-shenzhen.aliyuncs.com/{object_name}'
                desc = extractorVL(oss_url, "image")
                image_embeddings[oss_url] = [desc, extractor(oss_url, "image"), extractor(desc, "text")]
            time.sleep(1)
            df.loc[df['image'] == image_path, 'finish'] = True
            df.to_csv(csv_path, index=False)
        except Exception as e:
            logger.warning(f"处理{image_path}失败，已跳过: {str(e)}")
    # 确保返回的数据格式符合Milvus要求
    data = []
    for k, v in image_embeddings.items():
        data.append({
            "origin": str(k),
            "image_description": str(v[0]),
            "image_embedding": v[1],
            "text_embedding": v[2]
        })
    

    
    return data

# 图片上传，然后以文搜图，以图搜图流程
if __name__ == "__main__":
    MILVUS_HOST = "https://in03-0910f8786a91.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
    MILVUS_PORT = "80"
    MILVUS_TOKEN = "4eca3d22814dce9c1686f6e1624795715751ab7704e344e1e92ad7b2810f6b6ddf847ac3abfd8bd5d7a24ae"
    COLLECTION_NAME = "multimodal_search2"
    INDEX = "IVF_FLAT" # IVF_FLAT OR HNSW

    # Step1：初始化Milvus客户端
    milvus_client = MilvusClient(MILVUS_TOKEN, MILVUS_HOST, MILVUS_PORT, INDEX, COLLECTION_NAME)
    DASHSCOPE_API_KEY = "sk-cacaf8d34463aa99fbe6702adf4f"

    # Step2：初始化千问VL大模型与多模态Embedding模型
    extractor = FeatureExtractor(DASHSCOPE_API_KEY)
    extractorVL = FeatureExtractorVL(DASHSCOPE_API_KEY)

    # Step3：将图片数据集Embedding后插入到Milvus
    embeddings = load_image_embeddings(extractor, extractorVL, "sc_legos2.csv")
    milvus_client.insert(embeddings)

    #Step4：多模态搜索示例，以文搜图和以文搜文
    text_query = "大型车"
    text_embedding = extractor(text_query, "text")
    text_results_1 = milvus_client.search(text_embedding, feild = 'image_embedding')
    logger.info(f"以文搜图查询结果: {text_results_1}")
    text_results_2 = milvus_client.search(text_embedding, feild = 'text_embedding')
    logger.info(f"以文搜文查询结果: {text_results_2}")

    #Step5：多模态搜索示例，以图搜图和以图搜文
    image_query_path = "./test/lion/n02129165_13728.JPEG"
    image_embedding = extractor(image_query_path, "image")
    image_results_1 = milvus_client.search(image_embedding, feild = 'image_embedding')
    logger.info(f"以图搜图查询结果: {image_results_1}")
    image_results_2 = milvus_client.search(image_embedding, feild = 'text_embedding')
    logger.info(f"以图搜文查询结果: {image_results_2}")