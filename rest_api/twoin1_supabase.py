import base64
import csv
import dashscope
import os
import pandas as pd
import sys
import time
from tqdm import tqdm
from datetime import datetime, timedelta
from supabase import create_client
import vecs
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

class SupabaseClient:
    def __init__(self, SUPABASE_URL, SUPABASE_KEY, COLLECTION_NAME):
        self._url = SUPABASE_URL
        self._key = SUPABASE_KEY
        self._collection_name = COLLECTION_NAME
        self._client = create_client(self._url, self._key)
        self._table = self._client.table(self._collection_name)
        logger.info("Connected to Supabase successfully.")

    def insert(self, data):
        try:
            if isinstance(data, list):
                for item in data:
                    self._table.insert(item).execute()
            else:
                self._table.insert(data).execute()
            logger.info("数据插入成功.")
        except Exception as e:
            logger.error(f"插入数据失败: {str(e)}")
            raise

    def search(self, query_embedding, field, limit=3):
        try:
            # 使用vecs进行向量搜索
            vx = vecs.create_client(self._url)
            collection = vx.get_or_create_collection(
                name=self._collection_name,
                dimension=len(query_embedding)
            )
            
            results = collection.query(
                data=query_embedding,
                limit=limit,
                include_value=True
            )
            
            return [{
                "id": item[0],
                "distance": item[1],
                "origin": item[2]["origin"],
                "image_description": item[2]["image_description"]
            } for item in results]
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return None

def load_image_embeddings(extractor, extractorVL, csv_path, storage_type='oss'):
    # 初始化存储客户端
    if storage_type == 'oss':
        from oss2 import Auth, Bucket
        auth = Auth('LTAI5tR5f73KRxrwL9Q', 'LU6nYTFACA5pGS6qFMy77')
        bucket = Bucket(auth, 'https://oss-cn-shenzhen.aliyuncs.com', 'legoallbricks')
    elif storage_type == 'supabase':
        import os
        supabase_url = os.getenv('SUPABASE_URL')
        supabase_key = os.getenv('SUPABASE_KEY')
        supabase: Client = create_client(supabase_url, supabase_key)
        bucket_name = 'legoallbricks'
 
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
                                    if storage_type == 'oss':
                                        with open(temp_path, 'rb') as fileobj:
                                            bucket.put_object(object_name, fileobj)
                                        file_url = f'https://legoallbricks.oss-cn-shenzhen.aliyuncs.com/{object_name}'
                                    elif storage_type == 'supabase':
                                        with open(temp_path, 'rb') as fileobj:
                                            supabase.storage.from_(bucket_name).upload(
                                                path=object_name,
                                                file=fileobj,
                                                file_options={"content-type": "image/jpeg"}
                                            )
                                        file_url = f'{supabase_url}/storage/v1/object/public/{bucket_name}/{object_name}'
                                    
                                    os.remove(temp_path)
                                    print(f"上传成功: {file_url}")
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
                if storage_type == 'oss':
                    with open(image_path, 'rb') as fileobj:
                        bucket.put_object(object_name, fileobj)
                    file_url = f'https://legoallbricks.oss-cn-shenzhen.aliyuncs.com/{object_name}'
                elif storage_type == 'supabase':
                    with open(image_path, 'rb') as fileobj:
                        supabase.storage.from_(bucket_name).upload(
                            path=object_name,
                            file=fileobj,
                            file_options={"content-type": "image/jpeg"}
                        )
                    file_url = f'{supabase_url}/storage/v1/object/public/{bucket_name}/{object_name}'
                desc = extractorVL(file_url, "image")
                image_embeddings[file_url] = [desc, extractor(file_url, "image"), extractor(desc, "text")]
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
    SUPABASE_URL = "https://your-supabase-url.supabase.co"
    SUPABASE_KEY = "your-supabase-key"
    COLLECTION_NAME = "multimodal_search2"

    # Step1：初始化Supabase客户端
    supabase_client = SupabaseClient(SUPABASE_URL, SUPABASE_KEY, COLLECTION_NAME)
    DASHSCOPE_API_KEY = "sk-cacaf8d34463aa99fbe6702adf4f"

    # Step2：初始化千问VL大模型与多模态Embedding模型
    extractor = FeatureExtractor(DASHSCOPE_API_KEY)
    extractorVL = FeatureExtractorVL(DASHSCOPE_API_KEY)

    # Step3：将图片数据集Embedding后插入到Supabase
    embeddings = load_image_embeddings(extractor, extractorVL, "sc_legos2.csv")
    supabase_client.insert(embeddings)

    #Step4：多模态搜索示例，以文搜图和以文搜文
    text_query = "大型车"
    text_embedding = extractor(text_query, "text")
    text_results_1 = supabase_client.search(text_embedding, field = 'image_embedding')
    logger.info(f"以文搜图查询结果: {text_results_1}")
    text_results_2 = supabase_client.search(text_embedding, field = 'text_embedding')
    logger.info(f"以文搜文查询结果: {text_results_2}")

    #Step5：多模态搜索示例，以图搜图和以图搜文
    image_query_path = "./test/lion/n02129165_13728.JPEG"
    image_embedding = extractor(image_query_path, "image")
    image_results_1 = supabase_client.search(image_embedding, field = 'image_embedding')
    logger.info(f"以图搜图查询结果: {image_results_1}")
    image_results_2 = supabase_client.search(image_embedding, field = 'text_embedding')
    logger.info(f"以图搜文查询结果: {image_results_2}")