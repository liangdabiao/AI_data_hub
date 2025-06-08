import os
import oss2
import requests
import os
from dotenv import load_dotenv
from utils.logger import logger

load_dotenv()

class OSSUploader:
    def __init__(self):
        # 从环境变量获取OSS配置
        self.access_key_id = os.getenv("ALIYUN_OSS_ACCESS_KEY_ID")
        self.access_key_secret = os.getenv("ALIYUN_OSS_ACCESS_KEY_SECRET")
        self.endpoint = os.getenv("ALIYUN_OSS_ENDPOINT")
        self.bucket_name = os.getenv("ALIYUN_OSS_BUCKET_NAME")
        
        # 初始化OSS客户端
        self.auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(self.auth, self.endpoint, self.bucket_name)
    
    def upload_to_oss(self, image_url):
        """
        上传图片到阿里云OSS
        :param image_url: 图片URL
        :return: OSS访问URL
        """
        try:
            # 下载图片
            response = requests.get(image_url)
            response.raise_for_status()
            
            # 生成OSS对象名称
            file_name = os.path.basename(image_url)
            object_name = f"xhs_images/{file_name}.jpg"
            
            # 上传到OSS
            result = self.bucket.put_object(object_name, response.content)
            
            if result.status == 200:
                # 返回OSS访问URL
                print("OSS上传成功:")
                print(f"https://{self.bucket_name}.{self.endpoint}/{object_name}")
                return f"https://{self.bucket_name}.{self.endpoint}/{object_name}"
            else:
                logger.error(f"OSS上传失败，状态码: {result.status}")
                return None
                
        except Exception as e:
            logger.error(f"上传图片到OSS失败: {e}")
            return None

# 全局上传器实例
uploader = OSSUploader()

def upload_to_oss(image_url):
    """
    上传图片到OSS的全局函数
    :param image_url: 图片URL
    :return: OSS访问URL
    """
    return uploader.upload_to_oss(image_url)