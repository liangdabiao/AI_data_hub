import os
import requests
from time import sleep
from dotenv import load_dotenv
from db.mongo_db import MongoDBClient
from db.postgres_db import PostgresDBClient
from scraper.generic_scraper import GenericScraper
from utils.logger import logger

## 小红书爬虫

load_dotenv()

class XHSscraper(GenericScraper):
    def __init__(self, collection_name, scrape_num_pages=1):
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {os.getenv("XHS_AUTH_TOKEN")}'
        }
        self.collection_name = collection_name
        self.scrape_num_pages = scrape_num_pages
        self.mongo_client = MongoDBClient()
        # 初始化Postgres客户端
        user = os.getenv("POSTGRES_USERNAME")
        password = os.getenv("POSTGRES_PASSWORD")
        host = os.getenv("POSTGRES_HOST")
        port = os.getenv("POSTGRES_PORT", 5432)
        db_name = os.getenv("POSTGRES_DB")
        self.postgres_client = PostgresDBClient(
            host=host,
            database=db_name,
            user=user,
            password=password,
            port=port,
        )

    def extract_note(self, item):
        note_card = item.get("note_card", {})
        user_info = note_card.get("user", {})
        interact_info = note_card.get("interact_info", {})
        cover_info = note_card.get("cover", {})
        image_list = note_card.get("image_list", [])  # 修正默认值为列表
        # 提取image_list中image_scene为WB_PRV的图片URL
        image_arr = []
        for img in image_list:
            info_list = img.get("info_list", [])
            prv_url = next((info.get("url") for info in info_list if info.get("image_scene") == "WB_PRV"), None)
            if prv_url:
                image_arr.append(prv_url)
        publish_time_str = next((tag.get("text") for tag in note_card.get("corner_tag_info", []) if tag.get("type") == "publish_time"), None)
        publish_time = self._parse_publish_time(publish_time_str) if publish_time_str else None

        # 上传图片到阿里云OSS
        image_arr_new = []
        if image_arr:
            from utils.oss_uploader import upload_to_oss
            for img_url in image_arr:
                try:
                    print("上传图片到OSS:")
                    print(img_url)
                    oss_url = upload_to_oss(img_url)
                    print(oss_url)
                    image_arr_new.append(oss_url)
                    
                except Exception as e:
                    logger.error(f"上传图片到OSS失败: {e}")
                    #image_arr_new.append(img_url)  # 失败时保留原URL
        
        return {
            'note_id': item.get("id"),
            'title': note_card.get("display_title"),
            'user_nickname': user_info.get("nick_name") or user_info.get("nickname"),
            'user_avatar': user_info.get("avatar"),
            'user_id': user_info.get("user_id"),
            'liked_count': interact_info.get("liked_count"),
            'collected_count': interact_info.get("collected_count"),
            'comment_count': interact_info.get("comment_count"),
            'shared_count': interact_info.get("shared_count"),
            'cover_url': cover_info.get("url_default"),
            'publish_time': publish_time,
            'image_arr': image_arr,
            'image_arr_new': image_arr_new,
            'synced': False
        }

    def create_xhs_notes_table(self):
        """
        创建小红书笔记表（如果不存在），字段与note对象完全匹配
        """
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS xhs_notes (
            note_id TEXT PRIMARY KEY,  -- 笔记ID（主键）
            title TEXT,                -- 笔记标题
            user_nickname TEXT,        -- 用户昵称
            user_avatar TEXT,          -- 用户头像URL
            user_id TEXT,              -- 用户ID
            liked_count INTEGER,       -- 点赞数
            collected_count INTEGER,   -- 收藏数
            comment_count INTEGER,     -- 评论数
            shared_count INTEGER,      -- 分享数
            cover_url TEXT,            -- 封面图URL
            publish_time TEXT,         -- 发布时间
            image_arr TEXT[],          -- 图片数组（存储多个图片URL）
            image_arr_new TEXT[],          -- 图片数组（存储多个图片URL）
            synced BOOLEAN DEFAULT FALSE  -- 是否同步标记（默认未同步）
        );
        '''
        self.postgres_client.execute_query(create_table_sql)

    def scrape_notes(self, search_keyword):
        from urllib.parse import quote
        template = 'https://api.tikhub.io/api/v1/xiaohongshu/web_v2/fetch_search_notes?keywords={}&page={}&sort_type=general&note_type=0'
        notes = []
        note_ids = set()
        current_page = 1

        while current_page <= self.scrape_num_pages:
            encoded_keyword = quote(search_keyword)
            url = template.format(encoded_keyword, current_page)
            try:
                logger.info(f"请求URL: {url}")
                response = requests.get(url, headers=self.headers, proxies={"http": None, "https": None})
                response.raise_for_status()
                data = response.json()
                if data.get("code") != 200:
                    logger.error(f"API请求失败，状态码：{data.get('code')}")
                    break

                items = data.get("data", {}).get("items", [])
                for item in items:
                    note = self.extract_note(item)
                    if note['note_id'] not in note_ids:
                        note_ids.add(note['note_id'])
                        if note.get('title'):  # 检查title是否非空
                            notes.append(note)

                if not data.get("data", {}).get("has_more"):
                    break
                current_page += 1
                sleep(1)
            except Exception as e:
                logger.error(f"爬取失败，关键词：{search_keyword}，URL：{url}，错误：{str(e)}", exc_info=True)
                break

        if notes:
            self.mongo_client.insert_many(self.collection_name, notes)
            logger.info(f"插入{len(notes)}条小红书笔记到MongoDB。")
            # 创建表（如果不存在）
            self.create_xhs_notes_table()
            # 插入数据到Postgres（过滤MongoDB的'_id'字段）
            for note in notes:
                # 移除MongoDB自动生成的'_id'字段（ObjectId类型Postgres无法处理）
                note.pop('_id', None)
                # 构造包含ON CONFLICT的插入语句（忽略重复note_id）
                columns = note.keys()
                placeholders = ", ".join(["%s"] * len(columns))
                sql = f"""
                INSERT INTO xhs_notes ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (note_id) DO NOTHING;
                """
                self.postgres_client.execute_query(sql, list(note.values()))
            logger.info(f"插入{len(notes)}条小红书笔记到PostgresDB。")
        return notes

    def _parse_publish_time(self, time_str):
        """
        将不同格式的发布时间字符串转换为时间戳
        :param time_str: 时间字符串 (2024-09-01 / 05-13 / 12小时前)
        :return: 时间戳 (int)
        """
        from datetime import datetime, timedelta
        import re
        
        try:
            # 处理完整日期格式 (2024-09-01)
            if re.match(r'\d{4}-\d{2}-\d{2}', time_str):
                return int(datetime.strptime(time_str, '%Y-%m-%d').timestamp())
            
            # 处理月-日格式 (05-13)，假设为当前年
            elif re.match(r'\d{2}-\d{2}', time_str):
                current_year = datetime.now().year
                return int(datetime.strptime(f"{current_year}-{time_str}", '%Y-%m-%d').timestamp())
            
            # 处理相对时间 (12小时前)
            elif "小时前" in time_str:
                hours = int(re.search(r'(\d+)', time_str).group(1))
                return int((datetime.now() - timedelta(hours=hours)).timestamp())
            
            # 处理其他未知格式
            return int(datetime.now().timestamp())
        except Exception as e:
            logger.error(f"时间格式转换失败: {time_str}, 错误: {e}")
            return int(datetime.now().timestamp())
            
    def scrape_all_tickers(self, keywords):
        for keyword in keywords:
            logger.info(f"爬取小红书笔记，关键词：{keyword}")
            try:
                self.scrape_notes(keyword)
            except Exception as e:
                logger.error(f"爬取关键词{keyword}时出错：{e}")


if __name__ == "__main__":
    scraper = XHSscraper(
        collection_name=os.getenv("XHS_COLLECTION_NAME"),
        scrape_num_pages=int(os.getenv("XHS_SCRAPE_NUM_PAGES", 1))
    )

    target_keywords = [ "moc大神"]  # 示例关键词列表
    scraper.scrape_all_tickers(target_keywords)