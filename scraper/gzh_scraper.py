import os
import requests
import datetime
from time import sleep
from dotenv import load_dotenv
from db.mongo_db import MongoDBClient
from db.postgres_db import PostgresDBClient
from scraper.generic_scraper import GenericScraper
from utils.logger import logger
from urllib.parse import quote

## 公众号爬虫

load_dotenv()

class GZHscraper(GenericScraper):
    def __init__(self, collection_name, ghid, scrape_num_pages=1):
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {os.getenv("GZH_AUTH_TOKEN")}'
        }
        self.collection_name = collection_name
        self.ghid = ghid  # 微信公众号ID
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
        
        return {
            'note_id': item.get("comment_topic_id"),  # 使用评论主题ID作为唯一标识
            'title': item.get("Title"),
            'digest': item.get("Digest"),  # 新增摘要字段
            'user_nickname': self.ghid,  # 固定为公众号来源
            'content_url': item.get("ContentUrl"),  # 文章链接
            'source_url': item.get("SourceUrl"),  # 原文链接
            'cover_url': item.get("CoverImgUrl"),
            'publish_time': datetime.datetime.fromtimestamp(item.get("send_time", 0)).strftime("%Y-%m-%d %H:%M:%S"),  # 时间戳转字符串
            'is_original': bool(item.get("IsOriginal", 0)),  # 是否原创（整数转布尔）
            'synced': False,
        
        }

    def create_gzh_notes_table(self):
        """
        创建公众号笔记表（如果不存在），字段与note对象完全匹配
        """
        create_table_sql = '''
        CREATE TABLE IF NOT EXISTS gzh_notes (
            note_id TEXT PRIMARY KEY,  -- 笔记ID（主键）
            title TEXT,                -- 笔记标题
            digest TEXT,               -- 文章摘要
            user_nickname TEXT,        -- 用户昵称（固定为微信公众号）
            content_url TEXT,          -- 文章链接
            source_url TEXT,           -- 原文链接
            cover_url TEXT,            -- 封面图URL
            publish_time TEXT,         -- 发布时间（格式化字符串）
            is_original BOOLEAN,       -- 是否原创
            synced BOOLEAN DEFAULT FALSE,  -- 是否同步标记（默认未同步）
            description TEXT           -- 新增文章描述字段
        );
        '''
        self.postgres_client.execute_query(create_table_sql)

    def scrape_notes(self):
        template = 'https://api.tikhub.io/api/v1/wechat_mp/web/fetch_mp_article_list?ghid={}&page={}'
        notes = []
        note_ids = set()
        current_page = 1

        while current_page <= self.scrape_num_pages:
            url = template.format(self.ghid, current_page)
            try:
                logger.info(f"请求URL：{url}")
                response = requests.get(url, headers=self.headers, proxies={"http": None, "https": None})
                response.raise_for_status()
                data = response.json()
                if data.get("code") != 200:
                    logger.error(f"API请求失败，状态码：{data.get('code')}，信息：{data.get('message')}")
                    break

                items = data.get("data", {}).get("list", [])
                for item in items:
                    note = self.extract_note(item)
                    if note['note_id'] and note['note_id'] not in note_ids:
                        note_ids.add(note['note_id'])
                        notes.append(note)

                # 根据新API调整分页判断（假设返回字段为has_next_page）
                if not data.get("data", {}).get("has_next_page", False):
                    break
                current_page += 1
                sleep(1)
            except Exception as e:
                logger.error(f"爬取失败，公众号ID：{self.ghid}，错误：{e}")
                break

        if notes:
            self.mongo_client.insert_many(self.collection_name, notes)
            logger.info(f"插入{len(notes)}条公众号笔记到MongoDB。")
            # 创建表（如果不存在）
            self.create_gzh_notes_table()
            # 插入数据到Postgres（过滤MongoDB的'_id'字段）
            for note in notes:
                # 移除MongoDB自动生成的'_id'字段（ObjectId类型Postgres无法处理）
                note.pop('_id', None)
                # 构造包含ON CONFLICT的插入语句（忽略重复note_id）
                columns = note.keys()
                placeholders = ", ".join(["%s"] * len(columns))
                sql = f"""
                INSERT INTO gzh_notes ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT (note_id) DO NOTHING;
                """
                self.postgres_client.execute_query(sql, list(note.values()))
            logger.info(f"插入{len(notes)}条公众号笔记到PostgresDB。")

        # 单独更新description字段
        for note in notes:
            # 检查是否已存在该note_id
            sql = "SELECT description FROM gzh_notes WHERE note_id = %s::TEXT LIMIT 1;"
            exists = self.postgres_client.execute_query(sql, (str(note['note_id']),))
            if exists and exists[0]['description']:
                continue
                
            content_url = note['content_url']
            if content_url:
                try:
                    logger.info( " content_url字段:")
                    logger.info(content_url)
                    encoded_content_url = quote(content_url)  # 对content_url进行URL编码
                    detail_url = f"https://api.tikhub.io/api/v1/wechat_mp/web/fetch_mp_article_detail_json?url={encoded_content_url}"
                    logger.info(f"请求detail_url URL：{detail_url}")
                    response = requests.get(detail_url, headers=self.headers, proxies={"http": None, "https": None})
                    response.raise_for_status()
                    detail_data = response.json() 
                    if detail_data.get("code") == 200:
                        logger.info( " description字段:")
                        description = detail_data.get("data", {}).get("content", {}).get("article", {}).get("full_text", '') 
                         
                        # 更新MongoDB记录
                        self.mongo_client.update_one(
                            self.collection_name,
                            {"note_id": note['note_id']},
                            {"$set": {"description": description}}
                        )
                        # 更新Postgres记录
                        update_sql = """
                        UPDATE gzh_notes
                        SET description = %s
                        WHERE note_id = %s;
                        """
                        self.postgres_client.execute_query(update_sql, (description, str(note['note_id'])))
                        logger.info(f"成功更新note_id {note['note_id']} 的description字段")
                except Exception as e:
                    logger.error(f"更新note_id {note['note_id']} 的description失败，错误：{e}")
        return notes

    def scrape_all_tickers(self, ghids):
        for ghid in ghids:
            logger.info(f"爬取微信公众号文章，GHID：{ghid}")
            self.ghid = ghid  # 设置当前公众号ID
            try:
                self.scrape_notes()
            except Exception as e:
                logger.error(f"爬取公众号{ghid}时出错：{e}")


if __name__ == "__main__":
    scraper = GZHscraper(
        collection_name=os.getenv("GZH_COLLECTION_NAME"),
        ghid=os.getenv("GZH_DEFAULT_GHID", "gh_f35fc487107d"),  # 默认GHID
        scrape_num_pages=int(os.getenv("GZH_SCRAPE_NUM_PAGES", 1))
    )

    target_ghids = ["gh_f35fc487107d"]  # 示例公众号GHID列表
    scraper.scrape_all_tickers(target_ghids)