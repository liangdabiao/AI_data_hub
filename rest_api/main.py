from fastapi import FastAPI
from dotenv import load_dotenv

import asyncio
import os
import sys
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将项目根目录添加到 sys.path
sys.path.insert(0, project_root)
# 禁用 langsmith 相关追踪
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_DEBUG"] = "false"
os.environ["SSL_CERT_FILE"] = r"D:\cacert.pem"

from config.config_loader import ConfigLoader
#from rag_graphs.news_rag_graph.ingestion import DocumentSyncManager
from rag_graphs.gzh_rag_graph.ingestion import DocumentSyncManager
from rest_api.routes import stock_routes, news_routes, xhs_routes, gzh_routes
from utils.logger import logger
from scraper.scraper_factory import StockScraperFactory, NewsScraperFactory,XHSScraperFactory,GZHScraperFactory
from datetime import datetime

# 小红书 微信公众号相关 数据处理

# Load .env
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize ConfigLoader
config_loader = ConfigLoader(config_file="config/config.json")

# Load configurations
SCRAPE_TICKERS = config_loader.get("SCRAPE_TICKERS")
XHS_SCRAPE_TICKERS = config_loader.get("XHS_SCRAPE_TICKERS")
GZH_SCRAPE_TICKERS = config_loader.get("GZH_SCRAPE_TICKERS")
SCRAPING_INTERVAL = config_loader.get("SCRAPING_INTERVAL", 3600)

if not GZH_SCRAPE_TICKERS:
    raise ValueError("No tickers found in config.json. Please check the configuration.")

async def run_scrapers_in_background():
    """
    Run news_scraper and stock_scraper in parallel in the background.
    """
    logger.info( "Starting scraping at run_scrapers_in_background")
    loop = asyncio.get_event_loop()

    # stock_factory = StockScraperFactory()
    # stock_scraper = stock_factory.create_scraper()

    # news_factory = NewsScraperFactory()
    # news_scraper = news_factory.create_scraper(collection_name=os.getenv("COLLECTION_NAME"),
    #                                            scrape_num_articles=int(os.getenv("SCRAPE_NUM_ARTICLES", 1)))

    # # Run both scrapers concurrently
    # await asyncio.gather(
    #     loop.run_in_executor(None, news_scraper.scrape_all_tickers, SCRAPE_TICKERS),
    #     loop.run_in_executor(None, stock_scraper.scrape_all_tickers, SCRAPE_TICKERS)
    # )
    # DocumentSyncManager().sync_documents()

    xhs_factory = XHSScraperFactory()
    xhs_scraper = xhs_factory.create_scraper()
    logger.info( "Starting scraping at xhs_scraper")
    gzh_factory = GZHScraperFactory()
    gzh_scraper = gzh_factory.create_scraper()
    logger.info( "Starting scraping at gzh_scraper")
    # Run both scrapers concurrently
    await asyncio.gather(
        loop.run_in_executor(None, xhs_scraper.scrape_all_tickers, XHS_SCRAPE_TICKERS),
        loop.run_in_executor(None, gzh_scraper.scrape_all_tickers, GZH_SCRAPE_TICKERS)
    )
    # Sync scraped docs in Vector DB  微信文章
    DocumentSyncManager().sync_documents()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event: Start scraping task
    logger.info("Starting scraping at application startup")
    asyncio.create_task(scrape_in_interval(SCRAPING_INTERVAL))
    yield
    # Shutdown event (optional cleanup logic)
    logger.info("Application is shutting down")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

async def scrape_in_interval(interval: int):
    """
    Runs the scraping task at regular intervals.
    """
    while True:
        logger.info(f"Starting scraping at {datetime.now()}")

        # Run scrapers in parallel
        await run_scrapers_in_background()

        hours   = interval / 3600  # Convert seconds to hours
        logger.info(f"Scraping completed at {datetime.now()}. Next run in {hours:.2f} hours.")
        # Wait for the specified interval
        await asyncio.sleep(interval)


# Include routes
# app.include_router(stock_routes.router, prefix="/stock", tags=["Stock Data"])
# app.include_router(news_routes.router, prefix="/news", tags=["News Articles"])
app.include_router(xhs_routes.router, prefix="/xhs", tags=["xhs Data"])
app.include_router(gzh_routes.router, prefix="/gzh", tags=["gzh Data"])

@app.get("/")
def root():
 return {"message": "Welcome to the Financial Data API"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)