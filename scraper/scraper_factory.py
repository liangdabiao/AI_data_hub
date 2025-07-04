from abc import ABC, abstractmethod

from scraper.generic_scraper import GenericScraper
from scraper.news_scraper import NewsScraper
from scraper.stock_data_scraper import StockDataScraper
from scraper.xhs_scraper import XHSscraper
from scraper.gzh_scraper import GZHscraper
import os
class ScraperFactory(ABC):
    """
    Abstract factory interface for creating scraper instances.
    """
    @abstractmethod
    def create_scraper(self, **kwargs)->GenericScraper:
        pass

class StockScraperFactory(ScraperFactory):
    """
    Factory class for creating StockScraper instances.
    """
    def create_scraper(self, **kwargs)->StockDataScraper:
        """
        Create a StockScraper instance.
        """
        return StockDataScraper()

class NewsScraperFactory(ScraperFactory):
    """
    Factory class for creating NewsScraper instances.
    """
    def create_scraper(self, **kwargs)->NewsScraper:
        """
        Create a NewsScraper instance.
        """
        collection_name     = kwargs.get("collection_name", "default_collection")
        scrape_num_articles = kwargs.get("scrape_num_articles", 1)
        return NewsScraper(collection_name, scrape_num_articles)


class XHSScraperFactory(ScraperFactory):
    """
    Factory class for creating XHSscraper instances.
    """
    def create_scraper(self, **kwargs)->XHSscraper:
        """
        Create a XHSscraper instance.
        """
        collection_name     = kwargs.get("collection_name", os.getenv("XHS_COLLECTION_NAME"))
        scrape_num_pages = kwargs.get("scrape_num_pages", 1)
        return XHSscraper(collection_name, scrape_num_pages)


class GZHScraperFactory(ScraperFactory):
    """
    Factory class for creating GZHscraper instances.
    """
    def create_scraper(self, **kwargs)->GZHscraper:
        """
        Create a GZHscraper instance.
        """
        collection_name     = kwargs.get("collection_name", os.getenv("GZH_COLLECTION_NAME"))
        scrape_num_pages = kwargs.get("scrape_num_pages", 1)
        return GZHscraper(collection_name, scrape_num_pages)