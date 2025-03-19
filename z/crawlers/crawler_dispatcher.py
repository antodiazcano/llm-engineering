"""
This Crawler decides what Crawler to assign to a specific link.
"""

import re
from urllib.parse import urlparse
from loguru import logger

from src.data_collection.crawlers.crawler_linkedin import LinkedInCrawler
from src.data_collection.crawlers.crawler_default import DefaultCrawler


class CrawlerDispatcher:
    """
    This Crawler decides what Crawler to assign to a specific link.
    """

    def __init__(self) -> None:
        self._crawlers = {}

    def register_linkedin(self) -> "CrawlerDispatcher":
        self.register("https://linkedin.com", LinkedInCrawler)

        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc

        self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        for pattern, crawler in self._crawlers.items():
            if re.match(pattern, url):
                return crawler()
        else:
            logger.warning(
                f"No crawler found for {url}. Defaulting to CustomArticleCrawler."
            )

            return CustomArticleCrawler()
