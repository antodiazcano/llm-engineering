"""
Class to decide the Crawler to use.
"""

from src.data_collection.crawlers.crawlers import MediumCrawler


def select_crawler(
    user: str, link: str, scroll_limit: int
) -> tuple[str, MediumCrawler]:
    """
    Given the link, selects the type of Crawler used to scrape the text.
    """

    if "medium.com" in link:
        return "medium", MediumCrawler(user, link, scroll_limit=scroll_limit)
    raise ValueError(f"Link must be from LinkedIn or Medium, got {link}.")
