"""
Class to decide the Crawler to use.
"""

from src.data_collection.crawlers.crawlers import LinkedInCrawler, MediumCrawler


def select_crawler(link: str) -> LinkedInCrawler | MediumCrawler:
    """
    Given the link, selects the type of Crawler used to scrape the text.
    """

    if "www.linkedin.com" in link:
        return LinkedInCrawler(link)
    if "medium.com" in link:
        return MediumCrawler(link)
    raise ValueError(f"Link must be from LinkedIn or Medium, got {link}.")
