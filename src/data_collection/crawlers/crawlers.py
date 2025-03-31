"""
This script contains all the classes for the Crawlers of different platforms.
"""

from typing import Any
import time
from bs4 import BeautifulSoup
import numpy as np
from selenium import webdriver


class BaseCrawler:
    """
    Base class of the Crawlers.
    """

    def __init__(self, author: str, url: str, scroll_limit: int = 10) -> None:
        """
        Constructor of the class. It has some commented options, which are the standard
        ones, but sometimes they give problems.

        Parameters
        ----------
        author       : Author to scrape.
        url          : Url to scrape.
        scroll_limit : Scroll limit for scraping.
        """

        options = webdriver.ChromeOptions()
        # options.add_argument("--no-sandbox")
        # options.add_argument("--headless=new")
        # options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--log-level=3")
        # options.add_argument("--disable-popup-blocking")
        # options.add_argument("--disable-notifications")
        # options.add_argument("--disable-extensions")
        # options.add_argument("--disable-background-networking")
        # options.add_argument("--ignore-certificate-errors")
        # options.add_argument(f"--user-data-dir={mkdtemp()}")
        # options.add_argument(f"--data-path={mkdtemp()}")
        # options.add_argument(f"--disk-cache-dir={mkdtemp()}")
        # options.add_argument("--remote-debugging-port=9226")
        # options.add_argument(r"--profile-directory=Profile 2")

        self.author = author
        self.url = url
        self.scroll_limit = scroll_limit
        self.driver = webdriver.Chrome(
            options=options,
        )

    def scroll_page(self) -> None:
        """
        Scroll through the LinkedIn page based on the scroll limit.
        """

        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            self.wait()
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or (
                self.scroll_limit and current_scroll >= self.scroll_limit
            ):
                break
            last_height = new_height
            current_scroll += 1

    def wait(self) -> None:
        """
        Waits a random time simulating a person.
        """

        time.sleep(max(1 + np.random.uniform(1), np.random.normal(2, 2)))


class MediumCrawler(BaseCrawler):
    """
    Specific Crawler for Medium articles.
    """

    def __init__(self, author: str, url: str, scroll_limit: int = 10) -> None:
        """
        Constructor of the class. It has some commented options, which are the standard
        ones, but sometimes they give problems.

        Parameters
        ----------
        author       : Author to scrape.
        url          : Url to scrape.
        scroll_limit : Scroll limit for scraping.
        """

        super().__init__(author, url, scroll_limit=scroll_limit)

    def extract(self) -> tuple[bool, dict[str, str | Any]]:
        """
        Tries to scrape the given link.

        Returns
        -------
        If the scraping was successful or not and dict with its data.
        """

        try:
            self.driver.get(self.url)
            self.scroll_page()

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            title = soup.find_all("h1", class_="pw-post-title")
            subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

            data = {
                "Title": title[0].string if title else None,  # type: ignore
                "Subtitle": subtitle[0].string if subtitle else None,  # type: ignore
                "Content": soup.get_text(),
            }

            self.driver.close()

            return True, {
                "platform": "medium",
                "content": data,
                "link": self.url,
                "author": self.author,
            }
        except Exception as e:
            print(e)
            return False, {}
