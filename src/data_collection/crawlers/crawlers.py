from bs4 import BeautifulSoup  # type: ignore
import time
import numpy as np
from selenium import webdriver
from typing import Any


class BaseCrawler:
    def __init__(self, author: str, url: str, scroll_limit: int = 10) -> None:
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
        """Scroll through the LinkedIn page based on the scroll limit."""
        current_scroll = 0
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        while True:
            self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
            )
            self._wait()
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or (
                self.scroll_limit and current_scroll >= self.scroll_limit
            ):
                break
            last_height = new_height
            current_scroll += 1

    def _wait(self) -> None:
        time.sleep(max(1 + np.random.uniform(1), np.random.normal(2, 2)))


class MediumCrawler(BaseCrawler):
    def __init__(self, author: str, url: str, scroll_limit: int = 10) -> None:
        super().__init__(author, url, scroll_limit=scroll_limit)

    def extract(self, link: str) -> tuple[bool, dict[str, str | Any]]:
        try:
            self.driver.get(link)
            self.scroll_page()

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            title = soup.find_all("h1", class_="pw-post-title")
            subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

            data = {
                "Title": title[0].string if title else None,
                "Subtitle": subtitle[0].string if subtitle else None,
                "Content": soup.get_text(),
            }

            self.driver.close()

            return True, {
                "platfrom": "medium",
                "content": data,
                "link": link,
                "author": self.author,
            }
        except Exception as e:
            print(e)
            return False, {}
