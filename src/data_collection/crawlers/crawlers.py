class LinkedInCrawler:
    def __init__(self, url: str) -> None:
        self.url = url

    def extract(self) -> dict[str, str]:
        """
        Scrapes the link.
        """

        return {}

    def save(self, data: dict[str, str]) -> None:
        """
        Saves the data into Mongo DB.
        """

        pass


class MediumCrawler:
    def __init__(self, url: str) -> None:
        self.url = url

    def extract(self) -> dict[str, str]:
        """
        Scrapes the link.
        """

        return {}

    def save(self, data: dict[str, str]) -> None:
        """
        Saves the data into Mongo DB.
        """

        pass
