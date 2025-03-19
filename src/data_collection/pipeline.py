"""
Pipeline of the data collection.
"""

from zenml import pipeline

from src.data_collection.steps import crawl_links, get_or_create_user


@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> None:
    """
    Pipeline of the data collection.
    """

    user_full_name = get_or_create_user(user_full_name)
    links = crawl_links(user=user_full_name, links=links)


if __name__ == "__main__":
    USER_FULL_NAME = "Maxime Labonne"
    LINKS = ["https://www.linkedin.com/in/maxime-labonne/"]
    digital_data_etl(USER_FULL_NAME, LINKS)
