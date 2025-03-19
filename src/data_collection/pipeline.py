"""
Pipeline of the data collection.
"""

from zenml import pipeline

from src.data_collection.steps import get_or_create_user, crawl_links


@pipeline
def digital_data_etl(user_full_name: str, links: list[str]) -> None:
    """
    Pipeline of the data collection.
    """

    user_full_name = get_or_create_user(user_full_name)
    links = crawl_links(user=user_full_name, links=links)


if __name__ == "__main__":
    USER_FULL_NAME = "Paul Iusztin"
    LINKS = [
        "https://medium.com/decodingml/an-end-to-end-framework-for-"
        "production-ready-llm-systems-by-building-your-llm-twin-2cc6bb01141f"
    ]
    digital_data_etl(USER_FULL_NAME, LINKS)
