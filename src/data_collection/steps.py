"""
Steps of the pipeline.
"""

from urllib.parse import urlparse
from loguru import logger
from zenml import get_step_context, step
from tqdm import tqdm  # type:ignore
from pymongo import MongoClient

from src.data_collection.crawlers.dispatcher import select_crawler
from src.data_collection.crawlers.crawlers import LinkedInCrawler, MediumCrawler


CLIENT: MongoClient = MongoClient("localhost", 27017)
DB = CLIENT.antonio
PEOPLE = DB.people
DOCUMENTS = DB.documents


@step
def get_or_create_user(user_full_name: str) -> str:
    """
    Tries to create an user in the db. If it already exists, returns the found one.
    """

    logger.info(f"Getting or creating user: {user_full_name}")
    first_name, last_name = _split_user_full_name(user_full_name)
    _get_or_create(first_name=first_name, last_name=last_name)
    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="output",
        metadata=_get_metadata_user(
            user_full_name, first_name, last_name
        ),  # type:ignore
    )
    return user_full_name


def _split_user_full_name(user: str | None) -> tuple[str, str]:
    """
    Splits the name into name and surname.
    """

    if user is None:
        raise ValueError("User name is empty")

    name_tokens = user.split(" ")
    if len(name_tokens) == 0:
        raise ValueError("User name is empty")
    if len(name_tokens) == 1:
        first_name, last_name = name_tokens[0], ""
    else:
        first_name, last_name = name_tokens[0], " ".join(name_tokens[1:])

    return first_name, last_name


def _get_metadata_user(
    user_full_name: str, first_name: str, last_name: str
) -> dict[str, dict[str, str]]:
    """
    a
    """

    return {
        "query": {
            "user_full_name": user_full_name,
        },
        "retrieved": {
            "first_name": first_name,
            "last_name": last_name,
        },
    }


def _get_or_create(first_name: str, last_name: str) -> None:
    """
    a
    """

    instance = PEOPLE.find_one({"first_name": first_name, "last_name": last_name})
    if instance:
        logger.info(f"User: {first_name} {last_name} already in the DB.")
    else:
        PEOPLE.insert_one({"first_name": first_name, "last_name": last_name})
        logger.info(f"User created correctly: {first_name} {last_name}.")


@step
def crawl_links(user: str, links: list[str]) -> list[str]:
    """
    Obtains information from all links provided.
    """

    logger.info(f"Starting to crawl {len(links)} link(s).")
    metadata: dict[str, dict[str, int]] = {}
    successful_crawls = 0

    for link in tqdm(links):
        crawler = select_crawler(link)
        successful_crawl, crawled_domain = crawler.extract(link)
        successful_crawls += successful_crawl
        metadata = _get_metadata_links(metadata, crawled_domain, successful_crawl)

    step_context = get_step_context()
    step_context.add_output_metadata(
        output_name="crawled_links", metadata=metadata  # type: ignore
    )
    logger.info(f"Successfully crawled {successful_crawls} / {len(links)} links.")

    return links


def _get_metadata_links(metadata: dict, domain: str, successful_crawl: bool) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = (
        metadata.get(domain, {}).get("successful", 0) + successful_crawl
    )
    metadata[domain]["total"] = metadata.get(domain, {}).get("total", 0) + 1
    return metadata
