"""
Steps of the pipeline.
"""

from typing import Any
from loguru import logger
from zenml import get_step_context, step
from tqdm import tqdm  # type:ignore
from pymongo import MongoClient

from src.data_collection.crawlers.dispatcher import select_crawler


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
    metadata = _get_metadata_user(user_full_name, first_name, last_name)
    step_context.add_output_metadata(metadata=metadata)  # type: ignore
    return user_full_name


def _split_user_full_name(user: str) -> tuple[str, str]:
    """
    Splits the name into name and surname.
    """

    name_tokens = user.split(" ")
    if len(name_tokens) == 0:
        return "Unknown", "Unknown"
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
def crawl_links(user: str, links: list[str], scroll_limit: int = 10) -> list[str]:
    """
    Obtains information from all links provided.
    """

    logger.info(f"Starting to crawl {len(links)} link(s).")
    metadata: dict[str, dict[str, int]] = {}
    successful_crawls = 0

    for link in tqdm(links):
        logger.info(f"Starting scrapping article: {link}")
        domain, crawler = select_crawler(user, link, scroll_limit=scroll_limit)
        successful_crawl, crawled_domain = crawler.extract(link)
        successful_crawls += successful_crawl
        if successful_crawl:
            _crawl_link(user, link, crawled_domain)
        metadata = _get_metadata_links(
            metadata,  # type: ignore
            domain,  # type: ignore
            successful_crawl,
        )

    step_context = get_step_context()
    step_context.add_output_metadata(
        metadata=metadata,  # type: ignore
    )
    logger.info(f"Successfully crawled {successful_crawls} / {len(links)} links.")

    return links


def _crawl_link(user: str, link: str, crawled_domain: dict[str, str | Any]) -> None:
    instance = DOCUMENTS.find_one({"link": link})
    print(f"Instance: {instance}")
    if instance:
        print(":(")
        logger.info(f"Link: {link}, of {user} already in the DB.")
    else:
        print(":)")
        DOCUMENTS.insert_one(crawled_domain)
        logger.info(f"Link: {link}, of {user}.")


def _get_metadata_links(metadata: dict, domain: str, successful_crawl: bool) -> dict:
    if domain not in metadata:
        metadata[domain] = {}
    metadata[domain]["successful"] = (
        metadata.get(domain, {}).get("successful", 0) + successful_crawl
    )
    metadata[domain]["total"] = metadata.get(domain, {}).get("total", 0) + 1
    return metadata
