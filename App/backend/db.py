import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient
MONGO_URL = os.environ.get("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = os.environ.get("DB_NAME", "test_database")

_client: Optional[AsyncIOMotorClient] = None # type: ignore


def get_client() -> AsyncIOMotorClient: # type: ignore
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(MONGO_URL)
    return _client


def get_db():
    client = get_client()
    return client[DB_NAME]


def close_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None