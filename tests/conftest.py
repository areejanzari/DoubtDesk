import os

os.environ["DATABASE_URL"] = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://doubtdesk:doubtdesk@127.0.0.1:5432/doubtdesk_test",
)

import pytest

from doubtdesk.db import close_pool, initialize_database
from tests.db_support import ensure_database, reset_database


@pytest.fixture(scope="session", autouse=True)
def database():
    ensure_database(os.environ["DATABASE_URL"])
    initialize_database(timeout=10)
    yield
    close_pool()


@pytest.fixture(autouse=True)
def clean_database(database):
    reset_database(os.environ["DATABASE_URL"])
