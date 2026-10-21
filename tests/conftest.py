import os

import pytest

os.environ.setdefault("FIELDOPS_OFFLINE", "true")

from capstone.data.loader import ensure_world  # noqa: E402

ensure_world()


@pytest.fixture
def crm():
    from capstone.core.crm import CRMRepository

    return CRMRepository()


@pytest.fixture
def known_customer(crm):
    from capstone.data.loader import load_table

    return next(c for c in load_table("customers") if c["tenant_id"] == "t1")
