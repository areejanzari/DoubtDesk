import os
from secrets import compare_digest
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

admin_key_header = APIKeyHeader(
    name="X-Admin-Key",
    auto_error=False,
)


def require_admin_key(
    provided_key: str | None = Security(admin_key_header),
) -> None:
    expected_key = os.getenv("DOUBTDESK_ADMIN_KEY")

    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Admin API is not configured.",
        )

    if not provided_key or not compare_digest(
        provided_key,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin API key.",
        )