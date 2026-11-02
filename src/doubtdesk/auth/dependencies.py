import os
import secrets
from fastapi import Header, HTTPException, Request, status
from doubtdesk.auth.store import get_session


def require_current_user(
    request: Request,
    x_csrf_token: str | None = Header(default=None),
) -> dict:
    session_token = request.cookies.get("doubtdesk_session")

    if not session_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
        )

    user = get_session(session_token)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
        )

    if request.method not in {"GET", "HEAD", "OPTIONS"}:
        
        origin = request.headers.get("Origin")

        if origin is not None and origin not in allowed_origins():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Request origin not allowed.",
            )
        
        if not x_csrf_token or not secrets.compare_digest(
            x_csrf_token,
            user["csrf_token"],
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid CSRF token.",
            )

    return user


def allowed_origins() -> set[str]:
    configured = os.getenv("AUTH_ALLOWED_ORIGINS")

    if configured:
        return {
            origin.strip().rstrip("/")
            for origin in configured.split(",")
            if origin.strip()
        }

    if os.getenv("APP_ENV", "development") == "production":
        raise RuntimeError("AUTH_ALLOWED_ORIGINS must be configured.")

    return {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    }
