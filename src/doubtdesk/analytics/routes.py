from fastapi import APIRouter, Depends, Query
from doubtdesk.analytics.auth import require_admin_key
from doubtdesk.analytics.queries import (
    get_usage_overview,
    get_request_usage,
    get_daily_usage,
)

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(require_admin_key)],
)


@router.get("/health")
def admin_health():
    return {"status": "ok"}

@router.get("/overview")
def admin_overview(
    days: int | None = Query(default=None, ge=1, le=365),
):
    return get_usage_overview(days=days)

@router.get("/usage/requests")
def admin_request_usage(
    days: int | None = Query(default=None, ge=1, le=365),
    limit: int = Query(default=20, ge=1, le=100),
    page: int = Query(default=1, ge=1),
):
    return get_request_usage(
        days=days,
        limit=limit,
        page=page,
    )

@router.get("/usage/daily")
def admin_daily_usage(
    days: int = Query(default=7, ge=1, le=365),
):
    return {
        "days": days,
        "data": get_daily_usage(days=days),
    }