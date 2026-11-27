import re
from functools import lru_cache
from doubtdesk.config import settings


def _object_key(doc_id: str, page_number: int) -> str:
    return f"page_images/{doc_id}/page_{page_number:03d}.png"


def _local_path(doc_id: str, page_number: int):
    return settings.page_image_dir / doc_id / f"page_{page_number:03d}.png"


@lru_cache
def _s3():
    if not settings.s3_bucket_name:
        raise RuntimeError("S3_BUCKET_NAME must be set when STORAGE_BACKEND=s3")
    import boto3
    return boto3.client("s3", region_name=settings.aws_region or None)


def _bucket_name() -> str:
    _s3()
    return settings.s3_bucket_name


def _is_missing(error) -> bool:
    return error.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}


def save_page_image(doc_id: str, page_number: int, data: bytes) -> None:
    if settings.storage_backend == "s3":
        _s3().put_object(
            Bucket=_bucket_name(),
            Key=_object_key(doc_id, page_number),
            Body=data,
            ContentType="image/png",
        )
        return

    path = _local_path(doc_id, page_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


_PAGE_FILE = re.compile(r"page_(\d+)\.png$")


def existing_page_numbers(doc_id: str) -> set[int]:
    if settings.storage_backend == "s3":
        paginator = _s3().get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=_bucket_name(), Prefix=f"page_images/{doc_id}/")
        names = (item["Key"] for page in pages for item in page.get("Contents", []))
    else:
        directory = settings.page_image_dir / doc_id
        if not directory.is_dir():
            return set()
        names = (path.name for path in directory.iterdir())
    return {int(match.group(1)) for name in names if (match := _PAGE_FILE.search(name))}


def page_image_exists(doc_id: str, page_number: int) -> bool:
    if settings.storage_backend == "s3":
        from botocore.exceptions import ClientError
        try:
            _s3().head_object(Bucket=_bucket_name(), Key=_object_key(doc_id, page_number))
        except ClientError as error:
            if _is_missing(error):
                return False
            raise
        return True
    return _local_path(doc_id, page_number).is_file()


def read_page_image(doc_id: str, page_number: int) -> bytes:
    if settings.storage_backend == "s3":
        from botocore.exceptions import ClientError
        key = _object_key(doc_id, page_number)
        try:
            response = _s3().get_object(Bucket=_bucket_name(), Key=key)
        except ClientError as error:
            if _is_missing(error):
                raise FileNotFoundError(f"Page image not found: {key}") from error
            raise
        return response["Body"].read()

    path = _local_path(doc_id, page_number)
    if not path.is_file():
        raise FileNotFoundError(f"Page image not found: {path}")
    return path.read_bytes()
