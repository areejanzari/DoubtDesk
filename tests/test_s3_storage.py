import io
import pytest
from botocore.exceptions import ClientError
from doubtdesk.storage import images


class FakeS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body, ContentType):
        self.objects[(Bucket, Key)] = Body

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def get_paginator(self, name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                keys = [key for bucket, key in objects if bucket == Bucket and key.startswith(Prefix)]
                return [{"Contents": [{"Key": key} for key in keys]}] if keys else [{}]

        return Paginator()


@pytest.fixture
def s3(monkeypatch):
    fake = FakeS3()
    monkeypatch.setattr(images.settings, "storage_backend", "s3")
    monkeypatch.setattr(images.settings, "s3_bucket_name", "bucket")
    monkeypatch.setattr(images, "_s3", lambda: fake)
    return fake


def test_s3_round_trip(s3):
    images.save_page_image("doc", 2, b"png")
    assert s3.objects[("bucket", "page_images/doc/page_002.png")] == b"png"
    assert images.page_image_exists("doc", 2)
    assert not images.page_image_exists("doc", 3)
    assert images.read_page_image("doc", 2) == b"png"
    assert images.existing_page_numbers("doc") == {2}
    assert images.existing_page_numbers("other") == set()


def test_s3_missing_image_raises_file_not_found(s3):
    with pytest.raises(FileNotFoundError):
        images.read_page_image("doc", 1)


def test_s3_requires_bucket(monkeypatch):
    images._s3.cache_clear()
    monkeypatch.setattr(images.settings, "s3_bucket_name", "")
    with pytest.raises(RuntimeError, match="S3_BUCKET_NAME"):
        images._s3()
