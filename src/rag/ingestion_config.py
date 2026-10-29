"""Constants for document ingestion."""

from enum import StrEnum

DEFAULT_START_PAGE = 1


class MetadataKey(StrEnum):
    FILE_NAME = "file_name"
    SOURCE_PATH = "source_path"
    SOURCE_ID = "source_id"
    PAGE_NUMBER = "page_number"
    SOURCE_LOCATION = "source_location"


class PageStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"
