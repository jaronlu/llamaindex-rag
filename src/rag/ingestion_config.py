"""Constants for document ingestion."""

from enum import StrEnum


class PageStatus(StrEnum):
    SUCCESS = "success"
    EMPTY = "empty"
    FAILED = "failed"
