from collections.abc import Callable
from typing import Any

from src.pdf_data_extractor.tools import classify_document


ToolFunction = Callable[..., dict[str, Any]]


TOOL_REGISTRY: dict[str, ToolFunction] = {
    "classify_document": classify_document,
}