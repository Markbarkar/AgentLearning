"""
元数据提取器模块

提供文档元数据提取功能
"""

from .base import ExtractorBase
from .legal_metadata import LegalMetadataExtractor

__all__ = [
    "ExtractorBase",
    "LegalMetadataExtractor",
]
