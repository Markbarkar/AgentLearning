"""
分块器模块

提供文档分块功能，支持多种分块策略：
- RecursiveChunker: 递归字符分块（通用）
- LegalChunker: 法律文档结构分块
- SemanticChunker: 语义相似度分块
"""

from .base import ChunkerBase
from .legal_chunker import LegalChunker
from .recursive_chunker import RecursiveChunker
from .semantic_chunker import SemanticChunker

__all__ = [
    "ChunkerBase",
    "LegalChunker",
    "RecursiveChunker",
    "SemanticChunker",
]
