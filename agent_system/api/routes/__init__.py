"""
API 路由模块

包含所有 API 路由定义
"""

from . import knowledge_base
from . import documents
from . import agent
from . import mcp
from . import knowledge_base_v2  # 新 RAG 架构

__all__ = [
    "knowledge_base",
    "documents",
    "agent",
    "mcp",
    "knowledge_base_v2",
]

