"""
Handler 模块

导入所有 Handler 以触发装饰器自动注册

注意：
- tool_call 和 complex_task 意图都由 ReActHandler 处理
- DirectToolHandler 已弃用，保留文件但不再使用
"""

from .base import BaseHandler
from .chat import ChatHandler
from .knowledge import KnowledgeQAHandler
from .react import ReActHandler

# DirectToolHandler 已弃用，tool_call 意图现在由 ReActHandler 处理
# 不再导入以避免注册冲突
# from .tool import DirectToolHandler

__all__ = [
    "BaseHandler",
    "ChatHandler",
    "KnowledgeQAHandler",
    "ReActHandler",
]
