"""
路由分发模块

提供意图路由和处理器管理功能
"""

from .registry import HandlerRegistry, register_handler
from .router import IntentRouter
from .factory import HandlerFactory

# 导入 handlers 模块以触发装饰器注册
from . import handlers

__all__ = [
    "HandlerRegistry",
    "register_handler",
    "IntentRouter",
    "HandlerFactory",
]
