"""
分类策略模块

提供可插拔的分类策略
"""

from .base import ClassifierStrategy, StrategyRegistry, register_strategy
from .rule import RuleClassifierStrategy
from .llm import LLMClassifierStrategy

__all__ = [
    "ClassifierStrategy",
    "StrategyRegistry",
    "register_strategy",
    "RuleClassifierStrategy",
    "LLMClassifierStrategy",
]
