"""
意图识别模块

提供意图分类和识别功能
"""

from .models import Intent, IntentConfig, load_intent_config
from .classifier import IntentClassifier

__all__ = [
    "Intent",
    "IntentConfig",
    "load_intent_config",
    "IntentClassifier",
]
