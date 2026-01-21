"""
向量化层模块

提供统一的嵌入服务接口，支持：
- 多种嵌入模型（Qwen、OpenAI 等）
- 异步批处理
- 嵌入缓存
- 与不同分块器的灵活组合
"""

from .service import EmbeddingService
from .batcher import AsyncBatcher, BatchProcessor
from .cache import EmbeddingCache
from .pipeline import EmbeddingPipeline

__all__ = [
    "EmbeddingService",
    "AsyncBatcher",
    "BatchProcessor",
    "EmbeddingCache",
    "EmbeddingPipeline",
]
