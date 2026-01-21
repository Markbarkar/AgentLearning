"""
存储层模块

提供向量存储和全文索引的统一接口：
- VectorStoreBase: 向量存储抽象基类
- MilvusStore: Milvus 向量存储适配器
- ChromaStore: ChromaDB 适配器（兼容现有）
- BM25Index: BM25 全文索引
- HybridStore: 混合存储（向量 + BM25）
"""

from .base import VectorStoreBase, SearchResult
from .user_isolation import UserIsolationMixin, get_collection_name, validate_user_id
from .bm25_index import BM25Index, BM25Result
from .factory import (
    StoreProvider,
    create_vector_store,
    create_bm25_index,
    HybridStore,
)

# 延迟导入具体存储实现（可能有依赖缺失）
def get_milvus_store():
    from .milvus_store import MilvusStore
    return MilvusStore

def get_chroma_store():
    from .chroma_store import ChromaStore
    return ChromaStore

__all__ = [
    # 基类
    "VectorStoreBase",
    "SearchResult",
    
    # 用户隔离
    "UserIsolationMixin",
    "get_collection_name",
    "validate_user_id",
    
    # BM25 索引
    "BM25Index",
    "BM25Result",
    
    # 工厂
    "StoreProvider",
    "create_vector_store",
    "create_bm25_index",
    "HybridStore",
    
    # 延迟导入函数
    "get_milvus_store",
    "get_chroma_store",
]
