"""
RAG (检索增强生成) 模块

提供向量数据库、文档处理、知识库管理等功能

模块结构：
- embeddings: 向量嵌入（现有）
- vector_store: 向量存储（ChromaDB，现有）
- document_processor: 文档处理（现有）
- knowledge_base: 知识库管理（现有）
- chunkers: 文档分块器（预处理层）
- extractors: 元数据提取器（预处理层）
- embedding: 向量化层
- stores: 存储层（Milvus/ChromaDB + BM25）
- retrieval: 检索层（混合检索 + 重排序）
"""

from .embeddings import QwenEmbeddings
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor
from .knowledge_base import KnowledgeBase

# 预处理层组件
from .chunkers import ChunkerBase, LegalChunker, RecursiveChunker
from .extractors import ExtractorBase, LegalMetadataExtractor

# 向量化层组件
from .embedding import (
    EmbeddingService,
    EmbeddingCache,
    BatchProcessor,
    AsyncBatcher,
    EmbeddingPipeline,
)

# 存储层组件
from .stores import (
    VectorStoreBase,
    SearchResult,
    BM25Index,
    BM25Result,
    StoreProvider,
    create_vector_store,
    create_bm25_index,
    HybridStore,
    get_collection_name,
    validate_user_id,
)

# 检索层组件
from .retrieval import (
    QueryParser,
    ParsedQuery,
    HybridRetriever,
    RetrievalResult,
    Reranker,
    RerankerType,
    RetrievalPipeline,
    RetrievalConfig,
)

__all__ = [
    # 现有组件
    "QwenEmbeddings",
    "VectorStoreManager",
    "DocumentProcessor",
    "KnowledgeBase",
    
    # 预处理层 - 分块器
    "ChunkerBase",
    "LegalChunker",
    "RecursiveChunker",
    
    # 预处理层 - 元数据提取器
    "ExtractorBase",
    "LegalMetadataExtractor",
    
    # 向量化层
    "EmbeddingService",
    "EmbeddingCache",
    "BatchProcessor",
    "AsyncBatcher",
    "EmbeddingPipeline",
    
    # 存储层
    "VectorStoreBase",
    "SearchResult",
    "BM25Index",
    "BM25Result",
    "StoreProvider",
    "create_vector_store",
    "create_bm25_index",
    "HybridStore",
    "get_collection_name",
    "validate_user_id",
    
    # 检索层
    "QueryParser",
    "ParsedQuery",
    "HybridRetriever",
    "RetrievalResult",
    "Reranker",
    "RerankerType",
    "RetrievalPipeline",
    "RetrievalConfig",
]



