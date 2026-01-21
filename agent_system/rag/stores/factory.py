"""
存储工厂模块

提供统一的存储创建接口
"""

from enum import Enum
from typing import Optional, Dict, Any, Union

from .base import VectorStoreBase
from .user_isolation import validate_user_id


class StoreProvider(Enum):
    """存储提供者"""
    MILVUS = "milvus"
    CHROMA = "chroma"
    AUTO = "auto"  # 自动选择（优先 Milvus）


def create_vector_store(
    provider: Union[StoreProvider, str] = StoreProvider.AUTO,
    user_id: Optional[str] = None,
    collection_prefix: str = "legal_kb",
    enable_isolation: bool = None,
    **kwargs
) -> VectorStoreBase:
    """
    创建向量存储实例
    
    Args:
        provider: 存储提供者
        user_id: 用户ID（用于隔离）
        collection_prefix: collection 名称前缀
        enable_isolation: 是否启用用户隔离
        **kwargs: 其他参数（传递给具体存储）
            - Milvus: host, port, dimension, index_params, search_params
            - Chroma: persist_directory, dimension
            
    Returns:
        向量存储实例
    """
    if isinstance(provider, str):
        provider = StoreProvider(provider.lower())
    
    # 自动选择
    if provider == StoreProvider.AUTO:
        provider = _detect_available_provider()
    
    if provider == StoreProvider.MILVUS:
        from .milvus_store import MilvusStore, MILVUS_AVAILABLE
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus 不可用，请安装 pymilvus")
        
        return MilvusStore(
            user_id=user_id,
            collection_prefix=collection_prefix,
            enable_isolation=enable_isolation,
            **kwargs
        )
    
    elif provider == StoreProvider.CHROMA:
        from .chroma_store import ChromaStore, CHROMA_AVAILABLE
        if not CHROMA_AVAILABLE:
            raise ImportError("ChromaDB 不可用，请安装 langchain-chroma")
        
        return ChromaStore(
            user_id=user_id,
            collection_prefix=collection_prefix,
            enable_isolation=enable_isolation,
            **kwargs
        )
    
    else:
        raise ValueError(f"不支持的存储提供者: {provider}")


def _detect_available_provider() -> StoreProvider:
    """检测可用的存储提供者"""
    # 优先尝试 Milvus
    try:
        from .milvus_store import MILVUS_AVAILABLE
        if MILVUS_AVAILABLE:
            # 尝试连接
            from pymilvus import connections
            connections.connect(alias="test_connection", host="localhost", port=19530)
            connections.disconnect("test_connection")
            return StoreProvider.MILVUS
    except:
        pass
    
    # 回退到 ChromaDB
    try:
        from .chroma_store import CHROMA_AVAILABLE
        if CHROMA_AVAILABLE:
            return StoreProvider.CHROMA
    except:
        pass
    
    raise RuntimeError("没有可用的向量存储后端")


def create_bm25_index(
    user_id: Optional[str] = None,
    collection_prefix: str = "legal_bm25",
    enable_isolation: bool = None,
    **kwargs
):
    """
    创建 BM25 索引实例
    
    Args:
        user_id: 用户ID（用于隔离）
        collection_prefix: 索引名称前缀
        enable_isolation: 是否启用用户隔离
        **kwargs: 其他参数
            - persist_directory: 持久化目录
            - use_jieba: 是否使用 jieba 分词
            
    Returns:
        BM25Index 实例
    """
    from .bm25_index import BM25Index
    
    return BM25Index(
        user_id=user_id,
        collection_prefix=collection_prefix,
        enable_isolation=enable_isolation,
        **kwargs
    )


class HybridStore:
    """
    混合存储管理器
    
    整合向量存储和 BM25 索引
    """
    
    def __init__(
        self,
        vector_store: VectorStoreBase,
        bm25_index = None,
        use_bm25: bool = True
    ):
        """
        初始化混合存储
        
        Args:
            vector_store: 向量存储实例
            bm25_index: BM25 索引实例
            use_bm25: 是否启用 BM25
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.use_bm25 = use_bm25 and bm25_index is not None
    
    @classmethod
    def create(
        cls,
        provider: Union[StoreProvider, str] = StoreProvider.AUTO,
        user_id: Optional[str] = None,
        collection_prefix: str = "legal_kb",
        enable_isolation: bool = None,
        use_bm25: bool = True,
        **kwargs
    ) -> "HybridStore":
        """
        创建混合存储
        
        Args:
            provider: 向量存储提供者
            user_id: 用户ID
            collection_prefix: 名称前缀
            enable_isolation: 是否启用用户隔离
            use_bm25: 是否启用 BM25
            **kwargs: 其他参数
            
        Returns:
            HybridStore 实例
        """
        # 创建向量存储
        vector_store = create_vector_store(
            provider=provider,
            user_id=user_id,
            collection_prefix=collection_prefix,
            enable_isolation=enable_isolation,
            **kwargs
        )
        
        # 创建 BM25 索引
        bm25_index = None
        if use_bm25:
            try:
                bm25_index = create_bm25_index(
                    user_id=user_id,
                    collection_prefix=f"{collection_prefix}_bm25",
                    enable_isolation=enable_isolation,
                    persist_directory=kwargs.get("bm25_persist_directory", "./data/bm25_index")
                )
            except ImportError as e:
                print(f"BM25 索引创建失败: {e}")
        
        return cls(
            vector_store=vector_store,
            bm25_index=bm25_index,
            use_bm25=use_bm25
        )
    
    def add_documents(
        self,
        documents,
        embeddings,
        ids = None,
        batch_size: int = 100
    ):
        """
        添加文档到向量存储和 BM25 索引
        
        Args:
            documents: 文档列表
            embeddings: 向量列表
            ids: 文档ID列表
            batch_size: 批处理大小
            
        Returns:
            文档ID列表
        """
        # 添加到向量存储
        doc_ids = self.vector_store.add_documents(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            batch_size=batch_size
        )
        
        # 添加到 BM25 索引
        if self.use_bm25 and self.bm25_index:
            self.bm25_index.add_documents(documents, ids=doc_ids)
        
        return doc_ids
    
    def delete_documents(self, ids):
        """删除文档"""
        count = self.vector_store.delete_documents(ids)
        
        if self.use_bm25 and self.bm25_index:
            self.bm25_index.delete_documents(ids)
        
        return count
    
    def clear(self):
        """清空存储"""
        self.vector_store.clear_collection()
        
        if self.use_bm25 and self.bm25_index:
            self.bm25_index.clear()
    
    def get_count(self) -> int:
        """获取文档数量"""
        return self.vector_store.get_collection_count()
