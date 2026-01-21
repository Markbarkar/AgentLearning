"""
向量存储抽象基类

定义统一的向量存储接口，方便切换不同的存储后端
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class SearchResult:
    """检索结果"""
    document: Document
    score: float
    doc_id: str
    
    @property
    def content(self) -> str:
        return self.document.page_content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.document.metadata


class VectorStoreBase(ABC):
    """
    向量存储抽象基类
    
    定义所有向量存储适配器必须实现的接口
    """
    
    def __init__(
        self,
        collection_name: str,
        dimension: int = 1024,
        **kwargs
    ):
        """
        初始化向量存储
        
        Args:
            collection_name: 集合名称
            dimension: 向量维度
        """
        self.collection_name = collection_name
        self.dimension = dimension
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """
        添加文档到向量存储
        
        Args:
            documents: 文档列表
            embeddings: 对应的向量列表
            ids: 文档ID列表（可选）
            batch_size: 批处理大小
            
        Returns:
            文档ID列表
        """
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        相似度检索
        
        Args:
            query_embedding: 查询向量
            k: 返回结果数量
            filter: 元数据过滤条件
            
        Returns:
            检索结果列表
        """
        pass
    
    @abstractmethod
    def delete_documents(self, ids: List[str]) -> int:
        """
        删除文档
        
        Args:
            ids: 文档ID列表
            
        Returns:
            删除的文档数量
        """
        pass
    
    @abstractmethod
    def get_collection_count(self) -> int:
        """
        获取集合中的文档数量
        
        Returns:
            文档数量
        """
        pass
    
    @abstractmethod
    def clear_collection(self) -> None:
        """清空集合"""
        pass
    
    @abstractmethod
    def collection_exists(self) -> bool:
        """
        检查集合是否存在
        
        Returns:
            是否存在
        """
        pass
    
    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """
        根据过滤条件删除文档
        
        Args:
            filter: 元数据过滤条件
            
        Returns:
            删除的文档数量
        """
        raise NotImplementedError("此存储后端不支持按条件删除")
    
    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据ID获取文档
        
        Args:
            ids: 文档ID列表
            
        Returns:
            文档列表
        """
        raise NotImplementedError("此存储后端不支持按ID获取文档")
    
    def update_document(self, doc_id: str, document: Document, embedding: List[float]) -> bool:
        """
        更新文档
        
        Args:
            doc_id: 文档ID
            document: 新文档
            embedding: 新向量
            
        Returns:
            是否更新成功
        """
        raise NotImplementedError("此存储后端不支持更新文档")
