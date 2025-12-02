"""
向量数据库管理模块

封装 Chroma 向量数据库操作
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class VectorStoreManager:
    """
    向量数据库管理类
    
    封装 Chroma 向量数据库的操作，包括：
    - 初始化和持久化
    - 文档添加、删除
    - 相似度检索
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        """
        初始化向量数据库管理器
        
        Args:
            embeddings: 嵌入模型实例
            persist_directory: 持久化存储目录
            collection_name: 集合名称
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 确保存储目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化或加载向量数据库
        self.vector_store = self._init_vector_store()
    
    def _init_vector_store(self) -> Chroma:
        """
        初始化或加载向量数据库
        
        Returns:
            Chroma 向量数据库实例
        """
        try:
            # 尝试加载已有的向量数据库
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"✓ 成功加载向量数据库: {self.collection_name}")
            return vector_store
        except Exception as e:
            print(f"初始化新的向量数据库: {self.collection_name}")
            # 创建新的向量数据库
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            return vector_store
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
            
        Returns:
            文档 ID 列表
        """
        if not documents:
            return []
        
        print(f"正在添加 {len(documents)} 个文档到向量数据库...")
        
        # 分批添加文档
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = self.vector_store.add_documents(batch)
            all_ids.extend(ids)
            print(f"  已添加 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
        
        print(f"✓ 成功添加 {len(documents)} 个文档")
        return all_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        相似度检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            filter: 元数据过滤条件
            
        Returns:
            相关文档列表
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        相似度检索（带分数）
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            filter: 元数据过滤条件
            
        Returns:
            (文档, 相似度分数) 元组列表
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        删除文档
        
        Args:
            ids: 文档 ID 列表
        """
        if ids:
            self.vector_store.delete(ids=ids)
            print(f"✓ 已删除 {len(ids)} 个文档")
    
    def get_collection_count(self) -> int:
        """
        获取集合中的文档数量
        
        Returns:
            文档数量
        """
        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception as e:
            print(f"获取文档数量失败: {str(e)}")
            return 0
    
    def clear_collection(self) -> None:
        """
        清空集合中的所有文档
        """
        try:
            # 获取所有文档 ID
            collection = self.vector_store._collection
            all_ids = collection.get()['ids']
            
            if all_ids:
                self.delete_documents(all_ids)
                print(f"✓ 已清空集合 {self.collection_name}")
            else:
                print(f"集合 {self.collection_name} 已经是空的")
        except Exception as e:
            print(f"清空集合失败: {str(e)}")



