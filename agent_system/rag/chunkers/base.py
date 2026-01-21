"""
分块器抽象基类

定义分块器的标准接口
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

from langchain_core.documents import Document


class ChunkerBase(ABC):
    """
    分块器抽象基类
    
    所有分块器都应继承此类并实现 chunk 方法
    """
    
    @abstractmethod
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        将文本分块
        
        Args:
            text: 原始文本内容
            metadata: 文档元数据，将被复制到每个分块
            
        Returns:
            Document 列表，每个 Document 包含：
            - page_content: 分块内容
            - metadata: 合并后的元数据（原始元数据 + 分块元数据）
        """
        pass
    
    def chunk_documents(
        self,
        documents: List[Document]
    ) -> List[Document]:
        """
        对多个文档进行分块
        
        Args:
            documents: 原始文档列表
            
        Returns:
            分块后的 Document 列表
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)
        return all_chunks
