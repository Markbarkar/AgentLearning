"""
通用递归分块器

包装 LangChain 的 RecursiveCharacterTextSplitter
保持与现有代码的兼容性
"""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import ChunkerBase


class RecursiveChunker(ChunkerBase):
    """
    通用递归分块器
    
    使用 RecursiveCharacterTextSplitter 进行分块
    适用于通用文档，不考虑特定结构
    """
    
    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        """
        初始化递归分块器
        
        Args:
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
            separators: 分隔符列表，按优先级排序
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            # 默认分隔符，优先按段落、句子分割
            separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
    
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        将文本递归分块
        
        Args:
            text: 原始文本
            metadata: 文档元数据
            
        Returns:
            分块后的 Document 列表
        """
        if metadata is None:
            metadata = {}
        
        # 使用 splitter 分割文本
        chunks = self.splitter.split_text(text)
        
        # 创建 Document 对象
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            chunk_metadata["total_chunks"] = len(chunks)
            chunk_metadata["chunker"] = "recursive"
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents
