"""
语义分块器

使用嵌入模型进行语义相似度分块
基于 LangChain SemanticChunker 实现
"""

from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from .base import ChunkerBase


class SemanticChunker(ChunkerBase):
    """
    语义分块器
    
    基于文本的语义相似度进行分块，而不是固定字符数
    使用嵌入模型计算句子之间的语义距离，在语义断点处分割
    
    优点：
    - 保持语义完整性
    - 自动识别主题边界
    - 适合内容结构不明显的文档
    
    缺点：
    - 需要调用嵌入 API，成本较高
    - 处理速度较慢
    """
    
    def __init__(
        self,
        embeddings=None,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95,
        buffer_size: int = 1,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000
    ):
        """
        初始化语义分块器
        
        Args:
            embeddings: 嵌入模型实例（如 OpenAIEmbeddings）
                       如果为 None，将使用默认的 QwenEmbeddings
            breakpoint_threshold_type: 断点阈值类型
                - "percentile": 百分位数（默认）
                - "standard_deviation": 标准差
                - "interquartile": 四分位距
                - "gradient": 梯度变化
            breakpoint_threshold_amount: 断点阈值
                - percentile: 0-100，推荐 90-95
                - standard_deviation: 推荐 1.0-3.0
                - interquartile: 推荐 1.0-2.0
            buffer_size: 计算相似度时的缓冲区大小
            min_chunk_size: 最小分块大小（字符）
            max_chunk_size: 最大分块大小（字符）
        """
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.buffer_size = buffer_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # 延迟初始化嵌入模型和分块器
        self._embeddings = embeddings
        self._splitter = None
    
    def _get_splitter(self):
        """延迟初始化语义分块器"""
        if self._splitter is None:
            from langchain_experimental.text_splitter import SemanticChunker as LangChainSemanticChunker
            
            # 如果没有提供嵌入模型，使用默认的
            if self._embeddings is None:
                self._embeddings = self._create_default_embeddings()
            
            self._splitter = LangChainSemanticChunker(
                embeddings=self._embeddings,
                breakpoint_threshold_type=self.breakpoint_threshold_type,
                breakpoint_threshold_amount=self.breakpoint_threshold_amount,
                buffer_size=self.buffer_size
            )
        
        return self._splitter
    
    def _create_default_embeddings(self):
        """创建默认嵌入模型"""
        import os
        
        # 优先使用 OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=openai_key
                )
            except ImportError:
                pass
        
        # 备选使用 Qwen
        dashscope_key = os.getenv("DASHSCOPE_API_KEY")
        if dashscope_key:
            try:
                from ..embeddings import QwenEmbeddings
                return QwenEmbeddings()
            except ImportError:
                pass
        
        raise ValueError(
            "无法创建嵌入模型。请设置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY 环境变量，"
            "或在初始化时提供 embeddings 参数。"
        )
    
    def chunk(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        对文本进行语义分块
        
        Args:
            text: 原始文本
            metadata: 文档元数据
            
        Returns:
            分块后的 Document 列表
        """
        if metadata is None:
            metadata = {}
        
        if not text or len(text.strip()) < self.min_chunk_size:
            # 文本太短，直接返回
            return [Document(
                page_content=text.strip(),
                metadata={
                    **metadata,
                    "chunk_id": 0,
                    "total_chunks": 1,
                    "chunker": "semantic"
                }
            )]
        
        try:
            splitter = self._get_splitter()
            chunks = splitter.split_text(text)
        except Exception as e:
            print(f"语义分块失败，回退到简单分块: {e}")
            # 回退到简单分块
            chunks = self._simple_split(text)
        
        # 后处理：合并过小的块，分割过大的块
        processed_chunks = self._post_process_chunks(chunks)
        
        # 创建 Document 对象
        documents = []
        for i, chunk in enumerate(processed_chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_id": i,
                "total_chunks": len(processed_chunks),
                "chunker": "semantic",
                "breakpoint_type": self.breakpoint_threshold_type,
                "breakpoint_threshold": self.breakpoint_threshold_amount
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
        
        return documents
    
    def _simple_split(self, text: str) -> List[str]:
        """简单分块（回退方案）"""
        # 按段落分割
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _post_process_chunks(self, chunks: List[str]) -> List[str]:
        """后处理分块：合并过小的块，分割过大的块"""
        if not chunks:
            return chunks
        
        processed = []
        buffer = ""
        
        for chunk in chunks:
            # 合并过小的块
            if len(chunk) < self.min_chunk_size:
                buffer += chunk + "\n\n"
                if len(buffer) >= self.min_chunk_size:
                    processed.append(buffer.strip())
                    buffer = ""
            # 分割过大的块
            elif len(chunk) > self.max_chunk_size:
                if buffer:
                    processed.append(buffer.strip())
                    buffer = ""
                
                # 按句子分割大块
                sub_chunks = self._split_large_chunk(chunk)
                processed.extend(sub_chunks)
            else:
                if buffer:
                    # 尝试与 buffer 合并
                    if len(buffer) + len(chunk) <= self.max_chunk_size:
                        buffer += chunk + "\n\n"
                        processed.append(buffer.strip())
                        buffer = ""
                    else:
                        processed.append(buffer.strip())
                        buffer = ""
                        processed.append(chunk)
                else:
                    processed.append(chunk)
        
        # 处理剩余的 buffer
        if buffer:
            if processed and len(processed[-1]) + len(buffer) <= self.max_chunk_size:
                processed[-1] = processed[-1] + "\n\n" + buffer.strip()
            else:
                processed.append(buffer.strip())
        
        return processed
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """分割过大的块"""
        import re
        
        # 按句子分割
        sentences = re.split(r'([。！？；\n])', chunk)
        
        sub_chunks = []
        current = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            if i + 1 < len(sentences):
                sentence += sentences[i + 1]
            
            if len(current) + len(sentence) <= self.max_chunk_size:
                current += sentence
            else:
                if current:
                    sub_chunks.append(current.strip())
                current = sentence
        
        if current:
            sub_chunks.append(current.strip())
        
        return sub_chunks
