"""
嵌入服务模块

提供统一的嵌入接口，支持多种嵌入模型
"""

import os
from abc import ABC, abstractmethod
from typing import List, Optional, Union
from enum import Enum


class EmbeddingProvider(Enum):
    """嵌入模型提供商"""
    QWEN = "qwen"
    OPENAI = "openai"
    LOCAL = "local"  # 本地模型，如 BGE


class EmbeddingServiceBase(ABC):
    """嵌入服务抽象基类"""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass


class EmbeddingService(EmbeddingServiceBase):
    """
    统一嵌入服务
    
    封装多种嵌入模型，提供统一接口
    支持：
    - Qwen text-embedding-v3
    - OpenAI text-embedding-3-small/large
    - 本地模型（如 BGE）
    
    使用示例:
        # 使用 Qwen
        service = EmbeddingService(provider="qwen")
        
        # 使用 OpenAI
        service = EmbeddingService(provider="openai", model="text-embedding-3-small")
        
        # 嵌入文档
        embeddings = service.embed_documents(["文本1", "文本2"])
    """
    
    # 模型维度映射
    DIMENSIONS = {
        # Qwen
        "text-embedding-v3": 1024,
        "text-embedding-v2": 1536,
        # OpenAI
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        # BGE
        "bge-large-zh-v1.5": 1024,
        "bge-m3": 1024,
    }
    
    def __init__(
        self,
        provider: Union[str, EmbeddingProvider] = "qwen",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 25,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        """
        初始化嵌入服务
        
        Args:
            provider: 嵌入模型提供商 ("qwen", "openai", "local")
            model: 模型名称，如果为 None 则使用默认模型
            api_key: API 密钥，如果为 None 则从环境变量读取
            batch_size: 批处理大小
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
            **kwargs: 其他模型特定参数
        """
        if isinstance(provider, str):
            provider = EmbeddingProvider(provider.lower())
        
        self.provider = provider
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 根据提供商初始化具体的嵌入模型
        if provider == EmbeddingProvider.QWEN:
            self._init_qwen(model, api_key)
        elif provider == EmbeddingProvider.OPENAI:
            self._init_openai(model, api_key)
        elif provider == EmbeddingProvider.LOCAL:
            self._init_local(model, **kwargs)
        else:
            raise ValueError(f"不支持的嵌入提供商: {provider}")
    
    def _init_qwen(self, model: Optional[str], api_key: Optional[str]):
        """初始化 Qwen 嵌入模型"""
        self.model = model or "text-embedding-v3"
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        
        if not self._api_key:
            raise ValueError("Qwen 嵌入需要 DASHSCOPE_API_KEY")
        
        # 延迟导入
        import dashscope
        dashscope.api_key = self._api_key
        self._dashscope = dashscope
        
        self._dimension = self.DIMENSIONS.get(self.model, 1024)
    
    def _init_openai(self, model: Optional[str], api_key: Optional[str]):
        """初始化 OpenAI 嵌入模型"""
        self.model = model or "text-embedding-3-small"
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self._api_key:
            raise ValueError("OpenAI 嵌入需要 OPENAI_API_KEY")
        
        from langchain_openai import OpenAIEmbeddings
        self._embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=self._api_key
        )
        
        self._dimension = self.DIMENSIONS.get(self.model, 1536)
    
    def _init_local(self, model: Optional[str], **kwargs):
        """初始化本地嵌入模型"""
        self.model = model or "bge-large-zh-v1.5"
        
        try:
            from sentence_transformers import SentenceTransformer
            model_path = kwargs.get("model_path", f"BAAI/{self.model}")
            device = kwargs.get("device", "cuda")
            self._local_model = SentenceTransformer(model_path, device=device)
            self._dimension = self._local_model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("本地嵌入需要安装 sentence-transformers: pip install sentence-transformers")
    
    @property
    def dimension(self) -> int:
        """返回向量维度"""
        return self._dimension
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        if self.provider == EmbeddingProvider.QWEN:
            return self._embed_qwen(texts)
        elif self.provider == EmbeddingProvider.OPENAI:
            return self._embed_openai(texts)
        elif self.provider == EmbeddingProvider.LOCAL:
            return self._embed_local(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        embeddings = self.embed_documents([text])
        return embeddings[0] if embeddings else []
    
    def _embed_qwen(self, texts: List[str]) -> List[List[float]]:
        """使用 Qwen 嵌入"""
        import time
        from dashscope import TextEmbedding
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            for attempt in range(self.max_retries):
                try:
                    response = TextEmbedding.call(
                        model=self.model,
                        input=batch
                    )
                    
                    if response.status_code == 200:
                        embeddings = [item['embedding'] for item in response.output['embeddings']]
                        all_embeddings.extend(embeddings)
                        break
                    else:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_delay)
                        else:
                            raise Exception(f"Qwen API 错误: {response.code} - {response.message}")
                            
                except Exception as e:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        raise e
        
        return all_embeddings
    
    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """使用 OpenAI 嵌入"""
        return self._embeddings.embed_documents(texts)
    
    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        """使用本地模型嵌入"""
        embeddings = self._local_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embeddings.tolist()


def create_embedding_service(
    provider: str = "auto",
    **kwargs
) -> EmbeddingService:
    """
    工厂函数：创建嵌入服务
    
    Args:
        provider: 提供商名称，"auto" 表示自动选择
        **kwargs: 传递给 EmbeddingService 的参数
        
    Returns:
        EmbeddingService 实例
    """
    if provider == "auto":
        # 自动选择：优先 Qwen，其次 OpenAI
        if os.getenv("DASHSCOPE_API_KEY"):
            provider = "qwen"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise ValueError("未找到可用的嵌入服务 API Key")
    
    return EmbeddingService(provider=provider, **kwargs)
