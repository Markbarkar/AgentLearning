"""
Qwen Embedding 模型封装

使用 DashScope SDK 调用 Qwen 的 text-embedding-v3 模型
实现 LangChain 的 Embeddings 接口
"""

import time
from typing import List
from langchain_core.embeddings import Embeddings
import dashscope
from dashscope import TextEmbedding

from ..config.settings import QWEN_EMBEDDING_MODEL, QWEN_EMBEDDING_API_KEY


class QwenEmbeddings(Embeddings):
    """
    Qwen Embedding 模型封装类
    
    使用阿里云 DashScope API 调用 Qwen 的 text-embedding-v3 模型
    支持批量文本嵌入和错误重试
    """
    
    def __init__(
        self,
        model: str = QWEN_EMBEDDING_MODEL,
        api_key: str = QWEN_EMBEDDING_API_KEY,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化 Qwen Embeddings
        
        Args:
            model: 模型名称，默认为 text-embedding-v3
            api_key: DashScope API Key
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 设置 API Key
        if self.api_key:
            dashscope.api_key = self.api_key
        else:
            raise ValueError("DASHSCOPE_API_KEY 未设置，请在环境变量中配置")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        embeddings = []
        
        # 分批处理，每批最多 25 个文本（DashScope 限制）
        batch_size = 25
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_with_retry(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入单个查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        embeddings = self._embed_with_retry([text])
        return embeddings[0]
    
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        带重试机制的嵌入方法
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        for attempt in range(self.max_retries):
            try:
                response = TextEmbedding.call(
                    model=self.model,
                    input=texts
                )
                
                if response.status_code == 200:
                    # 提取嵌入向量
                    embeddings = [item['embedding'] for item in response.output['embeddings']]
                    return embeddings
                else:
                    error_msg = f"DashScope API 错误: {response.code} - {response.message}"
                    if attempt < self.max_retries - 1:
                        print(f"{error_msg}，{self.retry_delay} 秒后重试...")
                        time.sleep(self.retry_delay)
                    else:
                        raise Exception(error_msg)
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"嵌入失败: {str(e)}，{self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(f"嵌入失败（已重试 {self.max_retries} 次）: {str(e)}")
        
        return []



