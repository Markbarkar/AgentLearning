"""
批处理模块

提供同步和异步的批处理能力
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

from langchain_core.documents import Document


@dataclass
class BatchResult:
    """批处理结果"""
    documents: List[Document]
    embeddings: List[List[float]]
    success_count: int
    error_count: int
    duration: float
    errors: List[Dict[str, Any]]


class BatchProcessor:
    """
    同步批处理器
    
    使用线程池进行批量嵌入处理
    """
    
    def __init__(
        self,
        embedding_service,
        batch_size: int = 100,
        max_workers: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        初始化批处理器
        
        Args:
            embedding_service: 嵌入服务实例
            batch_size: 每批处理的文档数量
            max_workers: 最大工作线程数
            progress_callback: 进度回调函数 (已完成数, 总数)
        """
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.progress_callback = progress_callback
    
    def process(
        self,
        documents: List[Document],
        cache=None
    ) -> BatchResult:
        """
        批量处理文档
        
        Args:
            documents: 文档列表
            cache: 可选的嵌入缓存
            
        Returns:
            BatchResult 对象
        """
        start_time = time.time()
        
        if not documents:
            return BatchResult(
                documents=[],
                embeddings=[],
                success_count=0,
                error_count=0,
                duration=0,
                errors=[]
            )
        
        # 提取文本
        texts = [doc.page_content for doc in documents]
        
        # 检查缓存
        cached_embeddings = {}
        texts_to_embed = texts
        indices_to_embed = list(range(len(texts)))
        
        if cache is not None:
            cached, missing = cache.get_batch(texts)
            cached_embeddings = cached
            texts_to_embed = [texts[i] for i in missing]
            indices_to_embed = missing
        
        # 嵌入未缓存的文本
        new_embeddings = []
        errors = []
        
        if texts_to_embed:
            # 分批处理
            for i in range(0, len(texts_to_embed), self.batch_size):
                batch = texts_to_embed[i:i + self.batch_size]
                
                try:
                    batch_embeddings = self.embedding_service.embed_documents(batch)
                    new_embeddings.extend(batch_embeddings)
                    
                    # 更新缓存
                    if cache is not None:
                        cache.set_batch(batch, batch_embeddings)
                    
                except Exception as e:
                    errors.append({
                        "batch_start": i,
                        "batch_size": len(batch),
                        "error": str(e)
                    })
                    # 填充空向量
                    dim = self.embedding_service.dimension
                    new_embeddings.extend([[0.0] * dim] * len(batch))
                
                # 进度回调
                if self.progress_callback:
                    completed = len(cached_embeddings) + len(new_embeddings)
                    self.progress_callback(completed, len(texts))
        
        # 合并结果
        all_embeddings = [None] * len(texts)
        
        # 填充缓存命中的嵌入
        for idx, emb in cached_embeddings.items():
            all_embeddings[idx] = emb
        
        # 填充新计算的嵌入
        for i, idx in enumerate(indices_to_embed):
            if i < len(new_embeddings):
                all_embeddings[idx] = new_embeddings[i]
        
        duration = time.time() - start_time
        
        return BatchResult(
            documents=documents,
            embeddings=all_embeddings,
            success_count=len(texts) - len(errors),
            error_count=len(errors),
            duration=duration,
            errors=errors
        )


class AsyncBatcher:
    """
    异步批处理器
    
    使用异步 IO 进行批量嵌入处理
    支持并发控制和进度跟踪
    """
    
    def __init__(
        self,
        embedding_service,
        batch_size: int = 100,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """
        初始化异步批处理器
        
        Args:
            embedding_service: 嵌入服务实例
            batch_size: 每批处理的文档数量
            max_concurrent: 最大并发数
            progress_callback: 进度回调函数
        """
        self.embedding_service = embedding_service
        self.batch_size = batch_size
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback
        self._semaphore = None
    
    async def process(
        self,
        documents: List[Document],
        cache=None
    ) -> BatchResult:
        """
        异步批量处理文档
        
        Args:
            documents: 文档列表
            cache: 可选的嵌入缓存
            
        Returns:
            BatchResult 对象
        """
        start_time = time.time()
        
        if not documents:
            return BatchResult(
                documents=[],
                embeddings=[],
                success_count=0,
                error_count=0,
                duration=0,
                errors=[]
            )
        
        # 初始化信号量
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # 提取文本
        texts = [doc.page_content for doc in documents]
        
        # 检查缓存
        cached_embeddings = {}
        texts_to_embed = texts
        indices_to_embed = list(range(len(texts)))
        
        if cache is not None:
            cached, missing = cache.get_batch(texts)
            cached_embeddings = cached
            texts_to_embed = [texts[i] for i in missing]
            indices_to_embed = missing
        
        # 准备批次
        batches = []
        for i in range(0, len(texts_to_embed), self.batch_size):
            batch = texts_to_embed[i:i + self.batch_size]
            batch_indices = indices_to_embed[i:i + self.batch_size]
            batches.append((batch, batch_indices))
        
        # 并发处理
        all_embeddings = [None] * len(texts)
        errors = []
        completed = [len(cached_embeddings)]  # 使用列表以便在闭包中修改
        
        # 填充缓存命中的嵌入
        for idx, emb in cached_embeddings.items():
            all_embeddings[idx] = emb
        
        async def process_batch(batch: List[str], indices: List[int]):
            async with self._semaphore:
                try:
                    # 在线程池中运行同步嵌入
                    loop = asyncio.get_event_loop()
                    embeddings = await loop.run_in_executor(
                        None,
                        self.embedding_service.embed_documents,
                        batch
                    )
                    
                    # 更新缓存
                    if cache is not None:
                        cache.set_batch(batch, embeddings)
                    
                    # 更新结果
                    for i, idx in enumerate(indices):
                        all_embeddings[idx] = embeddings[i]
                    
                    completed[0] += len(batch)
                    
                except Exception as e:
                    errors.append({
                        "batch_indices": indices,
                        "error": str(e)
                    })
                    # 填充空向量
                    dim = self.embedding_service.dimension
                    for idx in indices:
                        all_embeddings[idx] = [0.0] * dim
                    completed[0] += len(batch)
                
                # 进度回调
                if self.progress_callback:
                    self.progress_callback(completed[0], len(texts))
        
        # 创建任务
        tasks = [process_batch(batch, indices) for batch, indices in batches]
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        duration = time.time() - start_time
        
        return BatchResult(
            documents=documents,
            embeddings=all_embeddings,
            success_count=len(texts) - len(errors),
            error_count=len(errors),
            duration=duration,
            errors=errors
        )
    
    def process_sync(
        self,
        documents: List[Document],
        cache=None
    ) -> BatchResult:
        """
        同步接口（内部使用 asyncio.run）
        
        Args:
            documents: 文档列表
            cache: 可选的嵌入缓存
            
        Returns:
            BatchResult 对象
        """
        return asyncio.run(self.process(documents, cache))
