"""
嵌入缓存模块

缓存已计算的嵌入向量，减少重复计算和 API 调用
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from collections import OrderedDict
import threading


class EmbeddingCache:
    """
    嵌入向量缓存
    
    支持：
    - 内存缓存（LRU）
    - 磁盘持久化
    - 基于内容哈希的去重
    - 线程安全
    
    使用示例:
        cache = EmbeddingCache(cache_dir="./data/embedding_cache")
        
        # 检查缓存
        embedding = cache.get("文本内容")
        if embedding is None:
            embedding = embedding_service.embed_query("文本内容")
            cache.set("文本内容", embedding)
        
        # 批量操作
        hits, misses = cache.get_batch(["文本1", "文本2", "文本3"])
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        max_memory_items: int = 10000,
        enable_disk_cache: bool = True,
        hash_algorithm: str = "md5"
    ):
        """
        初始化缓存
        
        Args:
            cache_dir: 磁盘缓存目录，None 表示仅使用内存缓存
            max_memory_items: 内存缓存最大条目数
            enable_disk_cache: 是否启用磁盘缓存
            hash_algorithm: 哈希算法 ("md5", "sha256")
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_memory_items = max_memory_items
        self.enable_disk_cache = enable_disk_cache and cache_dir is not None
        self.hash_algorithm = hash_algorithm
        
        # 内存缓存（LRU）
        self._memory_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = threading.RLock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        
        # 创建缓存目录
        if self.enable_disk_cache and self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, text: str) -> str:
        """计算文本哈希"""
        if self.hash_algorithm == "md5":
            return hashlib.md5(text.encode('utf-8')).hexdigest()
        elif self.hash_algorithm == "sha256":
            return hashlib.sha256(text.encode('utf-8')).hexdigest()
        else:
            raise ValueError(f"不支持的哈希算法: {self.hash_algorithm}")
    
    def _get_disk_path(self, text_hash: str) -> Path:
        """获取磁盘缓存路径"""
        # 使用哈希前两位作为子目录，避免单目录文件过多
        subdir = text_hash[:2]
        return self.cache_dir / subdir / f"{text_hash}.pkl"
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        获取缓存的嵌入向量
        
        Args:
            text: 文本内容
            
        Returns:
            嵌入向量，未命中返回 None
        """
        text_hash = self._compute_hash(text)
        
        with self._lock:
            # 1. 检查内存缓存
            if text_hash in self._memory_cache:
                # 移动到末尾（LRU）
                self._memory_cache.move_to_end(text_hash)
                self._hits += 1
                return self._memory_cache[text_hash]
            
            # 2. 检查磁盘缓存
            if self.enable_disk_cache:
                disk_path = self._get_disk_path(text_hash)
                if disk_path.exists():
                    try:
                        with open(disk_path, 'rb') as f:
                            embedding = pickle.load(f)
                        
                        # 加载到内存缓存
                        self._add_to_memory(text_hash, embedding)
                        self._hits += 1
                        return embedding
                    except Exception:
                        pass
            
            self._misses += 1
            return None
    
    def set(self, text: str, embedding: List[float]) -> None:
        """
        缓存嵌入向量
        
        Args:
            text: 文本内容
            embedding: 嵌入向量
        """
        text_hash = self._compute_hash(text)
        
        with self._lock:
            # 添加到内存缓存
            self._add_to_memory(text_hash, embedding)
            
            # 写入磁盘缓存
            if self.enable_disk_cache:
                self._write_to_disk(text_hash, embedding)
    
    def _add_to_memory(self, text_hash: str, embedding: List[float]) -> None:
        """添加到内存缓存"""
        # 如果已存在，移动到末尾
        if text_hash in self._memory_cache:
            self._memory_cache.move_to_end(text_hash)
            return
        
        # 检查容量
        while len(self._memory_cache) >= self.max_memory_items:
            # 移除最旧的项
            self._memory_cache.popitem(last=False)
        
        self._memory_cache[text_hash] = embedding
    
    def _write_to_disk(self, text_hash: str, embedding: List[float]) -> None:
        """写入磁盘缓存"""
        try:
            disk_path = self._get_disk_path(text_hash)
            disk_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(disk_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            print(f"写入磁盘缓存失败: {e}")
    
    def get_batch(
        self,
        texts: List[str]
    ) -> Tuple[Dict[int, List[float]], List[int]]:
        """
        批量获取缓存
        
        Args:
            texts: 文本列表
            
        Returns:
            (命中的 {索引: 嵌入}, 未命中的索引列表)
        """
        hits = {}
        misses = []
        
        for i, text in enumerate(texts):
            embedding = self.get(text)
            if embedding is not None:
                hits[i] = embedding
            else:
                misses.append(i)
        
        return hits, misses
    
    def set_batch(
        self,
        texts: List[str],
        embeddings: List[List[float]]
    ) -> None:
        """
        批量缓存嵌入
        
        Args:
            texts: 文本列表
            embeddings: 对应的嵌入向量列表
        """
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0
            
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "memory_size": len(self._memory_cache),
                "max_memory_items": self.max_memory_items,
            }
    
    def clear_memory(self) -> None:
        """清空内存缓存"""
        with self._lock:
            self._memory_cache.clear()
    
    def clear_disk(self) -> None:
        """清空磁盘缓存"""
        if self.enable_disk_cache and self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def clear_all(self) -> None:
        """清空所有缓存"""
        self.clear_memory()
        self.clear_disk()
        
        with self._lock:
            self._hits = 0
            self._misses = 0
    
    def clear(self) -> None:
        """清空所有缓存（clear_all 的别名）"""
        self.clear_all()
