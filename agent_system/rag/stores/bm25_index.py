"""
BM25 全文索引

提供关键词检索能力，与向量检索互补
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field

from langchain_core.documents import Document

from .user_isolation import get_collection_name, validate_user_id
from ...config.settings import ENABLE_USER_ISOLATION

# 尝试导入 jieba 和 rank_bm25
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("警告: jieba 未安装，中文分词功能将受限。请运行 'pip install jieba' 安装")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("警告: rank_bm25 未安装，请运行 'pip install rank_bm25' 安装")


@dataclass
class BM25Result:
    """BM25 检索结果"""
    document: Document
    score: float
    doc_id: str
    
    @property
    def content(self) -> str:
        return self.document.page_content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.document.metadata


class BM25Index:
    """
    BM25 全文索引
    
    特性：
    - 中文分词支持 (jieba)
    - 高效关键词检索
    - 持久化存储
    - 用户隔离
    """
    
    # 中文停用词（可扩展）
    STOP_WORDS = {
        "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
        "或", "一个", "没有", "我们", "你们", "他们", "它们", "这个",
        "那个", "之", "以", "为", "在", "于", "等", "但", "到", "被",
        "也", "有", "不", "人", "这", "那", "上", "下", "中", "对",
        "其", "可以", "可", "会", "能", "要", "把", "让", "使", "从",
    }
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        collection_prefix: str = "legal_bm25",
        enable_isolation: bool = None,
        persist_directory: str = "./data/bm25_index",
        use_jieba: bool = True,
        **kwargs
    ):
        """
        初始化 BM25 索引
        
        Args:
            user_id: 用户ID（用于隔离）
            collection_prefix: 索引名称前缀
            enable_isolation: 是否启用用户隔离
            persist_directory: 持久化目录
            use_jieba: 是否使用 jieba 分词
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank_bm25 未安装，请运行 'pip install rank_bm25' 安装")
        
        self.user_id = validate_user_id(user_id)
        self.collection_prefix = collection_prefix
        self.enable_isolation = enable_isolation if enable_isolation is not None else ENABLE_USER_ISOLATION
        self.index_name = get_collection_name(
            self.user_id,
            self.collection_prefix,
            self.enable_isolation
        )
        
        self.persist_directory = persist_directory
        self.use_jieba = use_jieba and JIEBA_AVAILABLE
        
        # 确保目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 索引数据
        self._documents: List[Document] = []
        self._doc_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        
        # 加载已有索引
        self._load_index()
    
    def _get_index_path(self) -> Path:
        """获取索引文件路径"""
        return Path(self.persist_directory) / f"{self.index_name}.pkl"
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词
        
        Args:
            text: 待分词文本
            
        Returns:
            词元列表
        """
        if self.use_jieba:
            # 使用 jieba 分词
            tokens = list(jieba.cut(text))
        else:
            # 简单字符分割
            tokens = list(text)
        
        # 过滤停用词和空白
        tokens = [
            t.strip().lower() 
            for t in tokens 
            if t.strip() and t.strip() not in self.STOP_WORDS and len(t.strip()) > 0
        ]
        
        return tokens
    
    def _rebuild_bm25(self):
        """重建 BM25 索引"""
        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        添加文档到索引
        
        Args:
            documents: 文档列表
            ids: 文档ID列表（可选）
            
        Returns:
            文档ID列表
        """
        if not documents:
            return []
        
        import uuid
        
        # 生成 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        # 添加文档
        for doc_id, doc in zip(ids, documents):
            self._documents.append(doc)
            self._doc_ids.append(doc_id)
            
            # 分词
            tokens = self._tokenize(doc.page_content)
            self._tokenized_corpus.append(tokens)
        
        # 重建 BM25
        self._rebuild_bm25()
        
        # 持久化
        self._save_index()
        
        print(f"✓ 成功添加 {len(documents)} 个文档到 BM25 索引")
        return ids
    
    def search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[BM25Result]:
        """
        关键词检索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件
            
        Returns:
            检索结果列表
        """
        if not self._bm25 or not self._documents:
            return []
        
        # 分词查询
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []
        
        # BM25 打分
        scores = self._bm25.get_scores(query_tokens)
        
        # 结合过滤条件
        candidates = []
        for idx, score in enumerate(scores):
            if score <= 0:
                continue
            
            doc = self._documents[idx]
            doc_id = self._doc_ids[idx]
            
            # 应用过滤条件
            if filter:
                match = True
                for key, value in filter.items():
                    doc_value = doc.metadata.get(key)
                    if isinstance(value, (list, tuple)):
                        if doc_value not in value:
                            match = False
                            break
                    elif doc_value != value:
                        match = False
                        break
                if not match:
                    continue
            
            candidates.append((idx, score, doc, doc_id))
        
        # 按分数排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top-k
        results = []
        for idx, score, doc, doc_id in candidates[:k]:
            results.append(BM25Result(
                document=doc,
                score=score,
                doc_id=doc_id
            ))
        
        return results
    
    def delete_documents(self, ids: List[str]) -> int:
        """
        删除文档
        
        Args:
            ids: 文档ID列表
            
        Returns:
            删除的文档数量
        """
        if not ids:
            return 0
        
        ids_set = set(ids)
        
        # 找出要保留的文档
        new_documents = []
        new_doc_ids = []
        new_tokenized = []
        deleted_count = 0
        
        for i, doc_id in enumerate(self._doc_ids):
            if doc_id in ids_set:
                deleted_count += 1
            else:
                new_documents.append(self._documents[i])
                new_doc_ids.append(doc_id)
                new_tokenized.append(self._tokenized_corpus[i])
        
        self._documents = new_documents
        self._doc_ids = new_doc_ids
        self._tokenized_corpus = new_tokenized
        
        # 重建索引
        self._rebuild_bm25()
        
        # 持久化
        self._save_index()
        
        print(f"✓ 已删除 {deleted_count} 个文档")
        return deleted_count
    
    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """
        根据过滤条件删除文档
        
        Args:
            filter: 元数据过滤条件
            
        Returns:
            删除的文档数量
        """
        if not filter:
            return 0
        
        # 找出匹配的文档ID
        ids_to_delete = []
        for i, doc in enumerate(self._documents):
            match = True
            for key, value in filter.items():
                doc_value = doc.metadata.get(key)
                if isinstance(value, (list, tuple)):
                    if doc_value not in value:
                        match = False
                        break
                elif doc_value != value:
                    match = False
                    break
            if match:
                ids_to_delete.append(self._doc_ids[i])
        
        return self.delete_documents(ids_to_delete)
    
    def clear(self) -> None:
        """清空索引"""
        self._documents = []
        self._doc_ids = []
        self._tokenized_corpus = []
        self._bm25 = None
        
        # 删除持久化文件
        index_path = self._get_index_path()
        if index_path.exists():
            index_path.unlink()
        
        print(f"✓ 已清空 BM25 索引: {self.index_name}")
    
    def get_count(self) -> int:
        """获取文档数量"""
        return len(self._documents)
    
    def _save_index(self):
        """保存索引到磁盘"""
        index_path = self._get_index_path()
        
        data = {
            "documents": [(doc.page_content, doc.metadata) for doc in self._documents],
            "doc_ids": self._doc_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        
        with open(index_path, "wb") as f:
            pickle.dump(data, f)
    
    def _load_index(self):
        """从磁盘加载索引"""
        index_path = self._get_index_path()
        
        if not index_path.exists():
            return
        
        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
            
            # 恢复文档
            self._documents = [
                Document(page_content=content, metadata=metadata)
                for content, metadata in data["documents"]
            ]
            self._doc_ids = data["doc_ids"]
            self._tokenized_corpus = data["tokenized_corpus"]
            
            # 重建 BM25
            self._rebuild_bm25()
            
            print(f"✓ 已加载 BM25 索引: {self.index_name} ({len(self._documents)} 文档)")
        except Exception as e:
            print(f"加载 BM25 索引失败: {str(e)}")
