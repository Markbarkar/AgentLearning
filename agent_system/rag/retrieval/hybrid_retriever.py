"""
混合检索器

结合向量检索和 BM25 关键词检索
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

from .query_parser import ParsedQuery


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float                                  # 综合得分
    vector_score: Optional[float] = None          # 向量相似度得分
    bm25_score: Optional[float] = None            # BM25 得分
    rerank_score: Optional[float] = None          # 重排序得分
    source: str = "hybrid"                        # 来源: vector/bm25/hybrid
    doc_id: str = ""
    
    @property
    def content(self) -> str:
        return self.document.page_content
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self.document.metadata


class FusionMethod(Enum):
    """结果融合方法"""
    RRF = "rrf"                  # Reciprocal Rank Fusion
    LINEAR = "linear"            # 线性加权
    MAX = "max"                  # 取最大分数


class HybridRetriever:
    """
    混合检索器
    
    特性：
    - 向量检索 + BM25 检索
    - 多种融合策略
    - 结果去重
    - 元数据过滤
    """
    
    def __init__(
        self,
        vector_store = None,
        bm25_index = None,
        embedding_service = None,
        fusion_method: FusionMethod = FusionMethod.RRF,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ):
        """
        初始化混合检索器
        
        Args:
            vector_store: 向量存储实例
            bm25_index: BM25 索引实例
            embedding_service: 嵌入服务（用于查询向量化）
            fusion_method: 结果融合方法
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
            rrf_k: RRF 参数 k
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedding_service = embedding_service
        self.fusion_method = fusion_method
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
    
    def retrieve(
        self,
        query: Union[str, ParsedQuery],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        use_vector: bool = True,
        use_bm25: bool = True,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行混合检索
        
        Args:
            query: 查询（字符串或解析后的查询对象）
            k: 返回结果数量
            filter: 元数据过滤条件
            use_vector: 是否使用向量检索
            use_bm25: 是否使用 BM25 检索
            
        Returns:
            检索结果列表
        """
        # 处理 ParsedQuery
        if isinstance(query, ParsedQuery):
            query_text = query.search_query
            # 合并过滤条件
            if query.filters:
                filter = {**(filter or {}), **query.filters}
        else:
            query_text = query
        
        vector_results = []
        bm25_results = []
        
        # 向量检索
        if use_vector and self.vector_store and self.embedding_service:
            vector_results = self._vector_search(query_text, k * 2, filter)
        
        # BM25 检索
        if use_bm25 and self.bm25_index:
            bm25_results = self._bm25_search(query_text, k * 2, filter)
        
        # 如果只有一种检索方式
        if not bm25_results:
            return vector_results[:k]
        if not vector_results:
            return bm25_results[:k]
        
        # 融合结果
        fused_results = self._fuse_results(vector_results, bm25_results, k)
        
        return fused_results
    
    def _vector_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """向量检索"""
        # 生成查询向量
        query_embedding = self.embedding_service.embed_query(query)
        
        # 执行检索
        results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            k=k,
            filter=filter
        )
        
        # 转换结果
        retrieval_results = []
        for r in results:
            retrieval_results.append(RetrievalResult(
                document=r.document,
                score=r.score,
                vector_score=r.score,
                source="vector",
                doc_id=r.doc_id
            ))
        
        return retrieval_results
    
    def _bm25_search(
        self,
        query: str,
        k: int,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[RetrievalResult]:
        """BM25 检索"""
        results = self.bm25_index.search(query, k=k, filter=filter)
        
        # 归一化 BM25 分数
        max_score = max([r.score for r in results]) if results else 1.0
        
        retrieval_results = []
        for r in results:
            normalized_score = r.score / max_score if max_score > 0 else 0
            retrieval_results.append(RetrievalResult(
                document=r.document,
                score=normalized_score,
                bm25_score=r.score,
                source="bm25",
                doc_id=r.doc_id
            ))
        
        return retrieval_results
    
    def _fuse_results(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """融合检索结果"""
        if self.fusion_method == FusionMethod.RRF:
            return self._rrf_fusion(vector_results, bm25_results, k)
        elif self.fusion_method == FusionMethod.LINEAR:
            return self._linear_fusion(vector_results, bm25_results, k)
        else:
            return self._max_fusion(vector_results, bm25_results, k)
    
    def _rrf_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """
        Reciprocal Rank Fusion
        
        RRF score = sum(1 / (k + rank))
        """
        # 构建文档ID到结果的映射
        doc_scores = {}  # doc_id -> (rrf_score, result)
        
        # 处理向量结果
        for rank, result in enumerate(vector_results):
            doc_key = self._get_doc_key(result)
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            
            if doc_key in doc_scores:
                old_score, old_result = doc_scores[doc_key]
                # 合并分数
                new_result = RetrievalResult(
                    document=old_result.document,
                    score=old_score + rrf_score,
                    vector_score=result.vector_score,
                    bm25_score=old_result.bm25_score,
                    source="hybrid",
                    doc_id=old_result.doc_id
                )
                doc_scores[doc_key] = (old_score + rrf_score, new_result)
            else:
                doc_scores[doc_key] = (rrf_score, result)
        
        # 处理 BM25 结果
        for rank, result in enumerate(bm25_results):
            doc_key = self._get_doc_key(result)
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            
            if doc_key in doc_scores:
                old_score, old_result = doc_scores[doc_key]
                # 合并分数
                new_result = RetrievalResult(
                    document=old_result.document,
                    score=old_score + rrf_score,
                    vector_score=old_result.vector_score,
                    bm25_score=result.bm25_score,
                    source="hybrid",
                    doc_id=old_result.doc_id
                )
                doc_scores[doc_key] = (old_score + rrf_score, new_result)
            else:
                doc_scores[doc_key] = (rrf_score, result)
        
        # 排序并返回
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )
        
        return [r for _, r in sorted_results[:k]]
    
    def _linear_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """线性加权融合"""
        doc_scores = {}
        
        # 处理向量结果
        for result in vector_results:
            doc_key = self._get_doc_key(result)
            weighted_score = result.score * self.vector_weight
            
            if doc_key in doc_scores:
                old_score, old_result = doc_scores[doc_key]
                new_result = RetrievalResult(
                    document=old_result.document,
                    score=old_score + weighted_score,
                    vector_score=result.vector_score,
                    bm25_score=old_result.bm25_score,
                    source="hybrid",
                    doc_id=old_result.doc_id
                )
                doc_scores[doc_key] = (old_score + weighted_score, new_result)
            else:
                doc_scores[doc_key] = (weighted_score, result)
        
        # 处理 BM25 结果
        for result in bm25_results:
            doc_key = self._get_doc_key(result)
            weighted_score = result.score * self.bm25_weight
            
            if doc_key in doc_scores:
                old_score, old_result = doc_scores[doc_key]
                new_result = RetrievalResult(
                    document=old_result.document,
                    score=old_score + weighted_score,
                    vector_score=old_result.vector_score,
                    bm25_score=result.bm25_score,
                    source="hybrid",
                    doc_id=old_result.doc_id
                )
                doc_scores[doc_key] = (old_score + weighted_score, new_result)
            else:
                doc_scores[doc_key] = (weighted_score, result)
        
        # 排序并返回
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )
        
        return [r for _, r in sorted_results[:k]]
    
    def _max_fusion(
        self,
        vector_results: List[RetrievalResult],
        bm25_results: List[RetrievalResult],
        k: int
    ) -> List[RetrievalResult]:
        """取最大分数融合"""
        doc_scores = {}
        
        for result in vector_results + bm25_results:
            doc_key = self._get_doc_key(result)
            
            if doc_key in doc_scores:
                old_score, old_result = doc_scores[doc_key]
                if result.score > old_score:
                    doc_scores[doc_key] = (result.score, result)
            else:
                doc_scores[doc_key] = (result.score, result)
        
        sorted_results = sorted(
            doc_scores.values(),
            key=lambda x: x[0],
            reverse=True
        )
        
        return [r for _, r in sorted_results[:k]]
    
    def _get_doc_key(self, result: RetrievalResult) -> str:
        """获取文档唯一标识"""
        if result.doc_id:
            return result.doc_id
        # 使用内容哈希作为 fallback
        return str(hash(result.content[:200]))
