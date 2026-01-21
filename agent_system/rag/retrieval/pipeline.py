"""
检索管道

整合查询解析、混合检索、重排序
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

from .query_parser import QueryParser, ParsedQuery
from .hybrid_retriever import HybridRetriever, RetrievalResult, FusionMethod
from .reranker import Reranker, RerankerType, create_legal_reranker


@dataclass
class RetrievalConfig:
    """检索配置"""
    # 检索参数
    top_k: int = 10                              # 返回结果数量
    fetch_k: int = 50                            # 初始检索数量（重排序前）
    
    # 检索方式
    use_vector: bool = True                      # 是否使用向量检索
    use_bm25: bool = True                        # 是否使用 BM25 检索
    
    # 融合参数
    fusion_method: str = "rrf"                   # 融合方法: rrf/linear/max
    vector_weight: float = 0.6                   # 向量检索权重
    bm25_weight: float = 0.4                     # BM25 检索权重
    
    # 重排序
    enable_rerank: bool = True                   # 是否启用重排序
    reranker_type: str = "custom"                # 重排序类型
    
    # 查询解析
    enable_query_parsing: bool = True            # 是否启用查询解析
    enable_query_expansion: bool = True          # 是否启用查询扩展
    
    # 过滤
    similarity_threshold: float = 0.3            # 相似度阈值


class RetrievalPipeline:
    """
    检索管道
    
    整合完整的检索流程：
    1. 查询解析 - 提取关键词、实体、过滤条件
    2. 混合检索 - 向量 + BM25
    3. 结果融合 - RRF/线性加权
    4. 重排序 - 提升相关性
    5. 结果过滤 - 阈值过滤
    """
    
    def __init__(
        self,
        vector_store = None,
        bm25_index = None,
        embedding_service = None,
        config: Optional[RetrievalConfig] = None,
        reranker: Optional[Reranker] = None,
        query_parser: Optional[QueryParser] = None,
    ):
        """
        初始化检索管道
        
        Args:
            vector_store: 向量存储实例
            bm25_index: BM25 索引实例
            embedding_service: 嵌入服务
            config: 检索配置
            reranker: 重排序器（可选，默认使用法律专用重排序器）
            query_parser: 查询解析器（可选）
        """
        self.config = config or RetrievalConfig()
        
        # 初始化查询解析器
        self.query_parser = query_parser or QueryParser(
            enable_expansion=self.config.enable_query_expansion
        )
        
        # 初始化混合检索器
        fusion_method = FusionMethod(self.config.fusion_method)
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25_index,
            embedding_service=embedding_service,
            fusion_method=fusion_method,
            vector_weight=self.config.vector_weight,
            bm25_weight=self.config.bm25_weight,
        )
        
        # 初始化重排序器
        if reranker:
            self.reranker = reranker
        elif self.config.enable_rerank:
            self.reranker = create_legal_reranker()
        else:
            self.reranker = Reranker(reranker_type=RerankerType.NONE)
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[RetrievalResult]:
        """
        执行完整检索流程
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            filter: 额外的过滤条件
            **kwargs: 其他参数
            
        Returns:
            检索结果列表
        """
        top_k = top_k or self.config.top_k
        
        # 1. 查询解析
        if self.config.enable_query_parsing:
            parsed_query = self.query_parser.parse(query)
            # 合并过滤条件
            if filter:
                parsed_query.filters.update(filter)
        else:
            parsed_query = ParsedQuery(
                original_query=query,
                normalized_query=query,
                filters=filter or {}
            )
        
        # 2. 混合检索
        results = self.retriever.retrieve(
            query=parsed_query,
            k=self.config.fetch_k,
            use_vector=self.config.use_vector,
            use_bm25=self.config.use_bm25,
            **kwargs
        )
        
        # 3. 重排序
        if self.config.enable_rerank and results:
            results = self.reranker.rerank(
                query=query,
                results=results,
                top_k=top_k
            )
        else:
            results = results[:top_k]
        
        # 4. 阈值过滤
        if self.config.similarity_threshold > 0:
            results = [
                r for r in results
                if r.score >= self.config.similarity_threshold
            ]
        
        return results
    
    def search_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
        context_window: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        带上下文的检索
        
        返回检索结果及相关上下文信息
        
        Args:
            query: 用户查询
            top_k: 返回结果数量
            filter: 过滤条件
            context_window: 上下文窗口大小
            
        Returns:
            包含结果和上下文的字典
        """
        results = self.search(query, top_k, filter, **kwargs)
        
        # 解析查询以获取更多信息
        parsed_query = self.query_parser.parse(query)
        
        return {
            "query": query,
            "parsed_query": {
                "normalized": parsed_query.normalized_query,
                "query_type": parsed_query.query_type.value,
                "keywords": parsed_query.keywords,
                "entities": parsed_query.entities,
                "filters": parsed_query.filters,
                "expansion_terms": parsed_query.expansion_terms,
            },
            "results": results,
            "total_found": len(results),
            "metadata": {
                "use_vector": self.config.use_vector,
                "use_bm25": self.config.use_bm25,
                "fusion_method": self.config.fusion_method,
                "rerank_enabled": self.config.enable_rerank,
            }
        }
    
    def format_results_for_llm(
        self,
        results: List[RetrievalResult],
        max_tokens: int = 4000,
        include_metadata: bool = True
    ) -> str:
        """
        将检索结果格式化为 LLM 上下文
        
        Args:
            results: 检索结果
            max_tokens: 最大 token 数（近似按字符计算）
            include_metadata: 是否包含元数据
            
        Returns:
            格式化后的文本
        """
        if not results:
            return "未找到相关文档。"
        
        formatted_parts = []
        current_length = 0
        
        for i, result in enumerate(results):
            # 构建单个结果文本
            parts = [f"[文档 {i + 1}]"]
            
            if include_metadata:
                metadata = result.metadata
                if metadata.get("law_name"):
                    parts.append(f"来源: {metadata['law_name']}")
                if metadata.get("chapter"):
                    parts.append(f"章节: {metadata['chapter']}")
                if metadata.get("article_num"):
                    parts.append(f"条款: {metadata['article_num']}")
            
            parts.append(f"内容: {result.content}")
            parts.append("")  # 空行分隔
            
            result_text = "\n".join(parts)
            
            # 检查长度限制
            if current_length + len(result_text) > max_tokens:
                # 尝试截断当前结果
                remaining = max_tokens - current_length - 50  # 留一些余量
                if remaining > 200:
                    truncated = result_text[:remaining] + "..."
                    formatted_parts.append(truncated)
                break
            
            formatted_parts.append(result_text)
            current_length += len(result_text)
        
        return "\n".join(formatted_parts)
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # 如果融合方法改变，更新检索器
        if "fusion_method" in kwargs:
            self.retriever.fusion_method = FusionMethod(kwargs["fusion_method"])
        if "vector_weight" in kwargs:
            self.retriever.vector_weight = kwargs["vector_weight"]
        if "bm25_weight" in kwargs:
            self.retriever.bm25_weight = kwargs["bm25_weight"]
