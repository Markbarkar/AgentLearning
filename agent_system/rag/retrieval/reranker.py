"""
重排序模块

对检索结果进行重排序，提高相关性
"""

from typing import List, Dict, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass

from .hybrid_retriever import RetrievalResult


class RerankerType(Enum):
    """重排序器类型"""
    NONE = "none"                    # 不重排序
    CROSS_ENCODER = "cross_encoder"  # 交叉编码器（如 BGE-Reranker）
    LLM = "llm"                      # LLM 重排序
    CUSTOM = "custom"                # 自定义重排序


class Reranker:
    """
    结果重排序器
    
    支持多种重排序策略：
    - 交叉编码器（基于模型）
    - LLM 重排序
    - 自定义评分函数
    """
    
    def __init__(
        self,
        reranker_type: RerankerType = RerankerType.NONE,
        model_name: Optional[str] = None,
        llm = None,
        custom_scorer: Optional[Callable] = None,
        top_k: int = 10,
    ):
        """
        初始化重排序器
        
        Args:
            reranker_type: 重排序器类型
            model_name: 模型名称（用于交叉编码器）
            llm: LLM 实例（用于 LLM 重排序）
            custom_scorer: 自定义评分函数
            top_k: 重排序后保留的结果数量
        """
        self.reranker_type = reranker_type
        self.model_name = model_name
        self.llm = llm
        self.custom_scorer = custom_scorer
        self.top_k = top_k
        
        self._model = None
        
        if reranker_type == RerankerType.CROSS_ENCODER:
            self._init_cross_encoder()
    
    def _init_cross_encoder(self):
        """初始化交叉编码器模型"""
        try:
            from sentence_transformers import CrossEncoder
            
            model_name = self.model_name or "BAAI/bge-reranker-base"
            self._model = CrossEncoder(model_name)
            print(f"✓ 已加载重排序模型: {model_name}")
        except ImportError:
            print("警告: sentence_transformers 未安装，交叉编码器不可用")
            self.reranker_type = RerankerType.NONE
        except Exception as e:
            print(f"警告: 加载重排序模型失败: {e}")
            self.reranker_type = RerankerType.NONE
    
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None
    ) -> List[RetrievalResult]:
        """
        重排序检索结果
        
        Args:
            query: 查询文本
            results: 检索结果列表
            top_k: 返回数量（默认使用初始化时的 top_k）
            
        Returns:
            重排序后的结果列表
        """
        if not results:
            return []
        
        top_k = top_k or self.top_k
        
        if self.reranker_type == RerankerType.NONE:
            return results[:top_k]
        
        elif self.reranker_type == RerankerType.CROSS_ENCODER:
            return self._rerank_cross_encoder(query, results, top_k)
        
        elif self.reranker_type == RerankerType.LLM:
            return self._rerank_llm(query, results, top_k)
        
        elif self.reranker_type == RerankerType.CUSTOM:
            return self._rerank_custom(query, results, top_k)
        
        return results[:top_k]
    
    def _rerank_cross_encoder(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """使用交叉编码器重排序"""
        if not self._model:
            return results[:top_k]
        
        # 构建 query-document 对
        pairs = [(query, r.content) for r in results]
        
        # 计算相关性分数
        scores = self._model.predict(pairs)
        
        # 更新结果分数并排序
        for i, result in enumerate(results):
            result.rerank_score = float(scores[i])
            result.score = result.rerank_score  # 用重排序分数替换原分数
        
        # 按新分数排序
        sorted_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        
        return sorted_results[:top_k]
    
    def _rerank_llm(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """使用 LLM 重排序"""
        if not self.llm:
            return results[:top_k]
        
        # 构建评分提示
        prompt = self._build_llm_rerank_prompt(query, results)
        
        try:
            response = self.llm.invoke(prompt)
            scores = self._parse_llm_scores(response.content, len(results))
            
            for i, result in enumerate(results):
                result.rerank_score = scores.get(i, 0.0)
                result.score = result.rerank_score
            
            sorted_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
            return sorted_results[:top_k]
        except Exception as e:
            print(f"LLM 重排序失败: {e}")
            return results[:top_k]
    
    def _build_llm_rerank_prompt(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> str:
        """构建 LLM 重排序提示"""
        docs_text = "\n\n".join([
            f"[文档 {i}]\n{r.content[:500]}..."
            for i, r in enumerate(results)
        ])
        
        prompt = f"""请对以下文档与查询的相关性进行评分（0-10分）。

查询: {query}

文档列表:
{docs_text}

请以 JSON 格式返回评分，例如: {{"0": 8.5, "1": 3.2, "2": 9.0}}
只返回 JSON，不要其他文字。"""
        
        return prompt
    
    def _parse_llm_scores(
        self,
        response: str,
        num_docs: int
    ) -> Dict[int, float]:
        """解析 LLM 返回的分数"""
        import json
        import re
        
        # 尝试提取 JSON
        json_match = re.search(r'\{[^}]+\}', response)
        if json_match:
            try:
                scores_dict = json.loads(json_match.group())
                return {int(k): float(v) for k, v in scores_dict.items()}
            except:
                pass
        
        # 返回默认分数
        return {i: 0.0 for i in range(num_docs)}
    
    def _rerank_custom(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int
    ) -> List[RetrievalResult]:
        """使用自定义评分函数重排序"""
        if not self.custom_scorer:
            return results[:top_k]
        
        for result in results:
            try:
                result.rerank_score = self.custom_scorer(query, result)
                result.score = result.rerank_score
            except Exception as e:
                print(f"自定义评分失败: {e}")
                result.rerank_score = result.score
        
        sorted_results = sorted(results, key=lambda x: x.rerank_score, reverse=True)
        return sorted_results[:top_k]


def create_legal_reranker(
    boost_recent: bool = True,
    boost_article_match: bool = True
) -> Reranker:
    """
    创建法律文档专用重排序器
    
    Args:
        boost_recent: 是否提升较新文档
        boost_article_match: 是否提升条款匹配
        
    Returns:
        配置好的重排序器
    """
    def legal_scorer(query: str, result: RetrievalResult) -> float:
        score = result.score
        metadata = result.metadata
        
        # 条款号匹配加分
        if boost_article_match:
            import re
            query_articles = re.findall(r'第([一二三四五六七八九十百千\d]+)条', query)
            doc_article = metadata.get("article_num", "")
            if query_articles and doc_article:
                for article in query_articles:
                    if article in doc_article:
                        score += 0.2
        
        # 较新文档加分
        if boost_recent:
            effective_date = metadata.get("effective_date", "")
            if effective_date:
                try:
                    year = int(effective_date[:4])
                    if year >= 2020:
                        score += 0.1
                    if year >= 2023:
                        score += 0.1
                except:
                    pass
        
        # 地区匹配加分
        query_lower = query.lower()
        region = metadata.get("region", "")
        if region and region in query_lower:
            score += 0.15
        
        return score
    
    return Reranker(
        reranker_type=RerankerType.CUSTOM,
        custom_scorer=legal_scorer
    )
