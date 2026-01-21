# 检索层构建文档

## 概述

检索层负责根据用户查询检索相关文档，包括：
1. **QueryParser** - 查询解析器
2. **VectorRetriever** - 向量检索器
3. **HybridRetriever** - 混合检索器（向量 + BM25 + RRF 融合）
4. **BGEReranker** - 重排序器

## 目录结构

```
agent_system/rag/
├── retrievers/              # 新增
│   ├── __init__.py
│   ├── base.py              # RetrieverBase 抽象基类
│   ├── query_parser.py      # 查询解析器
│   ├── vector_retriever.py  # 向量检索器
│   └── hybrid_retriever.py  # 混合检索器
│
└── rerankers/               # 新增
    ├── __init__.py
    ├── base.py              # RerankerBase 抽象基类
    └── bge_reranker.py      # BGE 重排序器
```

## 组件详情

### 1. QueryParser (查询解析器)

**核心功能：**
- 提取查询意图
- 提取元数据过滤条件（地区、类型等）
- 关键词提取
- 查询清洗

**接口定义：**
```python
@dataclass
class ParsedQuery:
    original: str              # 原始查询
    cleaned: str               # 清洗后查询
    keywords: List[str]        # 关键词
    filters: Dict[str, Any]    # 元数据过滤条件
    intent: str                # 查询意图

class QueryParser:
    def parse(self, query: str) -> ParsedQuery:
        """解析查询"""
        pass
```

**过滤条件提取示例：**
```python
# 输入: "深圳的消防条例有哪些规定"
# 输出:
ParsedQuery(
    original="深圳的消防条例有哪些规定",
    cleaned="消防条例规定",
    keywords=["消防", "条例", "规定"],
    filters={"region": "深圳", "law_type": "条例"},
    intent="legal_query"
)
```

### 2. RetrieverBase (抽象基类)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document

class RetrieverBase(ABC):
    """检索器抽象基类"""
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        检索文档
        
        Returns:
            [{"document": Document, "score": float, "source": str}, ...]
        """
        pass
```

### 3. VectorRetriever (向量检索器)

简单包装向量存储的相似度搜索。

```python
class VectorRetriever(RetrieverBase):
    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_service: EmbeddingService
    ):
        pass
    
    def retrieve(self, query, top_k=10, filters=None):
        query_embedding = self.embedding_service.embed_query(query)
        results = self.vector_store.similarity_search_with_score(
            query_embedding, k=top_k, filter=filters
        )
        return [{"document": doc, "score": score, "source": "vector"} 
                for doc, score in results]
```

### 4. HybridRetriever (混合检索器)

**核心功能：**
- 查询解析 -> 提取过滤条件
- 向量检索 -> 召回 top_k * 2
- BM25 检索 -> 召回 top_k * 2
- RRF 融合 -> 合并两路结果
- 元数据过滤 -> 按条件筛选
- Rerank -> 精排

**RRF (Reciprocal Rank Fusion) 算法：**
```python
def reciprocal_rank_fusion(
    results_lists: List[List[Dict]],
    k: int = 60
) -> List[Dict]:
    """
    RRF 融合算法
    
    score = sum(1 / (k + rank_i))
    
    Args:
        results_lists: 多个检索结果列表
        k: RRF 参数，通常为 60
    """
    scores = {}
    for results in results_lists:
        for rank, item in enumerate(results):
            doc_id = item["document"].metadata.get("id")
            if doc_id not in scores:
                scores[doc_id] = {"item": item, "score": 0}
            scores[doc_id]["score"] += 1 / (k + rank + 1)
    
    sorted_items = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [item["item"] for item in sorted_items]
```

**完整流程：**
```python
class HybridRetriever(RetrieverBase):
    def __init__(
        self,
        vector_store: VectorStoreBase,
        bm25_index: BM25Index,
        embedding_service: EmbeddingService,
        reranker: Optional[RerankerBase] = None,
        query_parser: Optional[QueryParser] = None
    ):
        pass
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        use_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        # 1. 解析查询
        parsed = self.query_parser.parse(query)
        merged_filters = {**parsed.filters, **(filters or {})}
        
        # 2. 向量检索
        vector_results = self._vector_search(
            parsed.cleaned, top_k * 2, merged_filters
        )
        
        # 3. BM25 检索
        bm25_results = self._bm25_search(parsed.cleaned, top_k * 2)
        
        # 4. RRF 融合
        fused = self._reciprocal_rank_fusion(
            [vector_results, bm25_results]
        )
        
        # 5. Rerank
        if use_rerank and self.reranker:
            fused = self.reranker.rerank(query, fused, top_k)
        
        return fused[:top_k]
```

### 5. BGEReranker (重排序器)

**核心功能：**
- 使用 Cross-Encoder 模型精排
- 支持 GPU 加速
- 批量处理

```python
from sentence_transformers import CrossEncoder

class BGEReranker(RerankerBase):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        batch_size: int = 32
    ):
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not results:
            return []
        
        # 构建 (query, doc) 对
        pairs = [(query, r["document"].page_content) for r in results]
        
        # 计算相关性分数
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        
        # 更新分数并排序
        for r, score in zip(results, scores):
            r["rerank_score"] = float(score)
        
        sorted_results = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return sorted_results[:top_k]
```

## 配置项

```python
# settings.py 新增

# 检索配置
RETRIEVAL_TOP_K = 5
RETRIEVAL_RECALL_MULTIPLIER = 2  # 召回时 top_k 的倍数

# RRF 配置
RRF_K = 60

# Reranker 配置
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANKER_DEVICE = "cuda"  # 或 "cpu"
RERANKER_BATCH_SIZE = 32
RERANKER_ENABLED = True
```

## 依赖

```
# requirements.txt 新增
sentence-transformers>=2.2.0
torch>=2.0.0
```

## 使用示例

```python
from agent_system.rag.retrievers import HybridRetriever, QueryParser
from agent_system.rag.rerankers import BGEReranker
from agent_system.rag.stores import MilvusStore
from agent_system.rag.indexes import BM25Index

# 初始化组件
milvus = MilvusStore(...)
bm25 = BM25Index(...)
reranker = BGEReranker(device="cuda")
parser = QueryParser()

# 创建混合检索器
retriever = HybridRetriever(
    vector_store=milvus,
    bm25_index=bm25,
    embedding_service=embedding_service,
    reranker=reranker,
    query_parser=parser
)

# 检索
results = retriever.retrieve(
    query="深圳的消防安全规定有哪些",
    top_k=5,
    use_rerank=True
)

for r in results:
    print(f"[{r['score']:.3f}] {r['document'].metadata['law_name']}")
    print(f"  {r['document'].page_content[:100]}...")
```

## 验收标准

- [ ] QueryParser 正确提取地区、类型等过滤条件
- [ ] VectorRetriever 向量检索正常
- [ ] BM25 检索正常
- [ ] RRF 融合正确合并两路结果
- [ ] HybridRetriever 端到端检索正常
- [ ] BGEReranker 能正确加载模型
- [ ] Rerank 后结果质量提升
- [ ] GPU 加速正常（如有）

## 依赖

- 预处理层完成
- 向量化层完成
- 存储层完成（提供 MilvusStore 和 BM25Index）
