# 存储层构建文档

## 概述

存储层负责持久化存储文档向量和元数据，支持高效检索，包括：
1. **VectorStoreBase** - 向量存储抽象基类
2. **MilvusStore** - Milvus 向量数据库适配器
3. **ChromaStore** - ChromaDB 适配器（兼容现有）
4. **BM25Index** - BM25 全文索引

## 目录结构

```
agent_system/rag/
├── vector_store.py          # 现有（保留兼容）
│
├── stores/                  # 新增
│   ├── __init__.py
│   ├── base.py              # VectorStoreBase 抽象基类
│   ├── milvus_store.py      # Milvus 适配器
│   └── chroma_store.py      # ChromaDB 适配器
│
└── indexes/                 # 新增
    ├── __init__.py
    └── bm25_index.py        # BM25 索引
```

## 组件详情

### 1. VectorStoreBase (抽象基类)

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document

class VectorStoreBase(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[List[float]]] = None
    ) -> List[str]:
        """添加文档，返回文档 ID 列表"""
        pass
    
    @abstractmethod
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """向量相似度搜索"""
        pass
    
    @abstractmethod
    def similarity_search_with_score(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """向量相似度搜索（带分数）"""
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """删除文档"""
        pass
    
    @abstractmethod
    def get_count(self) -> int:
        """获取文档数量"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空所有文档"""
        pass
```

### 2. MilvusStore (Milvus 适配器)

**核心功能：**
- 连接池管理
- Collection 自动创建（含索引）
- 元数据字段映射（支持过滤）
- 批量插入优化
- 多种索引类型支持（IVF_FLAT, HNSW）

**Schema 设计：**
```python
fields = [
    FieldSchema("id", DataType.VARCHAR, max_length=64, is_primary=True),
    FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema("content", DataType.VARCHAR, max_length=65535),
    # 法律元数据字段
    FieldSchema("law_name", DataType.VARCHAR, max_length=256),
    FieldSchema("law_type", DataType.VARCHAR, max_length=32),
    FieldSchema("region", DataType.VARCHAR, max_length=64),
    FieldSchema("region_level", DataType.VARCHAR, max_length=32),
    FieldSchema("topics", DataType.VARCHAR, max_length=256),  # JSON array
    FieldSchema("chapter", DataType.VARCHAR, max_length=128),
    FieldSchema("article_num", DataType.VARCHAR, max_length=32),
    FieldSchema("source", DataType.VARCHAR, max_length=512),
    FieldSchema("chunk_id", DataType.INT64),
]
```

**索引配置：**
```python
index_params = {
    "metric_type": "IP",  # 内积（余弦相似度需归一化）
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024}
}
```

**元数据过滤示例：**
```python
# 按地区过滤
filter = {"region": "深圳"}

# 按法规类型过滤
filter = {"law_type": "条例"}

# 组合过滤
filter = {
    "region": ["深圳", "海南"],  # IN 查询
    "law_type": "条例"
}
```

### 3. ChromaStore (ChromaDB 适配器)

包装现有 `VectorStoreManager`，实现 `VectorStoreBase` 接口。

```python
class ChromaStore(VectorStoreBase):
    def __init__(self, ...):
        self._manager = VectorStoreManager(...)
    
    def add_documents(self, documents, embeddings=None):
        return self._manager.add_documents(documents)
    
    # ... 其他方法包装
```

### 4. BM25Index (BM25 全文索引)

**核心功能：**
- 中文分词（jieba）
- BM25 索引构建
- 索引持久化（pickle）
- 增量更新支持

**接口定义：**
```python
class BM25Index:
    def __init__(
        self,
        index_path: str,
        tokenizer: str = "jieba"  # "jieba" | "simple"
    ):
        pass
    
    def build(self, documents: List[Document]) -> None:
        """构建索引"""
        pass
    
    def add(self, documents: List[Document]) -> None:
        """增量添加"""
        pass
    
    def search(
        self,
        query: str,
        k: int = 10
    ) -> List[Tuple[Document, float]]:
        """搜索，返回 (文档, BM25分数) 列表"""
        pass
    
    def save(self) -> None:
        """保存索引到磁盘"""
        pass
    
    def load(self) -> None:
        """从磁盘加载索引"""
        pass
```

## 配置项

```python
# settings.py 新增

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION_PREFIX = "legal_kb"
MILVUS_INDEX_TYPE = "IVF_FLAT"
MILVUS_METRIC_TYPE = "IP"
MILVUS_NLIST = 1024

# BM25 配置
BM25_INDEX_DIR = "./data/bm25_index"
BM25_TOKENIZER = "jieba"
```

## 依赖

```
# requirements.txt 新增
pymilvus>=2.3.0
rank-bm25>=0.2.2
jieba>=0.42.1
```

## Milvus 部署

### Docker 单机部署

```bash
# 下载 docker-compose 文件
wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml

# 启动 Milvus
docker-compose up -d

# 检查状态
docker-compose ps
```

### 验证连接

```python
from pymilvus import connections

connections.connect(host="localhost", port="19530")
print("Milvus 连接成功")
```

## 使用示例

```python
from agent_system.rag.stores import MilvusStore
from agent_system.rag.indexes import BM25Index

# Milvus 存储
milvus = MilvusStore(
    host="localhost",
    port=19530,
    collection_name="legal_kb_user_1"
)

# 添加文档（带预计算的嵌入）
documents = [...]
embeddings = [...]
ids = milvus.add_documents(documents, embeddings)

# 向量搜索（带过滤）
results = milvus.similarity_search(
    query_embedding=query_vec,
    k=10,
    filter={"region": "深圳", "law_type": "条例"}
)

# BM25 索引
bm25 = BM25Index(index_path="./data/bm25_index/user_1")
bm25.build(documents)
bm25.save()

# BM25 搜索
bm25_results = bm25.search("消防安全规定", k=10)
```

## 验收标准

- [ ] MilvusStore 能正确连接 Milvus
- [ ] Collection 自动创建和索引配置正确
- [ ] 批量插入 10000+ 文档正常
- [ ] 元数据过滤查询正常
- [ ] BM25Index 中文分词正确
- [ ] BM25 索引持久化和恢复正常
- [ ] ChromaStore 兼容现有代码

## 依赖

- 预处理层完成（提供 Document 对象）
- 向量化层完成（提供 embeddings）
