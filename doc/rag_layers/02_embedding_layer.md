# 向量化层构建文档

## 状态：已完成 ✅

## 概述

向量化层负责将文本转换为向量表示，包括：
1. **EmbeddingService** - 统一的嵌入服务接口（支持 Qwen/OpenAI/本地模型）
2. **BatchProcessor / AsyncBatcher** - 同步/异步批处理器
3. **EmbeddingCache** - 嵌入缓存（内存 + 磁盘）
4. **EmbeddingPipeline** - 完整流水线（支持切换分块器对比）

## 目录结构

```
agent_system/rag/
├── embeddings.py            # 现有 QwenEmbeddings（保留兼容）
│
└── embedding/               # 新增 ✅
    ├── __init__.py          # 模块导出
    ├── service.py           # EmbeddingService 统一服务
    ├── batcher.py           # BatchProcessor / AsyncBatcher
    ├── cache.py             # EmbeddingCache 缓存
    └── pipeline.py          # EmbeddingPipeline 流水线
```

## 组件详情

### 1. EmbeddingService (统一嵌入服务)

**核心功能：**
- 封装 QwenEmbeddings，提供统一接口
- 支持批量处理和错误重试
- 可选的本地模型备选（如 BGE-M3）

**接口定义：**
```python
class EmbeddingService:
    def __init__(
        self,
        model: str = "text-embedding-v3",
        api_key: Optional[str] = None,
        batch_size: int = 25,
        max_retries: int = 3
    ):
        pass
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        pass
    
    async def embed_documents_async(self, texts: List[str]) -> List[List[float]]:
        """异步批量嵌入"""
        pass
```

### 2. AsyncBatcher (异步批处理器)

**核心功能：**
- 将大量文档分批异步处理
- 并发控制（避免 API 限流）
- 进度回调

**接口定义：**
```python
class AsyncBatcher:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        batch_size: int = 100,
        max_concurrent: int = 3,
        progress_callback: Optional[Callable] = None
    ):
        pass
    
    async def process(
        self,
        documents: List[Document]
    ) -> List[Tuple[Document, List[float]]]:
        """
        异步批量处理文档，返回 (文档, 向量) 元组列表
        """
        pass
```

### 3. EmbeddingCache (嵌入缓存)

**核心功能：**
- 缓存已计算的嵌入向量
- 支持内存缓存和磁盘持久化
- 基于内容哈希的去重

**接口定义：**
```python
class EmbeddingCache:
    def __init__(
        self,
        cache_dir: str = "./data/embedding_cache",
        max_memory_items: int = 10000
    ):
        pass
    
    def get(self, text: str) -> Optional[List[float]]:
        """获取缓存的嵌入"""
        pass
    
    def set(self, text: str, embedding: List[float]) -> None:
        """缓存嵌入"""
        pass
    
    def get_batch(self, texts: List[str]) -> Tuple[List[int], List[List[float]]]:
        """
        批量获取缓存
        返回: (未命中的索引列表, 命中的嵌入列表)
        """
        pass
```

## 配置项

```python
# settings.py 新增

# Embedding 服务配置
EMBEDDING_BATCH_SIZE = 25           # API 批处理大小
EMBEDDING_MAX_CONCURRENT = 3        # 最大并发数
EMBEDDING_MAX_RETRIES = 3           # 最大重试次数
EMBEDDING_RETRY_DELAY = 1.0         # 重试延迟（秒）

# 缓存配置
EMBEDDING_CACHE_ENABLED = True
EMBEDDING_CACHE_DIR = "./data/embedding_cache"
EMBEDDING_CACHE_MAX_MEMORY = 10000  # 内存缓存最大条目
```

## 使用示例

```python
from agent_system.rag.embedding import EmbeddingService, AsyncBatcher

# 初始化服务
service = EmbeddingService()

# 同步批量嵌入
texts = ["文本1", "文本2", "文本3"]
embeddings = service.embed_documents(texts)

# 异步批量处理大量文档
batcher = AsyncBatcher(
    embedding_service=service,
    batch_size=100,
    max_concurrent=3,
    progress_callback=lambda p: print(f"进度: {p}%")
)

documents = [...]  # 大量 Document 对象
results = await batcher.process(documents)
# results: [(doc1, embedding1), (doc2, embedding2), ...]
```

## 与现有代码的兼容性

现有的 `QwenEmbeddings` 类将被保留，`EmbeddingService` 内部包装它：

```python
class EmbeddingService:
    def __init__(self, ...):
        self._embeddings = QwenEmbeddings(...)
```

## 验收标准

- [ ] EmbeddingService 兼容现有 QwenEmbeddings 接口
- [ ] AsyncBatcher 能正确处理 1000+ 文档
- [ ] 并发控制有效，不触发 API 限流
- [ ] 进度回调正常工作
- [ ] EmbeddingCache 命中率正常
- [ ] 缓存持久化和恢复正常

## 依赖

- 预处理层完成（提供 Document 对象）
