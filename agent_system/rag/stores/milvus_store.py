"""
Milvus 向量存储适配器

提供 Milvus 向量数据库的统一接口实现
"""

import uuid
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document

from .base import VectorStoreBase, SearchResult
from .user_isolation import UserIsolationMixin, get_collection_name, validate_user_id
from ...config.settings import QWEN_EMBEDDING_DIMENSION

try:
    from pymilvus import (
        connections,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        utility,
        MilvusException,
    )
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    print("警告: pymilvus 未安装，请运行 'pip install pymilvus' 安装")


class MilvusStore(UserIsolationMixin, VectorStoreBase):
    """
    Milvus 向量存储适配器
    
    特性：
    - 高性能向量检索
    - 支持标量过滤
    - 支持分布式部署
    - 用户隔离（通过 collection 分离）
    """
    
    # 默认索引参数
    DEFAULT_INDEX_PARAMS = {
        "metric_type": "COSINE",  # 使用余弦相似度
        "index_type": "IVF_FLAT",  # IVF 索引
        "params": {"nlist": 1024}
    }
    
    # 默认搜索参数
    DEFAULT_SEARCH_PARAMS = {
        "metric_type": "COSINE",
        "params": {"nprobe": 16}
    }
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        collection_prefix: str = "legal_kb",
        enable_isolation: bool = None,
        host: str = "localhost",
        port: int = 19530,
        dimension: int = QWEN_EMBEDDING_DIMENSION,
        index_params: Optional[Dict[str, Any]] = None,
        search_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        初始化 Milvus 存储
        
        Args:
            user_id: 用户ID（用于隔离）
            collection_prefix: collection 名称前缀
            enable_isolation: 是否启用用户隔离
            host: Milvus 服务地址
            port: Milvus 服务端口
            dimension: 向量维度
            index_params: 索引参数
            search_params: 搜索参数
        """
        if not MILVUS_AVAILABLE:
            raise ImportError("pymilvus 未安装，请运行 'pip install pymilvus' 安装")
        
        # 初始化用户隔离
        UserIsolationMixin.__init__(
            self,
            user_id=user_id,
            collection_prefix=collection_prefix,
            enable_isolation=enable_isolation
        )
        
        # 初始化基类
        VectorStoreBase.__init__(
            self,
            collection_name=self.collection_name,
            dimension=dimension
        )
        
        self.host = host
        self.port = port
        self.index_params = index_params or self.DEFAULT_INDEX_PARAMS
        self.search_params = search_params or self.DEFAULT_SEARCH_PARAMS
        
        # 连接 Milvus
        self._connect()
        
        # 初始化或加载 collection
        self._collection: Optional[Collection] = None
        self._init_collection()
    
    def _connect(self):
        """连接到 Milvus 服务"""
        try:
            # 使用别名连接，避免重复连接
            alias = f"milvus_{self.host}_{self.port}"
            if alias not in connections.list_connections():
                connections.connect(alias=alias, host=self.host, port=self.port)
            self._alias = alias
            print(f"✓ 已连接 Milvus: {self.host}:{self.port}")
        except Exception as e:
            raise ConnectionError(f"无法连接到 Milvus {self.host}:{self.port}: {str(e)}")
    
    def _init_collection(self):
        """初始化或加载 collection"""
        try:
            if utility.has_collection(self.collection_name, using=self._alias):
                # 加载已有 collection
                self._collection = Collection(self.collection_name, using=self._alias)
                self._collection.load()
                print(f"✓ 已加载 Milvus Collection: {self.collection_name}")
            else:
                # 创建新 collection
                self._create_collection()
                print(f"✓ 已创建 Milvus Collection: {self.collection_name}")
        except Exception as e:
            print(f"初始化 Collection 失败: {str(e)}")
            raise
    
    def _create_collection(self):
        """创建新的 collection"""
        # 定义字段
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            # 法律文档常用元数据字段
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="law_name", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="law_type", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="chapter", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="article_num", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="publish_date", dtype=DataType.VARCHAR, max_length=16),
            FieldSchema(name="effective_date", dtype=DataType.VARCHAR, max_length=16),
            # 通用元数据 JSON 字段
            FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=16384),
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description=f"Legal knowledge base for {self.get_user_display_name()}"
        )
        
        self._collection = Collection(
            name=self.collection_name,
            schema=schema,
            using=self._alias
        )
        
        # 创建索引
        self._collection.create_index(
            field_name="embedding",
            index_params=self.index_params
        )
        
        # 加载 collection
        self._collection.load()
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """添加文档到 Milvus"""
        if not documents:
            return []
        
        if len(documents) != len(embeddings):
            raise ValueError("文档数量与向量数量不匹配")
        
        # 生成 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        all_ids = []
        import json
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # 准备数据
            data = []
            for doc_id, doc, emb in zip(batch_ids, batch_docs, batch_embeddings):
                metadata = doc.metadata or {}
                
                row = {
                    "id": doc_id,
                    "embedding": emb,
                    "content": doc.page_content[:65535],  # 限制长度
                    "source": str(metadata.get("source", ""))[:512],
                    "file_name": str(metadata.get("file_name", ""))[:256],
                    "law_name": str(metadata.get("law_name", ""))[:256],
                    "law_type": str(metadata.get("law_type", ""))[:64],
                    "region": str(metadata.get("region", ""))[:64],
                    "chapter": str(metadata.get("chapter", ""))[:128],
                    "article_num": str(metadata.get("article_num", ""))[:32],
                    "publish_date": str(metadata.get("publish_date", ""))[:16],
                    "effective_date": str(metadata.get("effective_date", ""))[:16],
                    "metadata_json": json.dumps(metadata, ensure_ascii=False)[:16384],
                }
                data.append(row)
            
            # 插入数据
            self._collection.insert(data)
            all_ids.extend(batch_ids)
            
            print(f"  已添加 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
        
        # 刷新确保数据持久化
        self._collection.flush()
        
        print(f"✓ 成功添加 {len(documents)} 个文档到 Milvus")
        return all_ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """相似度检索"""
        import json
        
        # 构建过滤表达式
        expr = None
        if filter:
            conditions = []
            for key, value in filter.items():
                if isinstance(value, str):
                    conditions.append(f'{key} == "{value}"')
                elif isinstance(value, (list, tuple)):
                    values_str = ", ".join([f'"{v}"' for v in value])
                    conditions.append(f'{key} in [{values_str}]')
            if conditions:
                expr = " and ".join(conditions)
        
        # 执行搜索
        search_params = kwargs.get("search_params", self.search_params)
        
        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            expr=expr,
            output_fields=["content", "source", "file_name", "law_name", "law_type", 
                          "region", "chapter", "article_num", "metadata_json"]
        )
        
        # 转换结果
        search_results = []
        for hits in results:
            for hit in hits:
                # 解析元数据
                try:
                    metadata = json.loads(hit.entity.get("metadata_json", "{}"))
                except:
                    metadata = {}
                
                # 补充字段
                metadata.update({
                    "source": hit.entity.get("source", ""),
                    "file_name": hit.entity.get("file_name", ""),
                    "law_name": hit.entity.get("law_name", ""),
                    "law_type": hit.entity.get("law_type", ""),
                    "region": hit.entity.get("region", ""),
                    "chapter": hit.entity.get("chapter", ""),
                    "article_num": hit.entity.get("article_num", ""),
                })
                
                doc = Document(
                    page_content=hit.entity.get("content", ""),
                    metadata=metadata
                )
                
                search_results.append(SearchResult(
                    document=doc,
                    score=hit.score,
                    doc_id=hit.id
                ))
        
        return search_results
    
    def delete_documents(self, ids: List[str]) -> int:
        """删除文档"""
        if not ids:
            return 0
        
        ids_str = ", ".join([f'"{id}"' for id in ids])
        expr = f"id in [{ids_str}]"
        
        self._collection.delete(expr)
        self._collection.flush()
        
        print(f"✓ 已删除 {len(ids)} 个文档")
        return len(ids)
    
    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """根据过滤条件删除文档"""
        conditions = []
        for key, value in filter.items():
            if isinstance(value, str):
                conditions.append(f'{key} == "{value}"')
            elif isinstance(value, (list, tuple)):
                values_str = ", ".join([f'"{v}"' for v in value])
                conditions.append(f'{key} in [{values_str}]')
        
        if not conditions:
            return 0
        
        expr = " and ".join(conditions)
        
        # 先查询匹配的数量
        count_before = self.get_collection_count()
        
        self._collection.delete(expr)
        self._collection.flush()
        
        count_after = self.get_collection_count()
        deleted = count_before - count_after
        
        print(f"✓ 已删除 {deleted} 个文档（条件: {expr}）")
        return deleted
    
    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        try:
            self._collection.flush()
            return self._collection.num_entities
        except Exception as e:
            print(f"获取文档数量失败: {str(e)}")
            return 0
    
    def clear_collection(self) -> None:
        """清空集合"""
        try:
            # 删除并重建 collection
            utility.drop_collection(self.collection_name, using=self._alias)
            self._create_collection()
            print(f"✓ 已清空并重建 Collection: {self.collection_name}")
        except Exception as e:
            print(f"清空集合失败: {str(e)}")
            raise
    
    def collection_exists(self) -> bool:
        """检查集合是否存在"""
        return utility.has_collection(self.collection_name, using=self._alias)
    
    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """根据ID获取文档"""
        import json
        
        if not ids:
            return []
        
        ids_str = ", ".join([f'"{id}"' for id in ids])
        expr = f"id in [{ids_str}]"
        
        results = self._collection.query(
            expr=expr,
            output_fields=["content", "metadata_json"]
        )
        
        documents = []
        for item in results:
            try:
                metadata = json.loads(item.get("metadata_json", "{}"))
            except:
                metadata = {}
            
            doc = Document(
                page_content=item.get("content", ""),
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def list_unique_files(self) -> List[Dict[str, Any]]:
        """获取唯一文件列表"""
        # 查询所有文档的 file_name
        results = self._collection.query(
            expr="id != ''",  # 查询所有
            output_fields=["file_name", "source", "law_type"]
        )
        
        # 聚合
        files_dict = {}
        for item in results:
            file_name = item.get("file_name", "")
            if file_name:
                if file_name not in files_dict:
                    files_dict[file_name] = {
                        "file_name": file_name,
                        "source": item.get("source", ""),
                        "law_type": item.get("law_type", ""),
                        "chunk_count": 0
                    }
                files_dict[file_name]["chunk_count"] += 1
        
        return list(files_dict.values())
    
    def close(self):
        """关闭连接"""
        try:
            if self._collection:
                self._collection.release()
            # 不断开共享连接
            print(f"✓ 已释放 Collection: {self.collection_name}")
        except Exception as e:
            print(f"关闭连接失败: {str(e)}")
