"""
ChromaDB 向量存储适配器

兼容现有的 ChromaDB 实现，提供统一接口
"""

import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document

from .base import VectorStoreBase, SearchResult
from .user_isolation import UserIsolationMixin, get_collection_name
from ...config.settings import CHROMA_PERSIST_DIR, QWEN_EMBEDDING_DIMENSION

try:
    from langchain_chroma import Chroma
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("警告: langchain_chroma 未安装")


class ChromaStore(UserIsolationMixin, VectorStoreBase):
    """
    ChromaDB 向量存储适配器
    
    特性：
    - 轻量级本地存储
    - 简单易用
    - 支持持久化
    - 用户隔离（通过 collection 分离）
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        collection_prefix: str = "legal_kb",
        enable_isolation: bool = None,
        persist_directory: str = CHROMA_PERSIST_DIR,
        dimension: int = QWEN_EMBEDDING_DIMENSION,
        embeddings: Optional[Any] = None,  # 用于兼容旧接口
        **kwargs
    ):
        """
        初始化 ChromaDB 存储
        
        Args:
            user_id: 用户ID（用于隔离）
            collection_prefix: collection 名称前缀
            enable_isolation: 是否启用用户隔离
            persist_directory: 持久化目录
            dimension: 向量维度
            embeddings: 嵌入模型（用于兼容旧接口，新接口不需要）
        """
        if not CHROMA_AVAILABLE:
            raise ImportError("langchain_chroma 未安装")
        
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
        
        self.persist_directory = persist_directory
        self.embeddings = embeddings  # 可选，仅用于兼容
        
        # 确保目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._init_collection()
    
    def _init_collection(self):
        """初始化或加载 collection"""
        try:
            collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print(f"✓ 已加载 ChromaDB Collection: {self.collection_name}")
            return collection
        except Exception as e:
            print(f"初始化 Collection 失败: {str(e)}")
            raise
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> List[str]:
        """添加文档到 ChromaDB"""
        if not documents:
            return []
        
        if len(documents) != len(embeddings):
            raise ValueError("文档数量与向量数量不匹配")
        
        # 生成 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        all_ids = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            # 准备数据
            contents = [doc.page_content for doc in batch_docs]
            metadatas = []
            for doc in batch_docs:
                # ChromaDB 元数据需要是基本类型
                metadata = {}
                for k, v in (doc.metadata or {}).items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
                    elif isinstance(v, list):
                        metadata[k] = str(v)
                    elif v is None:
                        metadata[k] = ""
                    else:
                        metadata[k] = str(v)
                metadatas.append(metadata)
            
            # 添加到 collection
            self._collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=contents,
                metadatas=metadatas
            )
            
            all_ids.extend(batch_ids)
            print(f"  已添加 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
        
        print(f"✓ 成功添加 {len(documents)} 个文档到 ChromaDB")
        return all_ids
    
    def similarity_search(
        self,
        query_embedding: List[float],
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> List[SearchResult]:
        """相似度检索"""
        # 构建 where 条件
        where = None
        if filter:
            where_conditions = []
            for key, value in filter.items():
                if isinstance(value, (list, tuple)):
                    where_conditions.append({key: {"$in": value}})
                else:
                    where_conditions.append({key: {"$eq": value}})
            
            if len(where_conditions) == 1:
                where = where_conditions[0]
            elif len(where_conditions) > 1:
                where = {"$and": where_conditions}
        
        # 执行查询
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # 转换结果
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for idx, doc_id in enumerate(results["ids"][0]):
                content = results["documents"][0][idx] if results["documents"] else ""
                metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                distance = results["distances"][0][idx] if results["distances"] else 0.0
                
                # ChromaDB 返回距离，转换为相似度分数
                # 对于余弦距离，相似度 = 1 - 距离
                score = 1.0 - distance
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                
                search_results.append(SearchResult(
                    document=doc,
                    score=score,
                    doc_id=doc_id
                ))
        
        return search_results
    
    def delete_documents(self, ids: List[str]) -> int:
        """删除文档"""
        if not ids:
            return 0
        
        self._collection.delete(ids=ids)
        print(f"✓ 已删除 {len(ids)} 个文档")
        return len(ids)
    
    def delete_by_filter(self, filter: Dict[str, Any]) -> int:
        """根据过滤条件删除文档"""
        # 构建 where 条件
        where_conditions = []
        for key, value in filter.items():
            if isinstance(value, (list, tuple)):
                where_conditions.append({key: {"$in": value}})
            else:
                where_conditions.append({key: {"$eq": value}})
        
        if not where_conditions:
            return 0
        
        if len(where_conditions) == 1:
            where = where_conditions[0]
        else:
            where = {"$and": where_conditions}
        
        # 查询匹配的文档
        results = self._collection.get(where=where)
        ids = results.get("ids", [])
        
        if ids:
            self._collection.delete(ids=ids)
            print(f"✓ 已删除 {len(ids)} 个文档")
        
        return len(ids)
    
    def get_collection_count(self) -> int:
        """获取集合中的文档数量"""
        try:
            return self._collection.count()
        except Exception as e:
            print(f"获取文档数量失败: {str(e)}")
            return 0
    
    def clear_collection(self) -> None:
        """清空集合"""
        try:
            # 获取所有文档 ID
            all_data = self._collection.get()
            ids = all_data.get("ids", [])
            
            if ids:
                self._collection.delete(ids=ids)
                print(f"✓ 已清空 Collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} 已经是空的")
        except Exception as e:
            print(f"清空集合失败: {str(e)}")
            raise
    
    def collection_exists(self) -> bool:
        """检查集合是否存在"""
        try:
            collections = self._client.list_collections()
            return any(c.name == self.collection_name for c in collections)
        except:
            return False
    
    def get_documents_by_ids(self, ids: List[str]) -> List[Document]:
        """根据ID获取文档"""
        if not ids:
            return []
        
        results = self._collection.get(ids=ids, include=["documents", "metadatas"])
        
        documents = []
        for idx, doc_id in enumerate(results.get("ids", [])):
            content = results["documents"][idx] if results.get("documents") else ""
            metadata = results["metadatas"][idx] if results.get("metadatas") else {}
            
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents
    
    def list_unique_files(self) -> List[Dict[str, Any]]:
        """获取唯一文件列表"""
        try:
            all_data = self._collection.get(include=["metadatas"])
            metadatas = all_data.get("metadatas", [])
            
            # 聚合
            files_dict = {}
            for metadata in metadatas:
                if metadata:
                    file_name = metadata.get("file_name", "")
                    if file_name:
                        if file_name not in files_dict:
                            files_dict[file_name] = {
                                "file_name": file_name,
                                "source": metadata.get("source", ""),
                                "law_type": metadata.get("law_type", ""),
                                "chunk_count": 0
                            }
                        files_dict[file_name]["chunk_count"] += 1
            
            return list(files_dict.values())
        except Exception as e:
            print(f"获取文件列表失败: {str(e)}")
            return []
