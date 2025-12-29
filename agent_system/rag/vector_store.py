"""
向量数据库管理模块

封装 Chroma 向量数据库操作
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ..config.settings import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class VectorStoreManager:
    """
    向量数据库管理类
    
    封装 Chroma 向量数据库的操作，包括：
    - 初始化和持久化
    - 文档添加、删除
    - 相似度检索
    """
    
    def __init__(
        self,
        embeddings: Embeddings,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME
    ):
        """
        初始化向量数据库管理器
        
        Args:
            embeddings: 嵌入模型实例
            persist_directory: 持久化存储目录
            collection_name: 集合名称
        """
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # 确保存储目录存在
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # 初始化或加载向量数据库
        self.vector_store = self._init_vector_store()
    
    def _init_vector_store(self) -> Chroma:
        """
        初始化或加载向量数据库
        
        Returns:
            Chroma 向量数据库实例
        """
        try:
            # 尝试加载已有的向量数据库
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            print(f"✓ 成功加载向量数据库: {self.collection_name}")
            return vector_store
        except Exception as e:
            print(f"初始化新的向量数据库: {self.collection_name}")
            # 创建新的向量数据库
            vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            return vector_store
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> List[str]:
        """
        添加文档到向量数据库
        
        Args:
            documents: 文档列表
            batch_size: 批处理大小
            
        Returns:
            文档 ID 列表
        """
        if not documents:
            return []
        
        print(f"正在添加 {len(documents)} 个文档到向量数据库...")
        
        # 分批添加文档
        all_ids = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            ids = self.vector_store.add_documents(batch)
            all_ids.extend(ids)
            print(f"  已添加 {min(i + batch_size, len(documents))}/{len(documents)} 个文档")
        
        print(f"✓ 成功添加 {len(documents)} 个文档")
        return all_ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        相似度检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            filter: 元数据过滤条件
            
        Returns:
            相关文档列表
        """
        results = self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[tuple[Document, float]]:
        """
        相似度检索（带分数）
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            filter: 元数据过滤条件
            
        Returns:
            (文档, 相似度分数) 元组列表
        """
        results = self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        return results
    
    def delete_documents(self, ids: List[str]) -> None:
        """
        删除文档
        
        Args:
            ids: 文档 ID 列表
        """
        if ids:
            self.vector_store.delete(ids=ids)
            print(f"✓ 已删除 {len(ids)} 个文档")
    
    def get_collection_count(self) -> int:
        """
        获取集合中的文档数量
        
        Returns:
            文档数量
        """
        try:
            collection = self.vector_store._collection
            return collection.count()
        except Exception as e:
            print(f"获取文档数量失败: {str(e)}")
            return 0
    
    def clear_collection(self) -> None:
        """
        清空集合中的所有文档
        """
        try:
            # 获取所有文档 ID
            collection = self.vector_store._collection
            all_ids = collection.get()['ids']
            
            if all_ids:
                self.delete_documents(all_ids)
                print(f"✓ 已清空集合 {self.collection_name}")
            else:
                print(f"集合 {self.collection_name} 已经是空的")
        except Exception as e:
            print(f"清空集合失败: {str(e)}")

    def get_all_documents(self) -> Dict[str, Any]:
        """
        获取collection中的所有文档及其metadata
        
        Returns:
            包含所有文档ID、metadata和内容的字典
        """
        try:
            collection = self.vector_store._collection
            result = collection.get(include=['metadatas', 'documents'])
            return result
        except Exception as e:
            print(f"获取文档失败: {str(e)}")
            return {'ids': [], 'metadatas': [], 'documents': []}
    
    def get_documents_by_source(self, source_path: str) -> Dict[str, Any]:
        """
        根据文件路径获取所有相关文档chunks
        
        Args:
            source_path: 文件路径（完整路径或文件名）
            
        Returns:
            包含该文件所有chunks的字典
        """
        try:
            # 获取所有文档
            all_docs = self.get_all_documents()
            
            # 筛选匹配的文档
            matching_indices = []
            for i, metadata in enumerate(all_docs.get('metadatas', [])):
                if metadata:
                    # 支持完整路径匹配或文件名匹配
                    source = metadata.get('source', '')
                    file_name = metadata.get('file_name', '')
                    if source == source_path or file_name == source_path or source.endswith(source_path):
                        matching_indices.append(i)
            
            # 返回匹配的文档
            if matching_indices:
                return {
                    'ids': [all_docs['ids'][i] for i in matching_indices],
                    'metadatas': [all_docs['metadatas'][i] for i in matching_indices],
                    'documents': [all_docs['documents'][i] for i in matching_indices]
                }
            else:
                return {'ids': [], 'metadatas': [], 'documents': []}
        except Exception as e:
            print(f"获取文档失败: {str(e)}")
            return {'ids': [], 'metadatas': [], 'documents': []}
    
    def delete_by_source(self, source_path: str) -> int:
        """
        根据文件路径删除所有相关文档chunks
        
        Args:
            source_path: 文件路径（完整路径或文件名）
            
        Returns:
            删除的文档数量
        """
        try:
            # 获取该文件的所有文档
            docs = self.get_documents_by_source(source_path)
            ids = docs.get('ids', [])
            
            if ids:
                self.delete_documents(ids)
                print(f"✓ 已删除文件 {source_path} 的 {len(ids)} 个文档块")
                return len(ids)
            else:
                print(f"未找到文件 {source_path} 的文档")
                return 0
        except Exception as e:
            print(f"删除文档失败: {str(e)}")
            return 0
    
    def list_unique_files(self) -> List[Dict[str, Any]]:
        """
        获取唯一文件列表（聚合chunks）
        
        Returns:
            文件列表，每个文件包含：file_name, file_type, chunk_count, source
        """
        try:
            all_docs = self.get_all_documents()
            metadatas = all_docs.get('metadatas', [])
            
            # 使用字典聚合文件信息
            files_dict = {}
            for metadata in metadatas:
                if metadata:
                    source = metadata.get('source', '')
                    file_name = metadata.get('file_name', '')
                    file_type = metadata.get('file_type', '')
                    
                    if file_name:
                        if file_name not in files_dict:
                            files_dict[file_name] = {
                                'file_name': file_name,
                                'file_type': file_type,
                                'chunk_count': 0,
                                'source': source
                            }
                        files_dict[file_name]['chunk_count'] += 1
            
            # 转换为列表
            files_list = list(files_dict.values())
            
            # 确保所有字段都是字符串类型，转换bytes为str
            for file_info in files_list:
                for key, value in file_info.items():
                    if isinstance(value, bytes):
                        try:
                            file_info[key] = value.decode('utf-8')
                        except UnicodeDecodeError:
                            try:
                                file_info[key] = value.decode('gbk')
                            except UnicodeDecodeError:
                                # 如果都失败，使用repr显示
                                file_info[key] = repr(value)
            
            # 按文件名排序
            files_list.sort(key=lambda x: x['file_name'])
            
            return files_list
        except Exception as e:
            print(f"获取文件列表失败: {str(e)}")
            return []



