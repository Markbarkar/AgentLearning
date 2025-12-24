"""
知识库管理模块

提供统一的知识库管理接口
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from langchain_core.documents import Document

from .embeddings import QwenEmbeddings
from .vector_store import VectorStoreManager
from .document_processor import DocumentProcessor
from ..config.settings import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    PUBLIC_COLLECTION_NAME,
    ENABLE_USER_ISOLATION,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD
)
import re


class KnowledgeBase:
    """
    知识库管理类
    
    整合嵌入模型、向量数据库和文档处理器
    提供统一的知识库构建、检索和管理接口
    支持多用户隔离，每个用户拥有独立的 collection
    """
    
    @staticmethod
    def get_user_collection_name(user_id: Optional[str]) -> str:
        """
        根据 user_id 生成 collection 名称
        
        Args:
            user_id: 用户ID，如果为 None 则返回公共知识库名称
            
        Returns:
            collection 名称
        """
        if not ENABLE_USER_ISOLATION or not user_id:
            return PUBLIC_COLLECTION_NAME
        
        # 清理 user_id，只保留字母数字下划线，限制长度
        cleaned_user_id = re.sub(r'[^a-zA-Z0-9_]', '', str(user_id))[:32]
        if not cleaned_user_id:
            return PUBLIC_COLLECTION_NAME
            
        return f"user_{cleaned_user_id}_documents"
    
    @staticmethod
    def validate_user_id(user_id: Optional[str]) -> Optional[str]:
        """
        验证并清理 user_id
        
        Args:
            user_id: 原始用户ID
            
        Returns:
            清理后的用户ID，如果无效则返回 None
        """
        if not user_id:
            return None
        
        # 只保留字母数字下划线，限制长度为32字符
        cleaned = re.sub(r'[^a-zA-Z0-9_]', '', str(user_id))[:32]
        return cleaned if cleaned else None
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        embeddings: Optional[QwenEmbeddings] = None,
        persist_directory: str = CHROMA_PERSIST_DIR,
        collection_name: Optional[str] = None,
        vl_tools=None
    ):
        """
        初始化知识库
        
        Args:
            user_id: 用户ID，用于实现多用户隔离
            embeddings: 嵌入模型实例，如果为 None 则自动创建
            persist_directory: 向量数据库持久化目录
            collection_name: 集合名称（已废弃，优先使用 user_id 生成）
            vl_tools: Qwen2.5-VL 工具实例（用于 PDF OCR）
        """
        # 保存用户ID
        self.user_id = self.validate_user_id(user_id)
        
        # 初始化嵌入模型
        if embeddings is None:
            embeddings = QwenEmbeddings()
        self.embeddings = embeddings
        
        # 根据 user_id 确定 collection 名称
        if collection_name is None:
            collection_name = self.get_user_collection_name(self.user_id)
        
        # 初始化向量数据库
        self.vector_store = VectorStoreManager(
            embeddings=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # 初始化文档处理器
        self.document_processor = DocumentProcessor(vl_tools=vl_tools)
    
    def build_from_directory(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        从目录构建知识库
        
        Args:
            directory: 文档目录路径
            recursive: 是否递归处理子目录
            file_extensions: 要处理的文件扩展名列表
            clear_existing: 是否清空已有数据
            
        Returns:
            构建统计信息
        """
        print(f"\n{'='*60}")
        print(f"开始构建知识库: {directory}")
        print(f"{'='*60}\n")
        
        # 清空已有数据（如果需要）
        if clear_existing:
            print("清空已有数据...")
            self.vector_store.clear_collection()
        
        # print('directory', directory)
        
        # 处理文档
        documents = self.document_processor.process_directory(
            directory=directory,
            recursive=recursive,
            file_extensions=file_extensions
        )
        
        if not documents:
            # print("\n未识别到信息")
            return {
                "success": False,
                "message": "暂未识别到信息",
                "documents_processed": 0,
                "chunks_added": 0
            }
        
        # 添加到向量数据库
        print(f"\n将文档添加到向量数据库...")
        ids = self.vector_store.add_documents(documents)
        
        # 统计信息
        total_count = self.vector_store.get_collection_count()
        
        print(f"\n{'='*60}")
        print(f"知识库构建完成")
        print(f"{'='*60}")
        print(f"文档块数量: {len(documents)}")
        print(f"向量数据库总数: {total_count}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "message": "知识库构建成功",
            "chunks_added": len(documents),
            "total_count": total_count,
            "document_ids": ids
        }
    
    def build_from_files(
        self,
        file_paths: List[str],
        clear_existing: bool = False
    ) -> Dict[str, Any]:
        """
        从文件列表构建知识库
        
        Args:
            file_paths: 文件路径列表
            clear_existing: 是否清空已有数据
            
        Returns:
            构建统计信息
        """
        print(f"\n{'='*60}")
        print(f"开始构建知识库（{len(file_paths)} 个文件）")
        print(f"{'='*60}\n")
        
        # 清空已有数据（如果需要）
        if clear_existing:
            print("清空已有数据...")
            self.vector_store.clear_collection()
        
        # 处理所有文件
        all_documents = []
        for i, file_path in enumerate(file_paths, 1):
            try:
                print(f"[{i}/{len(file_paths)}] 处理文件: {Path(file_path).name}")
                documents = self.document_processor.process_file(file_path)
                all_documents.extend(documents)
                print(f"  ✓ 生成 {len(documents)} 个文档块")
            except Exception as e:
                print(f"  ✗ 处理失败: {str(e)}")
        
        if not all_documents:
            print("\n未能处理任何文档")
            return {
                "success": False,
                "message": "未能处理任何文档",
                "documents_processed": 0,
                "chunks_added": 0
            }
        
        # 添加到向量数据库
        print(f"\n将文档添加到向量数据库...")
        ids = self.vector_store.add_documents(all_documents)
        
        # 统计信息
        total_count = self.vector_store.get_collection_count()
        
        print(f"\n{'='*60}")
        print(f"知识库构建完成")
        print(f"{'='*60}")
        print(f"文档块数量: {len(all_documents)}")
        print(f"向量数据库总数: {total_count}")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "message": "知识库构建成功",
            "chunks_added": len(all_documents),
            "total_count": total_count,
            "document_ids": ids
        }
    
    def search(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        with_score: bool = True,
        score_threshold: float = RAG_SIMILARITY_THRESHOLD
    ) -> List[Dict[str, Any]]:
        """
        检索知识库
        
        Args:
            query: 查询文本
            top_k: 返回的文档数量
            with_score: 是否返回相似度分数
            score_threshold: 相似度阈值（仅在 with_score=True 时有效）
            
        Returns:
            检索结果列表
        """
        if with_score:
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k
            )
            
            # 格式化结果并过滤低分结果
            formatted_results = []
            for doc, score in results:
                # Chroma 使用距离度量，距离越小越相似
                # 转换为相似度分数（0-1，越大越相似）
                similarity = 1 / (1 + score)
                
                if similarity >= score_threshold:
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "similarity": similarity,
                        "distance": score
                    })
            
            return formatted_results
        else:
            results = self.vector_store.similarity_search(
                query=query,
                k=top_k
            )
            
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
    
    def get_info(self) -> Dict[str, Any]:
        """
        获取知识库信息
        
        Returns:
            知识库统计信息
        """
        total_count = self.vector_store.get_collection_count()
        
        # 获取文档数量（唯一文件数量）
        unique_files = self.vector_store.list_unique_files()
        document_count = len(unique_files)
        
        return {
            "user_id": self.user_id or "public",
            "collection_name": self.vector_store.collection_name,
            # "persist_directory": self.vector_store.persist_directory,
            "chunks_count": total_count,
            "document_count": document_count,
            # "embedding_model": self.embeddings.model
        }
    
    def clear(self) -> None:
        """
        清空知识库
        """
        self.vector_store.clear_collection()
    
    def list_documents(self) -> Dict[str, Any]:
        """
        列出所有文档（基础信息）
        
        Returns:
            包含文档列表的字典
        """
        try:
            files = self.vector_store.list_unique_files()
            
            # 格式化返回信息
            documents = []
            for file_info in files:
                documents.append({
                    "file_name": file_info['file_name'],
                    "file_type": file_info['file_type'],
                    "chunk_count": file_info['chunk_count']
                })
            
            return {
                "success": True,
                "documents": documents,
                "total_files": len(documents)
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"获取文档列表失败: {str(e)}",
                "documents": [],
                "total_files": 0
            }
    
    def get_document_detail(self, file_name: str) -> Dict[str, Any]:
        """
        获取单个文档详情
        
        Args:
            file_name: 文件名
            
        Returns:
            文档详细信息
        """
        try:
            # 获取该文件的所有chunks
            docs = self.vector_store.get_documents_by_source(file_name)
            
            if not docs.get('ids'):
                return {
                    "success": False,
                    "message": f"未找到文档: {file_name}"
                }
            
            # 提取基本信息
            metadatas = docs.get('metadatas', [])
            documents = docs.get('documents', [])
            
            if not metadatas:
                return {
                    "success": False,
                    "message": "文档数据异常"
                }
            
            first_metadata = metadatas[0]
            
            # 构建chunks预览（每个chunk显示前100个字符）
            chunks = []
            for i, (metadata, content) in enumerate(zip(metadatas, documents)):
                chunk_id = metadata.get('chunk_id', i)
                preview = content[:100] + "..." if len(content) > 100 else content
                chunks.append({
                    "chunk_id": chunk_id,
                    "preview": preview
                })
            
            # 按chunk_id排序
            chunks.sort(key=lambda x: x['chunk_id'])
            
            return {
                "success": True,
                "file_name": first_metadata.get('file_name', ''),
                "file_type": first_metadata.get('file_type', ''),
                "chunk_count": len(chunks),
                "source_path": first_metadata.get('source', ''),
                "chunks": chunks
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"获取文档详情失败: {str(e)}"
            }
    
    def delete_document(self, file_name: str) -> Dict[str, Any]:
        """
        删除指定文档
        
        Args:
            file_name: 文件名
            
        Returns:
            删除结果
        """
        try:
            # 删除该文件的所有chunks
            deleted_count = self.vector_store.delete_by_source(file_name)
            
            if deleted_count > 0:
                return {
                    "success": True,
                    "message": f"成功删除文档: {file_name}",
                    "deleted_chunks": deleted_count
                }
            else:
                return {
                    "success": False,
                    "message": f"未找到文档: {file_name}",
                    "deleted_chunks": 0
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"删除文档失败: {str(e)}",
                "deleted_chunks": 0
            }
    
    def update_document(self, file_path: str, file_name: Optional[str] = None) -> Dict[str, Any]:
        """
        更新指定文档（先删除再重新添加）
        
        Args:
            file_path: 新文件路径
            file_name: 要替换的文件名（如果为None，则使用file_path的文件名）
            
        Returns:
            更新结果
        """
        try:
            # 确定要替换的文件名
            if file_name is None:
                file_name = Path(file_path).name
            
            # 1. 先删除旧文档
            print(f"删除旧文档: {file_name}")
            delete_result = self.delete_document(file_name)
            
            # 2. 处理新文档
            print(f"处理新文档: {file_path}")
            documents = self.document_processor.process_file(file_path)
            
            if not documents:
                return {
                    "success": False,
                    "message": "新文档处理失败",
                    "deleted_chunks": delete_result.get('deleted_chunks', 0),
                    "added_chunks": 0
                }
            
            # 3. 添加到向量数据库
            print(f"添加新文档到向量数据库...")
            ids = self.vector_store.add_documents(documents)
            
            return {
                "success": True,
                "message": f"成功更新文档: {file_name}",
                "deleted_chunks": delete_result.get('deleted_chunks', 0),
                "added_chunks": len(documents),
                "document_ids": ids
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"更新文档失败: {str(e)}"
            }



