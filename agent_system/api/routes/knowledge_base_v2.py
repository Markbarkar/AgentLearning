"""
增强知识库管理路由 (V2)

使用新的 RAG 架构：
- 预处理层：法律分块器 + 元数据提取
- 向量化层：嵌入服务 + 缓存
- 存储层：Milvus/ChromaDB + BM25
- 检索层：混合检索 + 重排序
"""

import os
import shutil
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from ..schemas import (
    EnhancedKBBuildRequest,
    EnhancedSearchRequest,
    CompareSearchRequest,
    KnowledgeBaseInfoRequest,
    KnowledgeBaseClearRequest,
)
from ..dependencies import get_knowledge_base
from ...config.settings import KNOWLEDGE_BASE_DOCS_DIR, QWEN_EMBEDDING_API_KEY


# 创建路由器
router = APIRouter(prefix="/agent/knowledge_base_v2", tags=["增强知识库 (V2)"])


# ==================== 缓存：增强知识库实例 ====================

_enhanced_kb_instances: Dict[str, Any] = {}


def get_enhanced_kb(user_id: Optional[str] = None):
    """
    获取增强知识库实例
    
    使用新的 RAG 架构组件
    """
    global _enhanced_kb_instances
    
    cache_key = user_id or "public"
    
    if cache_key not in _enhanced_kb_instances:
        try:
            from ...rag.stores import create_vector_store, create_bm25_index, StoreProvider
            from ...rag.embedding import EmbeddingService
            from ...rag.retrieval import RetrievalPipeline, RetrievalConfig
            
            # 创建向量存储（优先 Milvus，回退 ChromaDB）
            try:
                vector_store = create_vector_store(
                    provider=StoreProvider.MILVUS,
                    user_id=user_id,
                    collection_prefix="enhanced_kb",
                    host="localhost",
                    port=19530
                )
                store_type = "milvus"
            except Exception as e:
                print(f"Milvus 不可用，回退到 ChromaDB: {e}")
                vector_store = create_vector_store(
                    provider=StoreProvider.CHROMA,
                    user_id=user_id,
                    collection_prefix="enhanced_kb",
                    persist_directory="./data/enhanced_chroma"
                )
                store_type = "chroma"
            
            # 创建 BM25 索引
            bm25_index = create_bm25_index(
                user_id=user_id,
                collection_prefix="enhanced_bm25",
                persist_directory="./data/enhanced_bm25"
            )
            
            # 创建嵌入服务
            embedding_service = EmbeddingService()
            
            # 创建检索管道
            config = RetrievalConfig(
                use_vector=True,
                use_bm25=True,
                enable_rerank=True,
                fusion_method="rrf",
                top_k=10,
                fetch_k=50,
            )
            
            retrieval_pipeline = RetrievalPipeline(
                vector_store=vector_store,
                bm25_index=bm25_index,
                embedding_service=embedding_service,
                config=config
            )
            
            _enhanced_kb_instances[cache_key] = {
                "vector_store": vector_store,
                "bm25_index": bm25_index,
                "embedding_service": embedding_service,
                "retrieval_pipeline": retrieval_pipeline,
                "store_type": store_type,
            }
            
            user_label = f"用户 {user_id}" if user_id else "公共"
            print(f"✓ {user_label}增强知识库初始化成功 (存储: {store_type})")
            
        except Exception as e:
            print(f"✗ 增强知识库初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    return _enhanced_kb_instances.get(cache_key)


def clear_enhanced_kb_cache(user_id: Optional[str] = None):
    """清除增强知识库缓存"""
    global _enhanced_kb_instances
    
    if user_id:
        _enhanced_kb_instances.pop(user_id, None)
    else:
        _enhanced_kb_instances.clear()


# ==================== API 端点 ====================

@router.post("/build")
async def build_enhanced_kb(
    files: Optional[List[UploadFile]] = File(None),
    user_id: str = Form(...),
    directory: Optional[str] = Form(None),
    clear_existing: bool = Form(False),
    chunker_type: str = Form("legal"),
    chunk_size: int = Form(800),
    enable_bm25: bool = Form(True),
):
    """
    构建增强知识库
    
    使用新的预处理层（法律分块器 + 元数据提取）
    
    Args:
        files: 上传的文件列表
        user_id: 用户ID
        directory: 文档目录（管理员模式）
        clear_existing: 是否清空已有数据
        chunker_type: 分块器类型 (legal/recursive)
        chunk_size: 分块大小
        enable_bm25: 是否启用 BM25 索引
    """
    temp_dir = None
    start_time = time.time()
    
    try:
        # 获取增强知识库实例
        kb = get_enhanced_kb(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "增强知识库初始化失败",
                    "user_id": user_id or "public"
                }
            )
        
        vector_store = kb["vector_store"]
        bm25_index = kb["bm25_index"]
        embedding_service = kb["embedding_service"]
        
        # 清空已有数据
        if clear_existing:
            vector_store.clear_collection()
            if enable_bm25:
                bm25_index.clear()
        
        # 处理文件上传
        source_dir = None
        if files and len(files) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_folder = user_id or "public"
            temp_dir = os.path.join(KNOWLEDGE_BASE_DOCS_DIR, "temp_uploads_v2", user_folder, timestamp)
            os.makedirs(temp_dir, exist_ok=True)
            
            for file in files:
                if file.filename:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
            
            source_dir = temp_dir
        elif directory:
            source_dir = directory
        else:
            source_dir = KNOWLEDGE_BASE_DOCS_DIR
        
        if not os.path.exists(source_dir):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"目录不存在: {source_dir}",
                    "user_id": user_id or "public"
                }
            )
        
        # 导入预处理组件
        from ...rag.chunkers import LegalChunker, RecursiveChunker
        from ...rag.extractors import LegalMetadataExtractor
        from ...rag.document_processor import DocumentProcessor
        
        # 选择分块器
        if chunker_type == "legal":
            chunker = LegalChunker(max_chunk_size=chunk_size, min_chunk_size=100)
        else:
            chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=100)
        
        # 元数据提取器
        extractor = LegalMetadataExtractor()
        
        # 文档处理器（用于读取文件）
        doc_processor = DocumentProcessor()
        
        # 处理文件
        all_chunks = []
        processed_files = []
        
        for root, dirs, file_list in os.walk(source_dir):
            for filename in file_list:
                file_path = os.path.join(root, filename)
                
                # 文件名类型校验交给doc_processor处理
                # ext = Path(filename).suffix.lower()
                
                # if ext not in [".txt", ".pdf", ".docx", ".doc", ".md"]:
                #     continue
                
                try:
                    # 读取文件内容
                    text = doc_processor._extract_text_from_file(file_path)
                    if not text:
                        continue
                    
                    # 提取元数据
                    metadata = extractor.extract(file_path, text)
                    metadata["source"] = file_path
                    metadata["file_name"] = filename
                    
                    # 分块
                    chunks = chunker.chunk(text, metadata)
                    all_chunks.extend(chunks)
                    processed_files.append(filename)
                    
                except Exception as e:
                    print(f"处理文件失败 {filename}: {e}")
        
        if not all_chunks:
            return JSONResponse(
                content={
                    "success": False,
                    "message": "未能处理任何文档",
                    "user_id": user_id or "public"
                }
            )
        
        # 生成嵌入向量
        texts = [chunk.page_content for chunk in all_chunks]
        embeddings = embedding_service.embed_documents(texts)
        
        # 存储到向量数据库
        doc_ids = vector_store.add_documents(all_chunks, embeddings)
        
        # 存储到 BM25 索引
        if enable_bm25:
            bm25_index.add_documents(all_chunks, ids=doc_ids)
        
        elapsed_time = time.time() - start_time
        
        return JSONResponse(content={
            "success": True,
            "message": "增强知识库构建成功",
            "user_id": user_id or "public",
            "store_type": kb["store_type"],
            "chunker_type": chunker_type,
            "files_processed": len(processed_files),
            "chunks_added": len(all_chunks),
            "total_count": vector_store.get_collection_count(),
            "bm25_count": bm25_index.get_count() if enable_bm25 else 0,
            "elapsed_seconds": round(elapsed_time, 2),
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"构建失败: {str(e)}",
                "user_id": user_id or "public"
            }
        )
    
    finally:
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


@router.post("/search")
async def enhanced_search(request: EnhancedSearchRequest):
    """
    增强检索
    
    使用混合检索（向量 + BM25）和重排序
    
    Args:
        request: 检索请求
    
    Returns:
        检索结果（包含详细的检索信息）
    """
    start_time = time.time()
    
    try:
        kb = get_enhanced_kb(request.user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "增强知识库未初始化",
                    "user_id": request.user_id or "public"
                }
            )
        
        pipeline = kb["retrieval_pipeline"]
        
        # 更新配置
        pipeline.update_config(
            use_vector=request.use_vector,
            use_bm25=request.use_bm25,
            enable_rerank=request.enable_rerank,
            fusion_method=request.fusion_method,
            similarity_threshold=request.similarity_threshold,
            top_k=request.top_k,
        )
        
        # 执行检索
        result = pipeline.search_with_context(request.query, top_k=request.top_k)
        
        # 格式化结果
        formatted_results = []
        for r in result["results"]:
            formatted_results.append({
                "content": r.content,
                "metadata": r.metadata,
                "score": round(r.score, 4),
                "vector_score": round(r.vector_score, 4) if r.vector_score else None,
                "bm25_score": round(r.bm25_score, 4) if r.bm25_score else None,
                "rerank_score": round(r.rerank_score, 4) if r.rerank_score else None,
                "source": r.source,
            })
        
        elapsed_time = time.time() - start_time
        
        return JSONResponse(content={
            "success": True,
            "user_id": request.user_id or "public",
            "query": request.query,
            "parsed_query": result["parsed_query"],
            "results": formatted_results,
            "total_found": result["total_found"],
            "search_config": {
                "use_vector": request.use_vector,
                "use_bm25": request.use_bm25,
                "enable_rerank": request.enable_rerank,
                "fusion_method": request.fusion_method,
            },
            "elapsed_seconds": round(elapsed_time, 3),
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"检索失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@router.post("/compare")
async def compare_search(request: CompareSearchRequest):
    """
    对比检索
    
    同时使用旧版和新版 RAG 架构进行检索，返回对比结果
    
    Args:
        request: 对比检索请求
    
    Returns:
        新旧架构的对比结果
    """
    results = {
        "query": request.query,
        "user_id": request.user_id or "public",
        "comparison": {}
    }
    
    # 1. 旧版 RAG 检索
    try:
        old_start = time.time()
        old_kb = get_knowledge_base(request.user_id)
        
        if old_kb and old_kb.vector_store.get_collection_count() > 0:
            old_results = old_kb.search(
                query=request.query,
                top_k=request.top_k,
                with_score=True
            )
            old_elapsed = time.time() - old_start
            
            results["comparison"]["old_rag"] = {
                "success": True,
                "results": old_results,
                "total_found": len(old_results),
                "elapsed_seconds": round(old_elapsed, 3),
                "architecture": "ChromaDB + 基础向量检索"
            }
        else:
            results["comparison"]["old_rag"] = {
                "success": False,
                "message": "旧版知识库为空或未初始化",
                "results": [],
                "total_found": 0,
            }
    except Exception as e:
        results["comparison"]["old_rag"] = {
            "success": False,
            "message": str(e),
            "results": [],
            "total_found": 0,
        }
    
    # 2. 新版 RAG 检索
    try:
        new_start = time.time()
        new_kb = get_enhanced_kb(request.user_id)
        
        if new_kb:
            pipeline = new_kb["retrieval_pipeline"]
            
            # 使用默认配置进行混合检索
            search_result = pipeline.search_with_context(
                request.query, 
                top_k=request.top_k
            )
            
            new_elapsed = time.time() - new_start
            
            formatted_results = []
            for r in search_result["results"]:
                formatted_results.append({
                    "content": r.content,
                    "metadata": r.metadata,
                    "score": round(r.score, 4),
                    "vector_score": round(r.vector_score, 4) if r.vector_score else None,
                    "bm25_score": round(r.bm25_score, 4) if r.bm25_score else None,
                })
            
            results["comparison"]["new_rag"] = {
                "success": True,
                "results": formatted_results,
                "total_found": len(formatted_results),
                "elapsed_seconds": round(new_elapsed, 3),
                "parsed_query": search_result["parsed_query"],
                "architecture": f"{new_kb['store_type'].upper()} + BM25 混合检索 + 重排序",
            }
        else:
            results["comparison"]["new_rag"] = {
                "success": False,
                "message": "新版知识库未初始化",
                "results": [],
                "total_found": 0,
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        results["comparison"]["new_rag"] = {
            "success": False,
            "message": str(e),
            "results": [],
            "total_found": 0,
        }
    
    results["success"] = True
    return JSONResponse(content=results)


@router.post("/info")
async def get_enhanced_kb_info(request: KnowledgeBaseInfoRequest):
    """
    获取增强知识库信息
    """
    try:
        kb = get_enhanced_kb(request.user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "增强知识库未初始化",
                    "user_id": request.user_id or "public"
                }
            )
        
        vector_store = kb["vector_store"]
        bm25_index = kb["bm25_index"]
        
        return JSONResponse(content={
            "success": True,
            "user_id": request.user_id or "public",
            "store_type": kb["store_type"],
            "collection_name": vector_store.collection_name,
            "vector_count": vector_store.get_collection_count(),
            "bm25_count": bm25_index.get_count(),
            "unique_files": vector_store.list_unique_files() if hasattr(vector_store, 'list_unique_files') else [],
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"获取信息失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@router.post("/clear")
async def clear_enhanced_kb(request: KnowledgeBaseClearRequest):
    """
    清空增强知识库
    """
    try:
        kb = get_enhanced_kb(request.user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "增强知识库未初始化",
                    "user_id": request.user_id or "public"
                }
            )
        
        vector_store = kb["vector_store"]
        bm25_index = kb["bm25_index"]
        
        vector_store.clear_collection()
        bm25_index.clear()
        
        return JSONResponse(content={
            "success": True,
            "message": "增强知识库已清空",
            "user_id": request.user_id or "public"
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"清空失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@router.get("/health")
async def health_check():
    """
    健康检查
    
    检查新 RAG 架构组件是否可用
    """
    status = {
        "milvus": False,
        "chroma": False,
        "bm25": False,
        "embedding": False,
    }
    
    # 检查 Milvus
    try:
        from pymilvus import connections
        connections.connect(alias="health_check", host="localhost", port=19530)
        connections.disconnect("health_check")
        status["milvus"] = True
    except:
        pass
    
    # 检查 ChromaDB
    try:
        import chromadb
        status["chroma"] = True
    except:
        pass
    
    # 检查 BM25
    try:
        from rank_bm25 import BM25Okapi
        status["bm25"] = True
    except:
        pass
    
    # 检查嵌入服务
    try:
        if QWEN_EMBEDDING_API_KEY:
            status["embedding"] = True
    except:
        pass
    
    all_ok = all(status.values())
    
    return JSONResponse(content={
        "success": True,
        "status": status,
        "all_components_available": all_ok,
        "recommended_store": "milvus" if status["milvus"] else "chroma",
    })
