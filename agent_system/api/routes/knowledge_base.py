"""
知识库管理路由

包含知识库的构建、查询、检索、清空等操作
"""

import os
import shutil
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from ..schemas import (
    KnowledgeBaseSearchRequest,
    KnowledgeBaseInfoRequest,
    KnowledgeBaseClearRequest,
)
from ..dependencies import get_knowledge_base
from ...config.settings import KNOWLEDGE_BASE_DOCS_DIR


# 创建路由器
router = APIRouter(prefix="/agent/knowledge_base", tags=["知识库管理"])


@router.post("/build")
async def build_knowledge_base(
    files: Optional[List[UploadFile]] = File(None),
    user_id: str = Form(...),
    directory: Optional[str] = Form(None),
    file_paths: Optional[str] = Form(None),  # JSON字符串
    clear_existing: bool = Form(False),
    recursive: bool = Form(True)
):
    """
    构建/更新知识库
    
    支持两种模式:
    1. 文件上传模式(推荐): 通过 multipart/form-data 上传文件
    2. 路径模式(管理员): 通过表单参数指定 directory 或 file_paths
    
    Args:
        files: 上传的文件列表(可选)
        user_id: 用户ID(可选,用于知识库隔离)
        directory: 文档目录路径(可选,管理员模式)
        file_paths: 文件路径列表JSON字符串(可选,管理员模式)
        clear_existing: 是否清空已有数据
        recursive: 是否递归处理子目录
    
    Returns:
        构建结果
    """
    temp_dir = None
    try:
        # 获取用户专属知识库实例
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库初始化失败",
                    "user_id": user_id or "public"
                }
            )
        
        # 优先处理文件上传
        if files and len(files) > 0:
            # 创建临时目录: temp_uploads/{user_id or 'public'}/{timestamp}/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_folder = user_id or "public"
            temp_dir = os.path.join(KNOWLEDGE_BASE_DOCS_DIR, "temp_uploads", user_folder, timestamp)
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存所有上传的文件
            saved_files = []
            for file in files:
                if file.filename:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
                    saved_files.append(file.filename)
            
            # 使用临时目录构建知识库
            result = kb.build_from_directory(
                directory=temp_dir,
                recursive=recursive,
                clear_existing=clear_existing
            )
            result["uploaded_files"] = saved_files
            result["file_count"] = len(saved_files)
        
        # 降级处理: 从目录构建
        elif directory:
            result = kb.build_from_directory(
                directory=directory,
                recursive=recursive,
                clear_existing=clear_existing
            )
        
        # 降级处理: 从文件列表构建
        elif file_paths:
            import json
            try:
                paths_list = json.loads(file_paths)
            except:
                paths_list = [file_paths]  # 单个路径
            
            result = kb.build_from_files(
                file_paths=paths_list,
                clear_existing=clear_existing
            )
        
        else:
            # 使用默认目录
            result = kb.build_from_directory(
                directory=KNOWLEDGE_BASE_DOCS_DIR,
                recursive=recursive,
                clear_existing=clear_existing
            )
        
        # 在响应中包含用户ID信息
        result["user_id"] = user_id or "public"
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"构建失败: {str(e)}",
                "user_id": user_id or "public"
            }
        )
    
    finally:
        # 清理临时文件和目录
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                # 同时清理空的父目录
                parent_dir = os.path.dirname(temp_dir)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            except Exception as e:
                print(f"清理临时目录失败: {str(e)}")


@router.post("/info")
async def get_knowledge_base_info(request: KnowledgeBaseInfoRequest):
    """
    获取知识库信息
    
    Args:
        request: 知识库信息查询请求
    
    Returns:
        知识库统计信息
    """
    try:
        user_id = request.user_id
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库未初始化",
                    "user_id": user_id or "public"
                }
            )
        
        info = kb.get_info()
        return JSONResponse(content={
            "success": True,
            **info
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


@router.post("/search")
async def search_knowledge_base(request: KnowledgeBaseSearchRequest):
    """
    检索知识库
    
    Args:
        request: 检索请求
    
    Returns:
        检索结果
    """
    try:
        # 获取用户专属知识库实例
        user_id = request.user_id
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库未初始化",
                    "user_id": user_id or "public"
                }
            )
        
        results = kb.search(
            query=request.query,
            top_k=request.top_k,
            with_score=True
        )
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id or "public",
            "query": request.query,
            "results": results
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"检索失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@router.post("/clear")
async def clear_knowledge_base(request: KnowledgeBaseClearRequest):
    """
    清空知识库
    
    Args:
        request: 知识库清空请求
    
    Returns:
        操作结果
    """
    try:
        user_id = request.user_id
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库未初始化",
                    "user_id": user_id or "public"
                }
            )
        
        kb.clear()
        
        return JSONResponse(content={
            "success": True,
            "message": "知识库已清空",
            "user_id": user_id or "public"
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

