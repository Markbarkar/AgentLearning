"""
文档管理路由

包含文档的列表、详情、删除、更新等操作
"""

import os
import shutil
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import JSONResponse

from ..schemas import (
    DocumentListRequest,
    DocumentDetailRequest,
    DocumentDeleteRequest,
)
from ..dependencies import get_knowledge_base
from ...config.settings import KNOWLEDGE_BASE_DOCS_DIR


# 创建路由器
router = APIRouter(prefix="/agent/knowledge_base/documents", tags=["文档管理"])


@router.post("/list")
async def list_documents(request: DocumentListRequest):
    """
    列出知识库中的所有文档
    
    Args:
        request: 文档列表查询请求
    
    Returns:
        文档列表
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
        
        result = kb.list_documents()
        result["user_id"] = user_id or "public"
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"获取文档列表失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@router.post("/detail")
async def get_document_detail(request: DocumentDetailRequest):
    """
    获取单个文档的详细信息
    
    Args:
        request: 文档详情查询请求
    
    Returns:
        文档详细信息
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
        
        result = kb.get_document_detail(request.file_name)
        result["user_id"] = user_id or "public"
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"获取文档详情失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@router.post("/delete")
async def delete_document(request: DocumentDeleteRequest):
    """
    删除知识库中的指定文档
    
    Args:
        request: 文档删除请求
    
    Returns:
        删除结果
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
        
        result = kb.delete_document(request.file_name)
        result["user_id"] = user_id or "public"
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"删除文档失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@router.post("/update")
async def update_document(
    file: UploadFile = File(...),
    user_id: str = Form(None),
    file_name: Optional[str] = Form(None)
):
    """
    更新知识库中的指定文档
    
    Args:
        file: 新上传的文件
        user_id: 用户ID（用于知识库隔离）
        file_name: 要替换的原文件名（如果为None，则使用上传文件的文件名）
    
    Returns:
        更新结果
    """
    temp_file = None
    try:
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
        
        # 创建临时文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user_folder = user_id or "public"
        temp_dir = os.path.join(KNOWLEDGE_BASE_DOCS_DIR, "temp_updates", user_folder, timestamp)
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_file = os.path.join(temp_dir, file.filename)
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # 确定要替换的文件名
        target_file_name = file_name if file_name else file.filename
        
        # 执行更新
        result = kb.update_document(temp_file, target_file_name)
        result["user_id"] = user_id or "public"
        result["uploaded_file"] = file.filename
        
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"更新文档失败: {str(e)}",
                "user_id": user_id or "public"
            }
        )
    
    finally:
        # 清理临时文件
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                # 清理空目录
                temp_dir = os.path.dirname(temp_file)
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"清理临时文件失败: {str(e)}")

