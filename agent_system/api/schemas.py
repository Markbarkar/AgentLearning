"""
API 请求和响应模型定义

包含所有 Pydantic 模型
"""

from pydantic import BaseModel, Field
from typing import Optional


# ==================== 任务相关模型 ====================

class TaskRequest(BaseModel):
    """任务请求模型"""
    task: str = Field(..., description="用户任务描述", example="请提取 /path/to/附件九.pdf 的关键信息")
    file_path: Optional[str] = Field(None, description="文件路径（可选）", example="/path/to/附件九.pdf")
    temperature: Optional[float] = Field(None, description="LLM 温度参数（可选，默认使用配置值）", ge=0.0, le=1.0)
    use_rag: Optional[bool] = Field(True, description="是否使用 RAG 增强（默认启用）")
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")


class TaskResponse(BaseModel):
    """任务响应模型"""
    success: bool = Field(..., description="任务是否成功")
    result: str = Field(..., description="Agent 处理结果")
    error: Optional[str] = Field(None, description="错误信息（如果有）")


# ==================== 知识库相关模型 ====================

class KnowledgeBaseBuildRequest(BaseModel):
    """知识库构建请求模型"""
    directory: Optional[str] = Field(None, description="文档目录路径")
    file_paths: Optional[list] = Field(None, description="文件路径列表")
    clear_existing: bool = Field(False, description="是否清空已有数据")
    recursive: bool = Field(True, description="是否递归处理子目录")
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")


class KnowledgeBaseSearchRequest(BaseModel):
    """知识库检索请求模型"""
    query: str = Field(..., description="查询文本")
    top_k: int = Field(3, description="返回的文档数量", ge=1, le=10)
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")


class KnowledgeBaseInfoRequest(BaseModel):
    """知识库信息查询请求模型"""
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")


class KnowledgeBaseClearRequest(BaseModel):
    """知识库清空请求模型"""
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")


# ==================== 文档管理相关模型 ====================

class DocumentListRequest(BaseModel):
    """文档列表查询请求模型"""
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")


class DocumentDetailRequest(BaseModel):
    """文档详情查询请求模型"""
    file_name: str = Field(..., description="文件名", example="附件九.pdf")
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")


class DocumentDeleteRequest(BaseModel):
    """文档删除请求模型"""
    file_name: str = Field(..., description="文件名", example="附件九.pdf")
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="1")

