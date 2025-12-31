"""
Agent 任务路由

包含工具列表查询和任务处理
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas import TaskRequest, TaskResponse
from ..dependencies import get_agent, create_tools
from ...config.database import get_db


# 创建路由器
router = APIRouter(prefix="/agent", tags=["Agent任务"])


@router.get("/tools")
async def get_tools_info():
    """获取可用工具列表"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in create_tools()
        ]
    }


@router.post("/process_task", response_model=TaskResponse)
async def process_task(request: TaskRequest, db: Session = Depends(get_db)):
    """
    处理用户任务
    
    Args:
        request: 任务请求，包含任务描述和可选的文件路径
        db: 数据库会话，用于加载用户MCP配置
    
    Returns:
        任务处理结果
    """
    try:
        # 如果提供了文件路径，将其添加到任务描述中
        task = request.task
        if request.file_path and request.file_path not in task:
            task = f"{task}\n文件路径: {request.file_path}"
        
        # 获取用户专属 Agent 实例（始终带RAG能力，支持用户MCP配置）
        agent = get_agent(
            user_id=request.user_id,
            temperature=request.temperature,
            db_session=db
        )
        
        # 执行任务，use_rag 在运行时动态决定
        result = agent.run(task, use_rag=request.use_rag)
        
        return TaskResponse(
            success=True,
            result=result,
            error=None
        )
    
    except Exception as e:
        # 返回错误信息
        return TaskResponse(
            success=False,
            result="",
            error=str(e)
        )

