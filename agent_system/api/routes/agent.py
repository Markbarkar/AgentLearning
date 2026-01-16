"""
Agent 任务路由

包含工具列表查询和任务处理
支持意图识别、工具选择和路由分发
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..schemas import TaskRequest, TaskResponse
from ..dependencies import (
    get_agent,
    get_all_tools,
    get_intent_classifier,
    get_intent_router,
    get_tool_selector,
    get_knowledge_base,
    get_handler_factory,
)
from ...config.database import get_db
from ...tools.context import set_context, clear_context


# 创建路由器
router = APIRouter(prefix="/agent", tags=["Agent任务"])


@router.get("/tools")
async def get_tools_info():
    """获取可用工具列表"""
    return {
        "tools": [
            {
                "name": getattr(tool, 'name', str(tool)),
                "description": getattr(tool, 'description', ''),
            }
            for tool in get_all_tools()
        ]
    }


@router.get("/tool_groups")
async def get_tool_groups():
    """
    获取工具分组信息
    
    返回所有工具组及其包含的工具
    """
    selector = get_tool_selector()
    
    groups = []
    for name, desc in selector.get_available_groups().items():
        tools = selector.get_tools_in_group(name)
        groups.append({
            "name": name,
            "description": desc,
            "tools": tools,
            "tool_count": len(tools),
        })
    
    return {"groups": groups}


@router.get("/intent_handlers")
async def get_intent_handlers():
    """
    获取可用的意图处理器列表
    
    返回所有已注册的 Handler 信息
    """
    from ...core.router import HandlerRegistry
    
    handlers = []
    for intent_type in HandlerRegistry.get_registered_types():
        metadata = HandlerRegistry.get_metadata(intent_type) or {}
        handlers.append({
            "intent_type": intent_type,
            "class_name": metadata.get("class_name", ""),
            "priority": metadata.get("priority", 0),
            "description": metadata.get("description", ""),
        })
    
    return {
        "handlers": sorted(handlers, key=lambda h: h["priority"], reverse=True)
    }


@router.post("/process_task", response_model=TaskResponse)
async def process_task(request: TaskRequest, db: Session = Depends(get_db)):
    """
    处理用户任务（使用意图识别、工具选择和路由分发）
    
    流程：
    1. 意图分类：识别用户意图类型
    2. 工具选择：根据意图动态选择相关工具
    3. 路由分发：根据意图类型分发到对应的 Handler
    4. 处理执行：Handler 使用精选工具处理请求
    
    Args:
        request: 任务请求，包含任务描述和可选的文件路径
        db: 数据库会话，用于加载用户MCP配置
    
    Returns:
        任务处理结果
    """
    try:
        # 设置工具执行上下文
        set_context(
            user_id=request.user_id,
            token=request.token
        )
        
        # 构建任务描述
        task = request.task
        if request.file_path and request.file_path not in task:
            task = f"{task}\n文件路径: {request.file_path}"
        
        # ========== 1. 意图识别 ==========
        classifier = get_intent_classifier(use_llm=True)
        intent = classifier.classify(task)
        
        # ========== 2. 工具选择 ==========
        all_tools = get_all_tools(request.user_id, db)
        selector = get_tool_selector()
        selected_tools = selector.select_tools(intent, all_tools)
        
        # 将精选工具和其他依赖注入到 intent
        intent.selected_tools = selected_tools
        intent.matched_groups = list(selector.config.match_groups_from_query(task))
        
        # 添加用户级依赖到 metadata
        if request.use_rag is not None:
            intent.metadata["use_rag"] = request.use_rag
        intent.metadata["knowledge_base"] = get_knowledge_base(request.user_id)
        
        # ========== 3. 路由分发 ==========
        # 使用带有用户依赖的工厂创建路由器
        from ...core.router import IntentRouter
        factory = get_handler_factory(request.user_id, request.temperature, db)
        intent_router = IntentRouter(factory=factory)
        
        result = intent_router.route(intent)
        
        return TaskResponse(
            success=True,
            result=result,
            error=None
        )
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return TaskResponse(
            success=False,
            result="",
            error=str(e)
        )
    
    finally:
        clear_context()


@router.post("/process_task_legacy", response_model=TaskResponse)
async def process_task_legacy(request: TaskRequest, db: Session = Depends(get_db)):
    """
    处理用户任务（传统模式，直接使用 ReAct Agent）
    
    保留原有接口，不使用意图识别，直接调用 ReAct Agent
    
    Args:
        request: 任务请求，包含任务描述和可选的文件路径
        db: 数据库会话，用于加载用户MCP配置
    
    Returns:
        任务处理结果
    """
    try:
        set_context(
            user_id=request.user_id,
            token=request.token
        )
        
        task = request.task
        if request.file_path and request.file_path not in task:
            task = f"{task}\n文件路径: {request.file_path}"
        
        agent = get_agent(
            user_id=request.user_id,
            temperature=request.temperature,
            db_session=db
        )
        
        result = agent.run(task, use_rag=request.use_rag)
        
        return TaskResponse(
            success=True,
            result=result,
            error=None
        )
    
    except Exception as e:
        return TaskResponse(
            success=False,
            result="",
            error=str(e)
        )
    
    finally:
        clear_context()


@router.post("/classify_intent")
async def classify_intent(request: TaskRequest, db: Session = Depends(get_db)):
    """
    仅进行意图分类和工具选择（不执行任务）
    
    用于调试和测试意图分类效果
    
    Args:
        request: 任务请求
        db: 数据库会话
    
    Returns:
        意图分类和工具选择结果
    """
    try:
        task = request.task
        if request.file_path and request.file_path not in task:
            task = f"{task}\n文件路径: {request.file_path}"
        
        # 意图分类
        classifier = get_intent_classifier(use_llm=True)
        intent = classifier.classify(task)
        
        # 工具选择
        all_tools = get_all_tools(request.user_id, db)
        selector = get_tool_selector()
        selected_tools = selector.select_tools(intent, all_tools)
        
        intent.selected_tools = selected_tools
        intent.matched_groups = list(selector.config.match_groups_from_query(task))
        
        return {
            "success": True,
            "intent": intent.to_dict(),
            "all_tools_count": len(all_tools),
            "selected_tools_count": len(selected_tools),
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }


@router.post("/select_tools")
async def select_tools(request: TaskRequest, db: Session = Depends(get_db)):
    """
    仅进行工具选择（调试用）
    
    Args:
        request: 任务请求
        db: 数据库会话
    
    Returns:
        工具选择结果
    """
    try:
        task = request.task
        
        # 获取全部工具
        all_tools = get_all_tools(request.user_id, db)
        
        # 先进行意图分类
        classifier = get_intent_classifier(use_llm=True)
        intent = classifier.classify(task)
        
        # 工具选择
        selector = get_tool_selector()
        selected_tools = selector.select_tools(intent, all_tools)
        
        return {
            "success": True,
            "intent_type": intent.type,
            "matched_groups": selector.config.match_groups_from_query(task),
            "all_tools": [getattr(t, 'name', str(t)) for t in all_tools],
            "selected_tools": [getattr(t, 'name', str(t)) for t in selected_tools],
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
        }
