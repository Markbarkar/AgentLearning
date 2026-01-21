"""
API 共享依赖

包含全局变量、缓存实例和工具创建函数
采用共享核心组件策略，仅隔离用户数据（知识库、MCP配置）
"""

from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI

from ..config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_SEED,
)
from ..tools import Qwen25VLTools, finish_tool, create_rag_search_tool
from ..core import Agent
from ..rag import KnowledgeBase


# ==================== 全局共享组件（单例） ====================

_llm_instance = None              # 共享 LLM 实例
_vl_tools_instance = None         # 共享 VL 工具实例
_all_tools = None                 # 全部工具列表（不含用户 MCP）
_classifier_instance = None       # 共享意图分类器
_router_instance = None           # 共享意图路由器
_tool_selector_instance = None    # 共享工具选择器

# ==================== 用户级隔离组件 ====================

_knowledge_base_instances = {}    # {user_id: KnowledgeBase实例}
_user_mcp_tools = {}              # {user_id: MCP工具列表}
_agent_instances = {}             # {user_id: Agent实例} - 因工具列表不同需隔离


def clear_all_cache():
    """清除所有缓存"""
    global _llm_instance, _all_tools, _classifier_instance, _router_instance, _tool_selector_instance
    global _knowledge_base_instances, _user_mcp_tools, _agent_instances
    
    _llm_instance = None
    _all_tools = None
    _classifier_instance = None
    _router_instance = None
    _tool_selector_instance = None
    _knowledge_base_instances.clear()
    _user_mcp_tools.clear()
    _agent_instances.clear()
    print("✓ 已清除所有缓存")


def clear_user_cache(user_id: str):
    """
    清除指定用户的缓存
    
    Args:
        user_id: 用户ID
    """
    global _knowledge_base_instances, _user_mcp_tools, _agent_instances
    
    _knowledge_base_instances.pop(user_id, None)
    _user_mcp_tools.pop(user_id, None)
    _agent_instances.pop(user_id, None)
    print(f"✓ 已清除用户 {user_id} 的缓存")


# ==================== 共享 LLM 实例 ====================

def get_llm(temperature: Optional[float] = None) -> ChatOpenAI:
    """
    获取共享 LLM 实例（全局单例）
    
    Args:
        temperature: 温度参数（仅首次创建时生效）
        
    Returns:
        LLM 实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            model_kwargs={"seed": LLM_SEED}
        )
        print("✓ 创建共享 LLM 实例")
    
    return _llm_instance


# ==================== 共享 VL 工具实例 ====================

def get_vl_tools():
    """获取 Qwen2.5-VL 工具集实例（单例模式）"""
    global _vl_tools_instance
    if _vl_tools_instance is None:
        _vl_tools_instance = Qwen25VLTools()
    return _vl_tools_instance


# ==================== 用户级知识库 ====================

def get_knowledge_base(user_id: Optional[str] = None):
    """
    获取知识库实例（用户级隔离）
    
    Args:
        user_id: 用户ID，用于实现多用户知识库隔离
        
    Returns:
        对应用户的知识库实例
    """
    global _knowledge_base_instances
    
    cache_key = user_id or "public"
    
    if cache_key not in _knowledge_base_instances:
        try:
            vl_tools = get_vl_tools()
            _knowledge_base_instances[cache_key] = KnowledgeBase(
                user_id=user_id,
                vl_tools=vl_tools
            )
            user_label = f"用户 {user_id}" if user_id else "公共"
            print(f"✓ {user_label}知识库初始化成功")
        except Exception as e:
            print(f"✗ 知识库初始化失败: {str(e)}")
            return None
    
    return _knowledge_base_instances[cache_key]


# ==================== 全部工具加载 ====================

def get_all_tools(user_id: Optional[str] = None, db_session=None) -> List:
    """    
    获取全部工具列表
    
    基础工具共享，用户 MCP 工具单独加载
    
    Args:
        user_id: 用户ID
        db_session: 数据库会话
        
    Returns:
        工具列表
    """
    global _all_tools, _user_mcp_tools
    
    # 1. 加载基础工具（共享）
    if _all_tools is None:
        vl_tools = get_vl_tools()
        
        from ..tools.base import get_all_tools as load_registered_tools
        _all_tools = load_registered_tools(vl_tools=vl_tools)
        print(f"✓ 从注册表加载了 {len(_all_tools)} 个基础工具")
    
    tools = list(_all_tools)  # 复制列表避免污染共享数据
    
    # 2. 添加用户 RAG 工具
    try:
        kb = get_knowledge_base(user_id)
        if kb and kb.vector_store.get_collection_count() > 0:
            rag_tool = create_rag_search_tool(kb)
            tools.append(rag_tool)
    except Exception as e:
        print(f"RAG 工具初始化失败: {str(e)}")
    
    # 3. 加载用户 MCP 工具（用户级隔离）
    cache_key = user_id or "public"
    if cache_key not in _user_mcp_tools:
        try:
            from ..tools import create_tools_from_config
            
            if user_id and db_session:
                from ..services.user_mcp_service import get_merged_mcp_config
                user_servers = get_merged_mcp_config(db_session, int(user_id))
                mcp_tools, _ = create_tools_from_config(servers=user_servers)
            else:
                mcp_tools, _ = create_tools_from_config()
            
            _user_mcp_tools[cache_key] = mcp_tools or []
            if mcp_tools:
                print(f"✓ 加载 {len(mcp_tools)} 个 MCP 工具")
        except Exception as e:
            print(f"MCP 工具加载失败: {str(e)}")
            _user_mcp_tools[cache_key] = []
    
    tools.extend(_user_mcp_tools[cache_key])
    
    # 4. 添加 FINISH 工具
    if not any(getattr(t, 'name', '') == 'FINISH' for t in tools):
        tools.append(finish_tool)
    
    return tools


# ==================== 共享工具选择器 ====================

def get_tool_selector():
    """
    获取共享工具选择器（全局单例）
    
    Returns:
        ToolSelector 实例
    """
    global _tool_selector_instance
    
    if _tool_selector_instance is None:
        from ..core.tools import ToolSelector, ToolMatcher
        
        llm = get_llm()
        matcher = ToolMatcher(llm=llm)
        _tool_selector_instance = ToolSelector(matcher=matcher)
        print("✓ 创建共享工具选择器")
    
    return _tool_selector_instance


# ==================== 共享意图分类器 ====================

def get_intent_classifier(use_llm: bool = True):
    """
    获取共享意图分类器（全局单例）
    
    Args:
        use_llm: 是否使用 LLM 分类
        
    Returns:
        IntentClassifier 实例
    """
    global _classifier_instance
    
    if _classifier_instance is None:
        from ..core.intent import IntentClassifier
        from ..core.intent.strategies import RuleClassifierStrategy, LLMClassifierStrategy
        
        llm = get_llm() if use_llm else None
        
        strategies = [RuleClassifierStrategy()]
        if use_llm and llm:
            strategies.append(LLMClassifierStrategy(llm=llm))
        
        _classifier_instance = IntentClassifier(
            strategies=strategies,
            llm=llm,
        )
        print("✓ 创建共享意图分类器")
    
    return _classifier_instance


# ==================== 共享意图路由器 ====================

def get_intent_router(user_id: Optional[str] = None, db_session=None):
    """
    获取共享意图路由器（全局单例）
    
    注意：Handler 内部可能需要用户级依赖，通过 intent.selected_tools 传递
    
    Args:
        user_id: 用户ID（用于获取用户级依赖）
        db_session: 数据库会话
        
    Returns:
        IntentRouter 实例
    """
    global _router_instance
    
    if _router_instance is None:
        from ..core.router import IntentRouter, HandlerFactory
        
        # 使用共享依赖创建工厂
        llm = get_llm()
        
        # Handler 工厂使用共享组件
        factory = HandlerFactory(
            llm=llm,
            tools=[],  # 工具通过 intent.selected_tools 动态传入
            knowledge_base=None,  # 通过 intent.metadata 传入
            agent=None,  # 延迟创建
        )
        
        _router_instance = IntentRouter(factory=factory)
        print("✓ 创建共享意图路由器")
    
    return _router_instance


# ==================== 用户级 Agent 实例 ====================

def get_agent(
    user_id: Optional[str] = None,
    temperature: Optional[float] = None,
    db_session=None,
    tools: Optional[List] = None
):
    """
    获取 Agent 实例（用户级，因工具列表可能不同）
    
    Args:
        user_id: 用户ID
        temperature: LLM 温度参数
        db_session: 数据库会话
        tools: 可选的精选工具列表
    
    Returns:
        Agent 实例
    """
    global _agent_instances
    
    cache_key = user_id or "public"
    
    if cache_key not in _agent_instances:
        llm = get_llm(temperature)
        all_tools = tools or get_all_tools(user_id, db_session)
        knowledge_base = get_knowledge_base(user_id)
        
        _agent_instances[cache_key] = Agent(
            llm=llm,
            tools=all_tools,
            knowledge_base=knowledge_base,
            use_rag=True
        )
        print(f"✓ 创建 Agent 实例: {cache_key}")
    
    return _agent_instances[cache_key]


# ==================== 创建带精选工具的临时 Agent ====================

def create_agent_with_tools(
    tools: List,
    user_id: Optional[str] = None,
    temperature: Optional[float] = None
) -> Agent:
    """
    创建带有精选工具的临时 Agent（不缓存）
    
    用于处理需要特定工具集的任务
    
    Args:
        tools: 精选工具列表
        user_id: 用户ID
        temperature: 温度参数
        
    Returns:
        Agent 实例
    """
    llm = get_llm(temperature)
    knowledge_base = get_knowledge_base(user_id)
    
    return Agent(
        llm=llm,
        tools=tools,
        knowledge_base=knowledge_base,
        use_rag=True
    )


# ==================== 兼容性函数（保留旧接口） ====================

def create_tools(user_id: Optional[str] = None, db_session=None):
    """兼容旧接口，等同于 get_all_tools"""
    return get_all_tools(user_id, db_session)


def clear_agent_cache(user_id: Optional[str] = None):
    """兼容旧接口"""
    if user_id is None:
        clear_all_cache()
    else:
        clear_user_cache(user_id)


def get_handler_factory(
    user_id: Optional[str] = None,
    temperature: Optional[float] = None,
    db_session=None
):
    """
    获取 Handler 工厂（兼容旧接口）
    
    现在使用共享组件，工具通过 intent 动态传入
    """
    from ..core.router import HandlerFactory
    
    llm = get_llm(temperature)
    tools = get_all_tools(user_id, db_session)
    knowledge_base = get_knowledge_base(user_id)
    agent = get_agent(user_id, temperature, db_session)
    
    return HandlerFactory(
        llm=llm,
        tools=tools,
        knowledge_base=knowledge_base,
        agent=agent,
    )
