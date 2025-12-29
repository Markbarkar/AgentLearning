"""
API 共享依赖

包含全局变量、缓存实例和工具创建函数
"""

from typing import Optional
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


# ==================== 全局变量：缓存实例 ====================

_agent_instances = {}  # {user_id: Agent实例}
_vl_tools_instance = None
_knowledge_base_instances = {}  # {user_id: KnowledgeBase实例}，支持多用户隔离


# ==================== 工具实例获取 ====================

def get_vl_tools():
    """获取 Qwen2.5-VL 工具集实例（单例模式）"""
    global _vl_tools_instance
    if _vl_tools_instance is None:
        _vl_tools_instance = Qwen25VLTools()
    return _vl_tools_instance


# ==================== 知识库实例获取 ====================

def get_knowledge_base(user_id: Optional[str] = None):
    """
    获取知识库实例（多用户模式）
    
    Args:
        user_id: 用户ID，用于实现多用户知识库隔离
        
    Returns:
        对应用户的知识库实例
    """
    global _knowledge_base_instances
    
    # 使用 user_id 或 "public" 作为缓存key
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


# ==================== 工具创建函数 ====================

def create_qwen_vl_tools(user_id: Optional[str] = None):
    """
    创建 Qwen2.5-VL 相关的 LangChain 工具
    
    使用模块化注册机制自动加载工具
    
    Args:
        user_id: 用户ID，用于创建用户专属的RAG工具
    """
    vl_tools = get_vl_tools()
    
    # 从注册表动态加载所有工具
    from ..tools.registry import get_all_tools
    tools = get_all_tools(vl_tools=vl_tools)
    print(f"✓ 从注册表加载了 {len(tools)} 个工具")
    
    # 尝试添加 RAG 工具（使用用户专属知识库）
    try:
        kb = get_knowledge_base(user_id)
        if kb and kb.vector_store.get_collection_count() > 0:
            rag_tool = create_rag_search_tool(kb)
            tools.append(rag_tool)
            user_label = f"用户 {user_id}" if user_id else "公共"
            print(f"✓ {user_label} RAG 工具已添加")
    except Exception as e:
        print(f"RAG 工具初始化失败: {str(e)}")
    
    # 6. MCP 工具（从配置文件加载）
    # 注意：MCP 工具需要手动管理连接生命周期
    # 如需启用，取消下面的注释
    try:
        from ..tools import create_tools_from_config
        mcp_tools, mcp_adapters = create_tools_from_config()
        if mcp_tools:
            tools.extend(mcp_tools)
            print(f"✓ 已加载 {len(mcp_tools)} 个 MCP 工具")
            # 注意：需要在应用退出时调用 close_all_adapters(mcp_adapters)
    except Exception as e:
        print(f"MCP 工具加载失败（可忽略）: {str(e)}")
    
    tools.append(finish_tool)
    return tools


# ==================== Agent 实例获取 ====================

def get_agent(user_id: Optional[str] = None, temperature: Optional[float] = None):
    """
    获取 Agent 实例（多用户模式）
    
    Agent实例始终带有RAG能力，是否使用RAG在run()时通过参数动态决定。
    这样可以避免频繁切换RAG状态时重建Agent实例。
    
    Args:
        user_id: 用户ID，用于实现多用户知识库隔离
        temperature: LLM 温度参数，如果为 None 则使用默认配置
    
    Returns:
        对应用户的 Agent 实例
    """
    global _agent_instances
    
    # 使用 user_id 或 "public" 作为缓存key
    cache_key = user_id or "public"
    
    # 只有在缓存不存在时才创建新实例
    if cache_key not in _agent_instances:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            model_kwargs={"seed": LLM_SEED}
        )
        
        # 创建用户专属工具（包括用户专属RAG工具）
        tools = create_qwen_vl_tools(user_id)
        
        # 始终获取用户专属知识库（运行时决定是否使用）
        knowledge_base = None
        try:
            knowledge_base = get_knowledge_base(user_id)
        except Exception as e:
            print(f"知识库加载失败: {str(e)}")
        
        _agent_instances[cache_key] = Agent(
            llm=llm,
            tools=tools,
            knowledge_base=knowledge_base,
            use_rag=True  # 默认启用，实际使用由run()参数控制
        )
        print(f"✓ 创建Agent实例: {cache_key}")
    
    return _agent_instances[cache_key]

