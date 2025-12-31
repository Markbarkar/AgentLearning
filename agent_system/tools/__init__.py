"""
工具模块

使用 @register_tool 装饰器自动注册工具
"""

from .common_tools import finish_tool
from .rag_tools import create_rag_search_tool

# 工具注册系统（新的方法级装饰器）
from .base import (
    ToolRegistry,
    register_tool,
    get_all_tools,
)

# 导入工具模块以触发 @register_tool 装饰器注册
from .qwen_vl_tools import Qwen25VLTools
from .bash_tool import BashToolExecutor, get_allowed_commands, get_allowed_directories
from .project_tools import ProjectTools

# 工具执行上下文（用于传递 token 等请求级信息）
from .context import (
    get_current_token,
    get_current_user_id,
    get_current_context,
    set_context,
    clear_context,
    ContextManager,
    ToolContext,
)

# MCP 工具（可选，需要安装 mcp 包）
try:
    from .mcp_file_tools import (
        create_filesystem_tools,
        create_mcp_langchain_tools,
        create_sqlite_tools,
        create_tools_from_config,
        create_tools_by_server_name,
        close_all_adapters,
        MCPClientAdapter
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    create_filesystem_tools = None
    create_mcp_langchain_tools = None
    create_sqlite_tools = None
    create_tools_from_config = None
    create_tools_by_server_name = None
    close_all_adapters = None
    MCPClientAdapter = None

__all__ = [
    # 通用工具
    "finish_tool",
    # MCP 工具（需要 mcp 包）
    "MCP_AVAILABLE",
    "create_filesystem_tools",
    "create_mcp_langchain_tools",
    "create_sqlite_tools",
    "create_tools_from_config",
    "create_tools_by_server_name",
    "close_all_adapters",
    "MCPClientAdapter",
    # 视觉工具
    "Qwen25VLTools",
    # RAG 工具
    "create_rag_search_tool",
    # 工具注册系统
    "ToolRegistry",
    "register_tool",
    "get_all_tools",
    # Bash 工具
    "BashToolExecutor",
    "get_allowed_commands",
    "get_allowed_directories",
    # Project 工具
    "ProjectTools",
]
