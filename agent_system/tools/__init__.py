"""工具模块"""
from .train_tools import search_train_ticket_tool, purchase_train_ticket_tool
from .common_tools import finish_tool
from .mcp_file_tools import (
    create_filesystem_tools,
    create_mcp_langchain_tools,
    create_sqlite_tools,
    create_tools_from_config,
    create_tools_by_server_name,
    close_all_adapters,
    MCPClientAdapter
)
from .qwen_vl_tools import Qwen25VLTools
from .rag_tools import create_rag_search_tool

# 模块化工具注册系统
from .registry import (
    BaseTool,
    ToolRegistry,
    register_tool,
    get_all_tools,
    get_tool_by_name,
    get_tools_by_tags,
    list_registered_tools
)

__all__ = [
    # 火车票工具
    "search_train_ticket_tool",
    "purchase_train_ticket_tool", 
    # 通用工具
    "finish_tool",
    # MCP 工具（直接创建）
    "create_filesystem_tools",
    "create_mcp_langchain_tools",
    "create_sqlite_tools",
    # MCP 工具（配置文件驱动）
    "create_tools_from_config",
    "create_tools_by_server_name",
    "close_all_adapters",
    "MCPClientAdapter",
    # 视觉工具
    "Qwen25VLTools",
    # RAG 工具
    "create_rag_search_tool",
    # 工具注册系统
    "BaseTool",
    "ToolRegistry",
    "register_tool",
    "get_all_tools",
    "get_tool_by_name",
    "get_tools_by_tags",
    "list_registered_tools"
]


