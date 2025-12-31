"""
工具模块

使用 @register_tool 装饰器自动注册工具
"""

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
from .rag_tools import create_rag_search_tool

# 工具注册系统（新的方法级装饰器）
from .base import (
    ToolRegistry,
    register_tool,
    get_all_tools,
)

# 导入 Qwen25VLTools 以触发 @register_tool 装饰器注册
from .qwen_vl_tools import Qwen25VLTools

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
    "ToolRegistry",
    "register_tool",
    "get_all_tools",
]
