"""
工具注册模块

自动发现并加载 registry 目录下所有工具模块
"""

import os
import importlib
from pathlib import Path
from typing import List, Optional

from langchain_core.tools import Tool

from .base import (
    BaseTool,
    ToolRegistry,
    register_tool,
    create_tools_from_registry,
    get_tool_by_name,
    get_tools_by_tags
)


def _discover_and_load_tools():
    """
    自动发现并加载当前目录下的所有工具模块
    
    扫描 registry 目录下的所有 .py 文件（排除 __init__.py 和 base.py），
    并导入它们以触发 @register_tool 装饰器
    """
    current_dir = Path(__file__).parent
    
    for file_path in current_dir.glob("*.py"):
        module_name = file_path.stem
        
        # 跳过特殊文件
        if module_name.startswith("_") or module_name == "base":
            continue
        
        try:
            # 动态导入模块
            importlib.import_module(f".{module_name}", package=__name__)
        except Exception as e:
            print(f"⚠️ 加载工具模块 '{module_name}' 失败: {e}")


def get_all_tools(vl_tools=None) -> List[Tool]:
    """
    获取所有已注册的工具
    
    自动发现并加载工具模块，然后返回所有工具的 LangChain Tool 实例
    
    Args:
        vl_tools: Qwen VL 工具实例（可选）
        
    Returns:
        LangChain Tool 列表
    """
    # 确保工具已加载
    _discover_and_load_tools()
    
    # 创建工具实例
    return create_tools_from_registry(vl_tools)


def list_registered_tools() -> List[dict]:
    """
    列出所有已注册的工具信息
    
    Returns:
        工具信息列表，每个元素包含 name, description, tags
    """
    # 确保工具已加载
    _discover_and_load_tools()
    
    tools_info = []
    for name, tool_class in ToolRegistry.get_all().items():
        tools_info.append({
            "name": tool_class.name,
            "description": tool_class.description[:100] + "..." if len(tool_class.description) > 100 else tool_class.description,
            "tags": tool_class.tags,
            "class": tool_class.__name__
        })
    
    return tools_info


# 模块加载时自动发现工具
_discover_and_load_tools()


__all__ = [
    # 基类和装饰器
    "BaseTool",
    "ToolRegistry", 
    "register_tool",
    # 工具获取函数
    "get_all_tools",
    "get_tool_by_name",
    "get_tools_by_tags",
    "list_registered_tools",
    # 底层函数
    "create_tools_from_registry",
]

