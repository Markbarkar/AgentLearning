"""
工具注册基础设施

提供方法级装饰器，直接在工具方法上注册为 LangChain Tool
"""

from typing import List, Dict, Any, Optional, Callable
from langchain_core.tools import Tool


class ToolRegistry:
    """
    工具注册表（单例模式）
    
    存储所有已注册的工具方法元信息
    """
    _methods: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register_method(
        cls, 
        name: str, 
        description: str, 
        method: Callable,
        input_parser: Optional[Callable] = None
    ):
        """
        注册工具方法
        
        Args:
            name: 工具名称（唯一标识）
            description: 工具描述（供 LLM 理解如何使用）
            method: 原始方法
            input_parser: 输入解析函数（可选）
        """
        cls._methods[name] = {
            "method": method,
            "description": description,
            "input_parser": input_parser
        }
    
    @classmethod
    def get_all(cls) -> Dict[str, Dict[str, Any]]:
        """获取所有已注册的工具方法"""
        return cls._methods.copy()
    
    @classmethod
    def get(cls, name: str) -> Optional[Dict[str, Any]]:
        """根据名称获取工具方法信息"""
        return cls._methods.get(name)
    
    @classmethod
    def clear(cls):
        """清空注册表（用于测试）"""
        cls._methods = {}


def register_tool(
    name: str, 
    description: str,
    input_parser: Optional[Callable] = None
):
    """
    方法级工具注册装饰器
    
    直接在类方法上使用，自动注册为可供 Agent 调用的工具
    
    Args:
        name: 工具名称（唯一标识）
        description: 工具描述（详细说明用途、输入格式、使用场景）
        input_parser: 自定义输入解析函数（可选）
    
    用法:
        class MyTools:
            @register_tool(
                name="extract_info",
                description="提取文档信息，输入：文件路径"
            )
            def extract_info(self, file_path=None):
                ...
    """
    def decorator(func: Callable) -> Callable:
        # 注册方法元信息（延迟绑定实例）
        ToolRegistry.register_method(
            name=name,
            description=description,
            method=func,
            input_parser=input_parser
        )
        # 在方法上添加元数据标记
        func._tool_name = name
        func._tool_description = description
        return func
    return decorator


def _default_file_path_parser(input_str: str) -> Dict[str, Any]:
    """默认的文件路径解析器"""
    return {"file_path": input_str.strip()}


def _parse_annotate_input(input_str: str) -> Dict[str, Any]:
    """标注工具的输入解析器：file_path,key_fields"""
    parts = input_str.split(',', 1)
    if len(parts) == 2:
        return {
            "file_path": parts[0].strip(),
            "key_fields": parts[1].strip()
        }
    return {"file_path": input_str.strip()}


def _parse_form_input(input_str: str) -> Dict[str, Any]:
    """表单识别工具的输入解析器：file_path,table_type"""
    parts = input_str.split(',', 1)
    if len(parts) == 2:
        return {
            "file_path": parts[0].strip(),
            "table_type": parts[1].strip()
        }
    return {"file_path": input_str.strip()}


def get_all_tools(vl_tools=None) -> List[Tool]:
    """
    获取所有已注册的工具
    
    将注册的方法转换为 LangChain Tool 对象
    
    Args:
        vl_tools: Qwen25VLTools 实例（用于绑定方法）
        
    Returns:
        LangChain Tool 列表
    """
    tools = []
    
    for name, info in ToolRegistry.get_all().items():
        method = info["method"]
        description = info["description"]
        input_parser = info.get("input_parser") or _default_file_path_parser
        
        # 尝试从 vl_tools 实例获取绑定方法
        bound_method = None
        if vl_tools and hasattr(vl_tools, method.__name__):
            bound_method = getattr(vl_tools, method.__name__)
        
        if bound_method is None:
            print(f"⚠️ 工具 '{name}' 未找到绑定实例，跳过")
            continue
        
        # 创建包装函数
        def make_wrapper(m, parser, tool_name):
            def wrapper(input_str: str) -> str:
                try:
                    # 解析输入
                    kwargs = parser(input_str)
                    # 调用实际方法
                    result = m(**kwargs)
                    return str(result)
                except Exception as e:
                    return f"错误: {str(e)}"
            return wrapper
        
        tools.append(Tool(
            name=name,
            description=description,
            func=make_wrapper(bound_method, input_parser, name)
        ))
        print(f"  ✓ 已注册工具: {name}")
    
    return tools


# 导出常用的输入解析器
__all__ = [
    "ToolRegistry",
    "register_tool",
    "get_all_tools",
    "_default_file_path_parser",
    "_parse_annotate_input",
    "_parse_form_input",
]

