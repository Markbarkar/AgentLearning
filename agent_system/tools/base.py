"""
工具注册基础设施

提供方法级装饰器，直接在工具方法上注册为 LangChain Tool
支持多个工具类实例
"""

from typing import List, Dict, Any, Optional, Callable, Type
from langchain_core.tools import Tool


class ToolRegistry:
    """
    工具注册表（单例模式）
    
    存储所有已注册的工具方法元信息和工具实例
    """
    # 方法注册表: {tool_name: {method, description, input_parser, class_name}}
    _methods: Dict[str, Dict[str, Any]] = {}
    # 实例注册表: {class_name: instance}
    _instances: Dict[str, Any] = {}
    
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
        # 获取方法所属的类名（用于后续实例绑定）
        class_name = method.__qualname__.rsplit('.', 1)[0] if '.' in method.__qualname__ else None
        
        cls._methods[name] = {
            "method": method,
            "method_name": method.__name__,
            "description": description,
            "input_parser": input_parser,
            "class_name": class_name
        }
    
    @classmethod
    def register_instance(cls, instance: Any, class_name: Optional[str] = None):
        """
        注册工具类实例
        
        Args:
            instance: 工具类实例
            class_name: 类名（可选，默认使用 instance.__class__.__name__）
        """
        key = class_name or instance.__class__.__name__
        cls._instances[key] = instance
    
    @classmethod
    def get_instance(cls, class_name: str) -> Optional[Any]:
        """根据类名获取实例"""
        return cls._instances.get(class_name)
    
    @classmethod
    def get_all_instances(cls) -> Dict[str, Any]:
        """获取所有已注册的实例"""
        return cls._instances.copy()
    
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
        cls._instances = {}


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
        
        # 实例化时自动注册
        my_tools = MyTools()
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
        func._tool_input_parser = input_parser
        return func
    return decorator


def auto_register(cls: Type) -> Type:
    """
    类装饰器：自动注册实例
    
    装饰在工具类上，使得实例化时自动注册到 ToolRegistry
    
    用法:
        @auto_register
        class MyTools:
            @register_tool(name="my_tool", description="...")
            def my_method(self):
                ...
        
        tools = MyTools()  # 自动注册
    """
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # 自动注册实例
        ToolRegistry.register_instance(self)
    
    cls.__init__ = new_init
    return cls


def _default_input_parser(input_str: str) -> Dict[str, Any]:
    """
    默认输入解析器
    
    尝试智能解析输入：
    1. 如果是 JSON，解析为字典
    2. 如果包含逗号，按逗号分隔
    3. 否则作为单个参数
    """
    input_str = input_str.strip()
    
    # 尝试 JSON 解析
    if input_str.startswith('{'):
        try:
            import json
            return json.loads(input_str)
        except json.JSONDecodeError:
            pass
    
    # 作为文件路径或单个参数
    return {"input_str": input_str}


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
    支持多个工具类实例：
    1. 优先从 ToolRegistry._instances 查找实例
    2. 如果找不到，尝试从 vl_tools 参数获取（向后兼容）
    
    Args:
        vl_tools: Qwen25VLTools 实例（可选，向后兼容）
        
    Returns:
        LangChain Tool 列表
    """
    tools = []
    
    # 如果传入了 vl_tools，也注册它
    if vl_tools is not None:
        ToolRegistry.register_instance(vl_tools)
    
    for name, info in ToolRegistry.get_all().items():
        method = info["method"]
        method_name = info["method_name"]
        description = info["description"]
        input_parser = info.get("input_parser") or _default_input_parser
        class_name = info.get("class_name")
        
        # 查找实例
        bound_method = None
        instance = None
        
        # 1. 从注册的实例中查找
        if class_name:
            instance = ToolRegistry.get_instance(class_name)
            if instance and hasattr(instance, method_name):
                bound_method = getattr(instance, method_name)
        
        # 2. 如果还没找到，尝试遍历所有实例
        if bound_method is None:
            for inst_name, inst in ToolRegistry.get_all_instances().items():
                if hasattr(inst, method_name):
                    bound_method = getattr(inst, method_name)
                    instance = inst
                    break
        
        if bound_method is None:
            print(f"⚠️ 工具 '{name}' (方法: {method_name}, 类: {class_name}) 未找到绑定实例，跳过")
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
                except TypeError as e:
                    # 参数不匹配，尝试直接传入字符串
                    try:
                        result = m(input_str)
                        return str(result)
                    except Exception as inner_e:
                        return f"错误（参数解析）: {str(e)}"
                except Exception as e:
                    return f"错误: {str(e)}"
            return wrapper
        
        tools.append(Tool(
            name=name,
            description=description,
            func=make_wrapper(bound_method, input_parser, name)
        ))
        print(f"  ✓ 已注册工具: {name} (实例: {instance.__class__.__name__})")
    
    return tools


# 导出常用的输入解析器
__all__ = [
    "ToolRegistry",
    "register_tool",
    "auto_register",
    "get_all_tools",
    "_default_input_parser",
    "_default_file_path_parser",
    "_parse_annotate_input",
    "_parse_form_input",
]
