"""
工具注册基础设施

提供工具定义的基类和自动注册机制
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Type
from langchain_core.tools import Tool


class ToolRegistry:
    """
    工具注册表（单例模式）
    
    管理所有已注册的工具类
    """
    _instance = None
    _tools: Dict[str, Type["BaseTool"]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._tools = {}
        return cls._instance
    
    @classmethod
    def register(cls, tool_class: Type["BaseTool"]) -> Type["BaseTool"]:
        """注册工具类"""
        if cls._instance is None:
            cls()
        cls._tools[tool_class.name] = tool_class
        return tool_class
    
    @classmethod
    def get_all(cls) -> Dict[str, Type["BaseTool"]]:
        """获取所有已注册的工具类"""
        if cls._instance is None:
            cls()
        return cls._tools.copy()
    
    @classmethod
    def get(cls, name: str) -> Optional[Type["BaseTool"]]:
        """根据名称获取工具类"""
        if cls._instance is None:
            cls()
        return cls._tools.get(name)
    
    @classmethod
    def clear(cls):
        """清空注册表（用于测试）"""
        cls._tools = {}


def register_tool(cls: Type["BaseTool"]) -> Type["BaseTool"]:
    """
    工具注册装饰器
    
    用法:
        @register_tool
        class MyTool(BaseTool):
            name = "my_tool"
            description = "..."
            
            def execute(self, input_str: str) -> str:
                return "result"
    """
    ToolRegistry.register(cls)
    return cls


class BaseTool(ABC):
    """
    工具基类
    
    所有自定义工具都应继承此类并实现 execute 方法
    
    属性:
        name: 工具名称（唯一标识）
        description: 工具描述（供 LLM 理解如何使用）
        tags: 工具标签（可选，用于分类筛选）
    """
    
    # 子类必须定义这些类属性
    name: str = ""
    description: str = ""
    tags: List[str] = []
    
    # 运行时注入的依赖
    _vl_tools = None
    
    @classmethod
    def set_vl_tools(cls, vl_tools):
        """设置 VL 工具实例（依赖注入）"""
        cls._vl_tools = vl_tools
    
    @property
    def vl_tools(self):
        """获取 VL 工具实例"""
        return self._vl_tools
    
    @abstractmethod
    def execute(self, input_str: str) -> str:
        """
        执行工具逻辑
        
        Args:
            input_str: 输入字符串（由 LLM 生成）
            
        Returns:
            执行结果字符串
        """
        pass
    
    def to_langchain_tool(self) -> Tool:
        """
        转换为 LangChain Tool 对象
        
        Returns:
            LangChain Tool 实例
        """
        return Tool(
            name=self.name,
            description=self.description,
            func=self.execute
        )
    
    def __repr__(self):
        return f"<{self.__class__.__name__} name='{self.name}'>"


def create_tools_from_registry(vl_tools=None) -> List[Tool]:
    """
    从注册表创建所有工具的 LangChain Tool 实例
    
    Args:
        vl_tools: Qwen VL 工具实例（可选）
        
    Returns:
        LangChain Tool 列表
    """
    # 注入依赖
    if vl_tools is not None:
        BaseTool.set_vl_tools(vl_tools)
    
    tools = []
    for name, tool_class in ToolRegistry.get_all().items():
        try:
            tool_instance = tool_class()
            lc_tool = tool_instance.to_langchain_tool()
            tools.append(lc_tool)
        except Exception as e:
            print(f"⚠️ 工具 '{name}' 创建失败: {e}")
    
    return tools


def get_tool_by_name(name: str, vl_tools=None) -> Optional[Tool]:
    """
    根据名称获取单个工具
    
    Args:
        name: 工具名称
        vl_tools: Qwen VL 工具实例（可选）
        
    Returns:
        LangChain Tool 实例，不存在则返回 None
    """
    tool_class = ToolRegistry.get(name)
    if tool_class is None:
        return None
    
    if vl_tools is not None:
        BaseTool.set_vl_tools(vl_tools)
    
    tool_instance = tool_class()
    return tool_instance.to_langchain_tool()


def get_tools_by_tags(tags: List[str], vl_tools=None) -> List[Tool]:
    """
    根据标签筛选工具
    
    Args:
        tags: 标签列表（返回包含任意标签的工具）
        vl_tools: Qwen VL 工具实例（可选）
        
    Returns:
        匹配的 LangChain Tool 列表
    """
    if vl_tools is not None:
        BaseTool.set_vl_tools(vl_tools)
    
    tools = []
    for name, tool_class in ToolRegistry.get_all().items():
        if any(tag in tool_class.tags for tag in tags):
            try:
                tool_instance = tool_class()
                tools.append(tool_instance.to_langchain_tool())
            except Exception as e:
                print(f"⚠️ 工具 '{name}' 创建失败: {e}")
    
    return tools
