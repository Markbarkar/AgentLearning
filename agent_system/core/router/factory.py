"""
Handler 工厂

负责创建 Handler 实例并自动注入依赖
采用依赖注入模式，根据 Handler 构造函数签名自动匹配依赖
"""

import inspect
from typing import Dict, Any, Type, Optional, List

from .handlers.base import BaseHandler


class HandlerFactory:
    """
    Handler 工厂
    
    管理依赖项并根据 Handler 的构造函数签名自动注入
    
    使用方式:
        >>> factory = HandlerFactory(
        ...     llm=llm,
        ...     knowledge_base=kb,
        ...     tools=tools,
        ...     agent=agent
        ... )
        >>> handler = factory.create(ChatHandler)
    """
    
    def __init__(self, **dependencies):
        """
        初始化工厂
        
        Args:
            **dependencies: 依赖项，key 为依赖名称，value 为依赖实例
                常用依赖：
                - llm: 语言模型实例
                - knowledge_base: 知识库实例
                - tools: 工具列表
                - agent: ReAct Agent 实例
        """
        self._dependencies: Dict[str, Any] = dependencies
    
    def register_dependency(self, name: str, instance: Any) -> None:
        """
        注册新的依赖
        
        Args:
            name: 依赖名称
            instance: 依赖实例
        """
        self._dependencies[name] = instance
    
    def unregister_dependency(self, name: str) -> bool:
        """
        注销依赖
        
        Args:
            name: 依赖名称
            
        Returns:
            是否成功注销
        """
        if name in self._dependencies:
            del self._dependencies[name]
            return True
        return False
    
    def get_dependency(self, name: str) -> Optional[Any]:
        """
        获取依赖实例
        
        Args:
            name: 依赖名称
            
        Returns:
            依赖实例，不存在返回 None
        """
        return self._dependencies.get(name)
    
    def get_all_dependencies(self) -> Dict[str, Any]:
        """获取所有已注册的依赖"""
        return self._dependencies.copy()
    
    def create(self, handler_cls: Type[BaseHandler]) -> BaseHandler:
        """
        创建 Handler 实例
        
        根据 Handler 类的 __init__ 方法签名自动注入依赖
        
        Args:
            handler_cls: Handler 类
            
        Returns:
            Handler 实例
            
        Raises:
            TypeError: 如果缺少必要的依赖
        """
        # 获取构造函数签名
        sig = inspect.signature(handler_cls.__init__)
        
        # 收集需要注入的参数
        kwargs = {}
        missing_deps = []
        
        for param_name, param in sig.parameters.items():
            # 跳过 self
            if param_name == 'self':
                continue
            
            # 检查是否有对应的依赖
            if param_name in self._dependencies:
                kwargs[param_name] = self._dependencies[param_name]
            elif param.default == inspect.Parameter.empty:
                # 没有默认值且未提供依赖，记录缺失
                missing_deps.append(param_name)
            # 有默认值的参数不需要强制提供
        
        # 检查是否有缺失的必要依赖
        if missing_deps:
            raise TypeError(
                f"创建 {handler_cls.__name__} 失败，"
                f"缺少必要依赖: {', '.join(missing_deps)}。"
                f"可用依赖: {list(self._dependencies.keys())}"
            )
        
        return handler_cls(**kwargs)
    
    def create_all(self, handler_classes: List[Type[BaseHandler]]) -> Dict[str, BaseHandler]:
        """
        批量创建 Handler 实例
        
        Args:
            handler_classes: Handler 类列表
            
        Returns:
            {意图类型: Handler实例} 字典
        """
        handlers = {}
        for cls in handler_classes:
            intent_type = getattr(cls, '_intent_type', cls.__name__)
            try:
                handlers[intent_type] = self.create(cls)
            except TypeError as e:
                print(f"⚠️ 创建 Handler 失败: {e}")
        return handlers
    
    def can_create(self, handler_cls: Type[BaseHandler]) -> bool:
        """
        检查是否可以创建指定的 Handler
        
        Args:
            handler_cls: Handler 类
            
        Returns:
            是否可以创建（所有必要依赖都已注册）
        """
        sig = inspect.signature(handler_cls.__init__)
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            # 必要参数（没有默认值）必须有对应的依赖
            if param.default == inspect.Parameter.empty:
                if param_name not in self._dependencies:
                    return False
        
        return True
    
    def get_missing_dependencies(self, handler_cls: Type[BaseHandler]) -> List[str]:
        """
        获取创建 Handler 所缺少的依赖
        
        Args:
            handler_cls: Handler 类
            
        Returns:
            缺少的依赖名称列表
        """
        sig = inspect.signature(handler_cls.__init__)
        missing = []
        
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            
            if param.default == inspect.Parameter.empty:
                if param_name not in self._dependencies:
                    missing.append(param_name)
        
        return missing
    
    def update_dependencies(self, **new_deps) -> None:
        """
        批量更新依赖
        
        Args:
            **new_deps: 新的依赖项
        """
        self._dependencies.update(new_deps)
    
    def __repr__(self) -> str:
        deps = list(self._dependencies.keys())
        return f"HandlerFactory(dependencies={deps})"
