"""
Handler 抽象基类

定义所有意图处理器的接口规范
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ...intent.models import Intent


class BaseHandler(ABC):
    """
    意图处理器抽象基类
    
    所有具体的 Handler 都需要继承此类并实现 handle 方法
    
    设计原则：
    1. 依赖通过构造函数注入（由 HandlerFactory 管理）
    2. handle 方法是唯一必须实现的接口
    3. 可选实现 can_handle 方法进行前置检查
    """
    
    # 类级别元数据（由装饰器设置）
    _intent_type: str = ""
    _priority: int = 0
    
    @abstractmethod
    def handle(self, intent: Intent) -> str:
        """
        处理意图
        
        Args:
            intent: 意图识别结果，包含 type、query、confidence、metadata
            
        Returns:
            处理结果字符串
            
        Raises:
            可以抛出异常，由路由器统一处理
        """
        pass
    
    def can_handle(self, intent: Intent) -> bool:
        """
        检查是否可以处理该意图
        
        子类可重写此方法进行前置检查
        例如：检查必要的依赖是否存在
        
        Args:
            intent: 意图识别结果
            
        Returns:
            是否可以处理
        """
        return True
    
    def get_dependencies(self) -> Dict[str, Any]:
        """
        获取 Handler 的依赖信息
        
        用于调试和文档生成
        
        Returns:
            {依赖名称: 依赖值} 字典
        """
        # 返回实例的所有非私有属性
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    @classmethod
    def get_intent_type(cls) -> str:
        """获取处理的意图类型"""
        return cls._intent_type
    
    @classmethod
    def get_priority(cls) -> int:
        """获取优先级"""
        return cls._priority
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(intent_type={self._intent_type})"
