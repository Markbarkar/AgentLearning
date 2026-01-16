"""
分类策略基类

定义策略接口和注册机制
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, Type, List

from ..models import Intent


class ClassifierStrategy(ABC):
    """
    分类策略抽象基类
    
    所有分类策略都需要继承此类并实现 classify 方法
    """
    
    # 策略优先级（数字越大优先级越高）
    priority: int = 0
    # 策略名称
    name: str = "base"
    
    @abstractmethod
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Intent]:
        """
        对查询进行分类
        
        Args:
            query: 用户查询文本
            context: 上下文信息（可选），包含 tools、history 等
            
        Returns:
            Intent 实例，如果无法分类则返回 None
        """
        pass
    
    def can_handle(self, query: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        判断此策略是否可以处理该查询
        
        子类可以重写此方法以提供快速判断
        默认实现直接调用 classify 并检查结果
        
        Args:
            query: 用户查询文本
            context: 上下文信息
            
        Returns:
            是否可以处理
        """
        return self.classify(query, context) is not None


class StrategyRegistry:
    """
    策略注册表
    
    管理所有已注册的分类策略
    """
    _strategies: Dict[str, Type[ClassifierStrategy]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        注册策略的装饰器
        
        Args:
            name: 策略名称
            
        Usage:
            @StrategyRegistry.register("rule")
            class RuleStrategy(ClassifierStrategy):
                ...
        """
        def decorator(strategy_cls: Type[ClassifierStrategy]):
            cls._strategies[name] = strategy_cls
            strategy_cls.name = name
            return strategy_cls
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[ClassifierStrategy]]:
        """根据名称获取策略类"""
        return cls._strategies.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, Type[ClassifierStrategy]]:
        """获取所有已注册的策略"""
        return cls._strategies.copy()
    
    @classmethod
    def get_sorted_by_priority(cls) -> List[Type[ClassifierStrategy]]:
        """按优先级排序获取所有策略类"""
        return sorted(
            cls._strategies.values(),
            key=lambda s: getattr(s, 'priority', 0),
            reverse=True
        )
    
    @classmethod
    def clear(cls):
        """清空注册表（用于测试）"""
        cls._strategies = {}


# 便捷装饰器
def register_strategy(name: str, priority: int = 0):
    """
    注册分类策略的装饰器
    
    Args:
        name: 策略名称
        priority: 优先级（数字越大优先级越高）
        
    Usage:
        @register_strategy("rule", priority=100)
        class RuleStrategy(ClassifierStrategy):
            ...
    """
    def decorator(strategy_cls: Type[ClassifierStrategy]):
        strategy_cls.priority = priority
        strategy_cls.name = name
        StrategyRegistry._strategies[name] = strategy_cls
        return strategy_cls
    return decorator
