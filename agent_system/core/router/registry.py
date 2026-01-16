"""
Handler 注册表

提供处理器的自动注册机制
使用装饰器模式，新增 Handler 无需修改路由器代码
"""

from typing import Dict, Optional, Type, List, Callable


class HandlerRegistry:
    """
    Handler 注册表
    
    管理所有已注册的意图处理器
    使用类级别字典存储，支持动态注册和获取
    
    使用方式:
        # 注册
        @HandlerRegistry.register("chat")
        class ChatHandler(BaseHandler):
            ...
        
        # 获取
        handler_cls = HandlerRegistry.get("chat")
    """
    
    # 注册表：{意图类型: Handler类}
    _handlers: Dict[str, Type] = {}
    
    # 元数据：{意图类型: {priority, description, ...}}
    _metadata: Dict[str, Dict] = {}
    
    @classmethod
    def register(
        cls, 
        intent_type: str,
        priority: int = 0,
        description: str = ""
    ) -> Callable:
        """
        注册 Handler 的装饰器
        
        Args:
            intent_type: 意图类型名称
            priority: 优先级（多个 Handler 匹配同一意图时使用）
            description: Handler 描述
            
        Returns:
            装饰器函数
            
        Usage:
            @HandlerRegistry.register("chat", priority=100)
            class ChatHandler(BaseHandler):
                ...
        """
        def decorator(handler_cls: Type) -> Type:
            cls._handlers[intent_type] = handler_cls
            cls._metadata[intent_type] = {
                "priority": priority,
                "description": description,
                "class_name": handler_cls.__name__,
            }
            # 在类上保存元数据
            handler_cls._intent_type = intent_type
            handler_cls._priority = priority
            return handler_cls
        return decorator
    
    @classmethod
    def get(cls, intent_type: str) -> Optional[Type]:
        """
        根据意图类型获取 Handler 类
        
        Args:
            intent_type: 意图类型名称
            
        Returns:
            Handler 类，未找到返回 None
        """
        return cls._handlers.get(intent_type)
    
    @classmethod
    def get_all(cls) -> Dict[str, Type]:
        """获取所有已注册的 Handler"""
        return cls._handlers.copy()
    
    @classmethod
    def get_metadata(cls, intent_type: str) -> Optional[Dict]:
        """获取 Handler 元数据"""
        return cls._metadata.get(intent_type)
    
    @classmethod
    def get_all_metadata(cls) -> Dict[str, Dict]:
        """获取所有 Handler 的元数据"""
        return cls._metadata.copy()
    
    @classmethod
    def get_registered_types(cls) -> List[str]:
        """获取所有已注册的意图类型"""
        return list(cls._handlers.keys())
    
    @classmethod
    def is_registered(cls, intent_type: str) -> bool:
        """检查意图类型是否已注册"""
        return intent_type in cls._handlers
    
    @classmethod
    def unregister(cls, intent_type: str) -> bool:
        """
        注销 Handler
        
        Args:
            intent_type: 意图类型名称
            
        Returns:
            是否成功注销
        """
        if intent_type in cls._handlers:
            del cls._handlers[intent_type]
            cls._metadata.pop(intent_type, None)
            return True
        return False
    
    @classmethod
    def clear(cls) -> None:
        """清空注册表（主要用于测试）"""
        cls._handlers = {}
        cls._metadata = {}


def register_handler(
    intent_type: str,
    priority: int = 0,
    description: str = ""
) -> Callable:
    """
    注册 Handler 的便捷装饰器
    
    Args:
        intent_type: 意图类型名称
        priority: 优先级
        description: Handler 描述
        
    Usage:
        @register_handler("chat", priority=100, description="处理闲聊对话")
        class ChatHandler(BaseHandler):
            def __init__(self, llm):
                self.llm = llm
            
            def handle(self, intent):
                return self.llm.invoke(intent.query).content
    """
    return HandlerRegistry.register(
        intent_type=intent_type,
        priority=priority,
        description=description,
    )
