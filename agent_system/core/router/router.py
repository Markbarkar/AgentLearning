"""
意图路由器

根据意图类型将请求分发到对应的 Handler
从注册表动态加载 Handler，支持运行时扩展
"""

from typing import Dict, Optional, Type

from ..intent.models import Intent, load_intent_config
from .registry import HandlerRegistry
from .factory import HandlerFactory
from .handlers.base import BaseHandler


class IntentRouter:
    """
    意图路由器
    
    负责将意图分发到对应的 Handler 处理
    支持 Handler 懒加载和缓存
    
    使用方式:
        >>> factory = HandlerFactory(llm=llm, knowledge_base=kb, ...)
        >>> router = IntentRouter(factory)
        >>> result = router.route(intent)
    """
    
    def __init__(
        self, 
        factory: HandlerFactory,
        fallback_handler_type: Optional[str] = None
    ):
        """
        初始化路由器
        
        Args:
            factory: Handler 工厂，用于创建 Handler 实例
            fallback_handler_type: 兜底 Handler 类型，未找到匹配时使用
        """
        self.factory = factory
        self._handler_cache: Dict[str, BaseHandler] = {}
        
        # 加载配置获取默认的 fallback
        config = load_intent_config()
        self.fallback_handler_type = (
            fallback_handler_type or config.fallback_handler
        )
    
    def route(self, intent: Intent) -> str:
        """
        路由意图到对应的 Handler
        
        流程：
        1. 从缓存或注册表获取 Handler
        2. 检查 Handler 是否可以处理
        3. 调用 Handler 处理意图
        4. 返回处理结果
        
        Args:
            intent: 意图识别结果
            
        Returns:
            处理结果字符串
        """
        try:
            # 获取或创建 Handler
            handler = self._get_or_create_handler(intent.type)
            
            # 检查是否可以处理
            if not handler.can_handle(intent):
                # 尝试使用 fallback handler
                print(f"⚠️ Handler {handler} 无法处理意图 {intent.type}，使用 fallback")
                handler = self._get_or_create_handler(self.fallback_handler_type)
            
            # 处理意图
            result = handler.handle(intent)
            return result
            
        except Exception as e:
            return f"处理意图时发生错误: {str(e)}"
    
    def _get_or_create_handler(self, intent_type: str) -> BaseHandler:
        """
        获取或创建 Handler 实例
        
        使用缓存避免重复创建
        
        Args:
            intent_type: 意图类型
            
        Returns:
            Handler 实例
        """
        # 先检查缓存
        if intent_type in self._handler_cache:
            return self._handler_cache[intent_type]
        
        # 从注册表获取 Handler 类
        handler_cls = HandlerRegistry.get(intent_type)
        
        if handler_cls is None:
            # 未找到，尝试使用 fallback
            print(f"⚠️ 未找到意图类型 '{intent_type}' 的 Handler，使用 fallback")
            handler_cls = HandlerRegistry.get(self.fallback_handler_type)
            
            if handler_cls is None:
                raise ValueError(
                    f"找不到意图类型 '{intent_type}' 的 Handler，"
                    f"且 fallback '{self.fallback_handler_type}' 也未注册"
                )
        
        # 使用工厂创建实例
        handler = self.factory.create(handler_cls)
        
        # 缓存实例
        self._handler_cache[intent_type] = handler
        
        return handler
    
    def get_handler(self, intent_type: str) -> Optional[BaseHandler]:
        """
        获取指定类型的 Handler（不创建）
        
        Args:
            intent_type: 意图类型
            
        Returns:
            Handler 实例，未找到返回 None
        """
        return self._handler_cache.get(intent_type)
    
    def preload_handlers(self, intent_types: Optional[list] = None) -> Dict[str, BaseHandler]:
        """
        预加载 Handler
        
        在路由之前预先创建 Handler 实例，避免首次调用的延迟
        
        Args:
            intent_types: 要预加载的意图类型列表，为 None 则加载所有已注册的
            
        Returns:
            成功预加载的 Handler 字典
        """
        if intent_types is None:
            intent_types = HandlerRegistry.get_registered_types()
        
        loaded = {}
        for intent_type in intent_types:
            try:
                handler = self._get_or_create_handler(intent_type)
                loaded[intent_type] = handler
                print(f"✓ 预加载 Handler: {intent_type} -> {handler}")
            except Exception as e:
                print(f"⚠️ 预加载 Handler 失败: {intent_type}, 错误: {e}")
        
        return loaded
    
    def clear_cache(self, intent_type: Optional[str] = None) -> None:
        """
        清除 Handler 缓存
        
        Args:
            intent_type: 要清除的意图类型，为 None 则清除所有
        """
        if intent_type:
            self._handler_cache.pop(intent_type, None)
        else:
            self._handler_cache.clear()
    
    def get_cached_handlers(self) -> Dict[str, BaseHandler]:
        """获取所有已缓存的 Handler"""
        return self._handler_cache.copy()
    
    def set_fallback(self, intent_type: str) -> None:
        """
        设置 fallback Handler 类型
        
        Args:
            intent_type: 意图类型
        """
        if not HandlerRegistry.is_registered(intent_type):
            raise ValueError(f"意图类型 '{intent_type}' 未注册")
        self.fallback_handler_type = intent_type
    
    def get_available_handlers(self) -> Dict[str, Dict]:
        """
        获取所有可用的 Handler 信息
        
        Returns:
            {意图类型: {class_name, priority, description, can_create}} 字典
        """
        result = {}
        for intent_type in HandlerRegistry.get_registered_types():
            metadata = HandlerRegistry.get_metadata(intent_type) or {}
            handler_cls = HandlerRegistry.get(intent_type)
            
            result[intent_type] = {
                **metadata,
                "can_create": self.factory.can_create(handler_cls) if handler_cls else False,
                "cached": intent_type in self._handler_cache,
            }
        
        return result
    
    def __repr__(self) -> str:
        cached = list(self._handler_cache.keys())
        return f"IntentRouter(cached={cached}, fallback={self.fallback_handler_type})"
