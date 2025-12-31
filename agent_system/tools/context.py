"""
工具执行上下文

使用 contextvars 在请求级别传递用户相关信息（如 token）
这样工具可以在执行时获取当前用户的认证信息，而不需要修改工具签名
"""

from contextvars import ContextVar
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ToolContext:
    """
    工具执行上下文
    
    包含工具执行时可能需要的用户相关信息
    """
    user_id: Optional[str] = None
    token: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取额外的上下文数据"""
        return self.extra.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置额外的上下文数据"""
        self.extra[key] = value


# 全局上下文变量（线程/协程安全）
_tool_context: ContextVar[Optional[ToolContext]] = ContextVar(
    'tool_context', 
    default=None
)


def get_current_context() -> Optional[ToolContext]:
    """
    获取当前请求的工具上下文
    
    Returns:
        当前上下文，如果未设置则返回 None
    """
    return _tool_context.get()


def get_current_token() -> Optional[str]:
    """
    获取当前用户的 token
    
    便捷方法，工具可以直接调用获取 token
    
    Returns:
        当前用户的 token，如果未设置则返回 None
    """
    ctx = get_current_context()
    return ctx.token if ctx else None


def get_current_user_id() -> Optional[str]:
    """
    获取当前用户 ID
    
    Returns:
        当前用户 ID，如果未设置则返回 None
    """
    ctx = get_current_context()
    return ctx.user_id if ctx else None


def set_context(
    user_id: Optional[str] = None,
    token: Optional[str] = None,
    **extra
) -> ToolContext:
    """
    设置当前请求的工具上下文
    
    通常在 API 路由处理函数开始时调用
    
    Args:
        user_id: 用户 ID
        token: 用户认证 token
        **extra: 其他需要传递给工具的数据
        
    Returns:
        创建的上下文对象
    """
    ctx = ToolContext(
        user_id=user_id,
        token=token,
        extra=extra
    )
    _tool_context.set(ctx)
    return ctx


def clear_context():
    """
    清除当前请求的工具上下文
    
    通常在请求结束时调用（可选，因为 contextvars 会自动隔离）
    """
    _tool_context.set(None)


class ContextManager:
    """
    上下文管理器，用于 with 语句
    
    用法:
        with ContextManager(user_id="1", token="xxx"):
            # 在这个块内，工具可以访问 token
            agent.run(task)
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        token: Optional[str] = None,
        **extra
    ):
        self.user_id = user_id
        self.token = token
        self.extra = extra
        self._token = None
    
    def __enter__(self):
        self._token = _tool_context.set(ToolContext(
            user_id=self.user_id,
            token=self.token,
            extra=self.extra
        ))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _tool_context.reset(self._token)
        return False

