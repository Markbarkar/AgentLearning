"""
用户隔离工具模块

提供统一的用户隔离逻辑，供各存储适配器使用
"""

import re
from typing import Optional

from ...config.settings import (
    ENABLE_USER_ISOLATION,
    PUBLIC_COLLECTION_NAME,
)


def validate_user_id(user_id: Optional[str]) -> Optional[str]:
    """
    验证并清理 user_id
    
    Args:
        user_id: 原始用户ID
        
    Returns:
        清理后的用户ID，如果无效则返回 None
    """
    if not user_id:
        return None
    
    # 只保留字母数字下划线，限制长度为32字符
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '', str(user_id))[:32]
    return cleaned if cleaned else None


def get_collection_name(
    user_id: Optional[str],
    prefix: str = "legal_kb",
    enable_isolation: bool = None
) -> str:
    """
    根据 user_id 生成 collection 名称
    
    Args:
        user_id: 用户ID，如果为 None 则返回公共知识库名称
        prefix: collection 名称前缀
        enable_isolation: 是否启用用户隔离，None 则使用配置值
        
    Returns:
        collection 名称
    """
    if enable_isolation is None:
        enable_isolation = ENABLE_USER_ISOLATION
    
    if not enable_isolation or not user_id:
        return f"{prefix}_public"
    
    # 清理 user_id
    cleaned_user_id = validate_user_id(user_id)
    if not cleaned_user_id:
        return f"{prefix}_public"
    
    return f"{prefix}_user_{cleaned_user_id}"


class UserIsolationMixin:
    """
    用户隔离 Mixin 类
    
    为存储类提供用户隔离功能
    """
    
    def __init__(
        self,
        user_id: Optional[str] = None,
        collection_prefix: str = "legal_kb",
        enable_isolation: bool = None,
        **kwargs
    ):
        """
        初始化用户隔离
        
        Args:
            user_id: 用户ID
            collection_prefix: collection 名称前缀
            enable_isolation: 是否启用用户隔离
        """
        self.user_id = validate_user_id(user_id)
        self.collection_prefix = collection_prefix
        self.enable_isolation = enable_isolation if enable_isolation is not None else ENABLE_USER_ISOLATION
        self.collection_name = get_collection_name(
            self.user_id,
            self.collection_prefix,
            self.enable_isolation
        )
    
    @property
    def is_public(self) -> bool:
        """是否为公共知识库"""
        return not self.enable_isolation or not self.user_id
    
    def get_user_display_name(self) -> str:
        """获取用户显示名称"""
        return self.user_id or "public"
