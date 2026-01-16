"""
意图数据模型

支持动态类型加载，意图类型从配置文件读取
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class IntentTypeInfo:
    """意图类型信息"""
    name: str
    description: str
    priority: int = 0


@dataclass
class IntentConfig:
    """意图配置"""
    intent_types: List[IntentTypeInfo]
    default_intent: str
    confidence_threshold: float
    fallback_handler: str
    
    def get_type_names(self) -> List[str]:
        """获取所有意图类型名称"""
        return [t.name for t in self.intent_types]
    
    def is_valid_type(self, type_name: str) -> bool:
        """检查是否是有效的意图类型"""
        return type_name in self.get_type_names()


@dataclass
class Intent:
    """
    意图识别结果
    
    使用字符串类型而非枚举，支持动态扩展
    """
    type: str                           # 意图类型（动态，从配置加载）
    query: str                          # 原始查询
    confidence: float = 1.0             # 置信度 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)  # 附加信息
    selected_tools: List[Any] = field(default_factory=list)  # 精选的工具列表
    matched_groups: List[str] = field(default_factory=list)  # 匹配到的工具组
    
    def __post_init__(self):
        """验证置信度范围"""
        if not 0 <= self.confidence <= 1:
            self.confidence = max(0, min(1, self.confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": self.type,
            "query": self.query,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "selected_tools": [getattr(t, 'name', str(t)) for t in self.selected_tools],
            "matched_groups": self.matched_groups,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Intent":
        """从字典创建"""
        return cls(
            type=data["type"],
            query=data["query"],
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {}),
            selected_tools=data.get("selected_tools", []),
            matched_groups=data.get("matched_groups", []),
        )
    
    def get_tool_names(self) -> List[str]:
        """获取选中的工具名称列表"""
        return [getattr(t, 'name', str(t)) for t in self.selected_tools]


def _get_config_path() -> Path:
    """获取配置文件路径"""
    return Path(__file__).parent.parent.parent / "config" / "intent_config.json"


@lru_cache(maxsize=1)
def load_intent_config(config_path: Optional[str] = None) -> IntentConfig:
    """
    加载意图配置（带缓存）
    
    Args:
        config_path: 配置文件路径，为 None 则使用默认路径
        
    Returns:
        IntentConfig 实例
    """
    path = Path(config_path) if config_path else _get_config_path()
    
    if not path.exists():
        # 返回默认配置
        return IntentConfig(
            intent_types=[
                IntentTypeInfo("chat", "闲聊对话", 1),
                IntentTypeInfo("knowledge_qa", "知识问答", 2),
                IntentTypeInfo("tool_call", "单工具调用", 3),
                IntentTypeInfo("complex_task", "复杂任务", 4),
            ],
            default_intent="complex_task",
            confidence_threshold=0.7,
            fallback_handler="complex_task",
        )
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    intent_types = [
        IntentTypeInfo(
            name=t["name"],
            description=t.get("description", ""),
            priority=t.get("priority", 0),
        )
        for t in data.get("intent_types", [])
    ]
    
    return IntentConfig(
        intent_types=intent_types,
        default_intent=data.get("default_intent", "complex_task"),
        confidence_threshold=data.get("confidence_threshold", 0.7),
        fallback_handler=data.get("fallback_handler", "complex_task"),
    )


def reload_intent_config() -> IntentConfig:
    """
    重新加载意图配置（清除缓存）
    
    用于配置文件更新后重新加载
    """
    load_intent_config.cache_clear()
    return load_intent_config()
