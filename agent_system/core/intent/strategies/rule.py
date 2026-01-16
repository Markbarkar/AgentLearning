"""
规则分类策略

基于关键词和正则表达式匹配进行意图分类
规则从配置文件动态加载
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from functools import lru_cache

from .base import ClassifierStrategy, register_strategy
from ..models import Intent


@dataclass
class RuleConfig:
    """单个意图类型的规则配置"""
    intent_type: str
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    tool_indicators: bool = False  # 是否检查工具名匹配
    priority: int = 50
    confidence: float = 0.8
    
    # 编译后的正则表达式（缓存）
    _compiled_patterns: List[re.Pattern] = field(default_factory=list, repr=False)
    
    def __post_init__(self):
        """编译正则表达式"""
        self._compiled_patterns = []
        for pattern in self.patterns:
            try:
                self._compiled_patterns.append(re.compile(pattern, re.IGNORECASE))
            except re.error:
                pass  # 跳过无效的正则表达式


def _get_rules_config_path() -> Path:
    """获取规则配置文件路径"""
    return Path(__file__).parent.parent.parent.parent / "config" / "intent_rules.json"


@lru_cache(maxsize=1)
def load_rules_config(config_path: Optional[str] = None) -> Dict[str, RuleConfig]:
    """
    加载规则配置（带缓存）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        {意图类型: RuleConfig} 字典
    """
    path = Path(config_path) if config_path else _get_rules_config_path()
    
    if not path.exists():
        return {}
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    rules = {}
    for intent_type, config in data.items():
        rules[intent_type] = RuleConfig(
            intent_type=intent_type,
            keywords=config.get("keywords", []),
            patterns=config.get("patterns", []),
            tool_indicators=config.get("tool_indicators", False),
            priority=config.get("priority", 50),
            confidence=config.get("confidence", 0.8),
        )
    
    return rules


def reload_rules_config() -> Dict[str, RuleConfig]:
    """重新加载规则配置（清除缓存）"""
    load_rules_config.cache_clear()
    return load_rules_config()


@register_strategy("rule", priority=100)
class RuleClassifierStrategy(ClassifierStrategy):
    """
    规则分类策略
    
    基于关键词和正则表达式匹配
    优先级最高，用于处理明确的意图
    """
    
    priority = 100
    name = "rule"
    
    def __init__(self, rules_config_path: Optional[str] = None, tools: Optional[List] = None):
        """
        初始化规则分类器
        
        Args:
            rules_config_path: 规则配置文件路径
            tools: 工具列表（用于 tool_call 意图的工具名匹配）
        """
        self.rules_config_path = rules_config_path
        self.tools = tools or []
        self._rules: Optional[Dict[str, RuleConfig]] = None
    
    @property
    def rules(self) -> Dict[str, RuleConfig]:
        """懒加载规则配置"""
        if self._rules is None:
            self._rules = load_rules_config(self.rules_config_path)
        return self._rules
    
    def reload_rules(self):
        """重新加载规则"""
        self._rules = reload_rules_config()
    
    def classify(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Intent]:
        """
        基于规则分类
        
        匹配顺序：
        1. 按规则优先级排序
        2. 先检查关键词匹配
        3. 再检查正则表达式匹配
        4. 对于 tool_call，还检查工具名匹配
        """
        context = context or {}
        query_lower = query.lower().strip()
        
        # 获取上下文中的工具列表
        tools = context.get("tools", self.tools)
        
        # 存储匹配结果：(置信度, 优先级, 意图类型)
        matches: List[tuple] = []
        
        # 按优先级排序规则
        sorted_rules = sorted(
            self.rules.values(),
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            match_result = self._match_rule(query, query_lower, rule, tools)
            if match_result:
                confidence, match_type = match_result
                matches.append((confidence, rule.priority, rule.intent_type, match_type))
        
        if not matches:
            return None
        
        # 选择最佳匹配（先按置信度，再按优先级）
        best_match = max(matches, key=lambda m: (m[0], m[1]))
        confidence, priority, intent_type, match_type = best_match
        
        return Intent(
            type=intent_type,
            query=query,
            confidence=confidence,
            metadata={
                "strategy": "rule",
                "match_type": match_type,
                "priority": priority,
            }
        )
    
    def _match_rule(
        self, 
        query: str, 
        query_lower: str, 
        rule: RuleConfig,
        tools: List
    ) -> Optional[tuple]:
        """
        匹配单个规则
        
        Returns:
            (置信度, 匹配类型) 或 None
        """
        # 1. 关键词匹配
        for keyword in rule.keywords:
            if keyword.lower() in query_lower:
                return (rule.confidence, "keyword")
        
        # 2. 正则表达式匹配
        for pattern in rule._compiled_patterns:
            if pattern.search(query):
                return (rule.confidence, "pattern")
        
        # 3. 工具名匹配（仅对启用了 tool_indicators 的规则）
        if rule.tool_indicators and tools:
            for tool in tools:
                tool_name = getattr(tool, 'name', str(tool))
                if tool_name.lower() in query_lower:
                    return (rule.confidence, "tool_name")
        
        return None
