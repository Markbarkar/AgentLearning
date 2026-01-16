"""
工具选择器

根据意图和查询动态选择相关工具
支持工具分组、别名匹配和 LLM 智能匹配
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from functools import lru_cache

from ..intent.models import Intent


@dataclass
class ToolGroup:
    """工具组配置"""
    name: str
    description: str
    tools: List[str]
    aliases: List[str] = field(default_factory=list)
    
    def matches_query(self, query: str) -> bool:
        """检查查询是否匹配该工具组的别名"""
        query_lower = query.lower()
        for alias in self.aliases:
            if alias.lower() in query_lower:
                return True
        return False


@dataclass
class ToolGroupConfig:
    """工具分组配置"""
    groups: Dict[str, ToolGroup]
    intent_mapping: Dict[str, List[str]]  # {意图类型: [工具组名称]}
    default_tools: List[str]
    always_include: List[str]
    
    def get_groups_for_intent(self, intent_type: str) -> List[str]:
        """获取意图类型关联的工具组"""
        return self.intent_mapping.get(intent_type, [])
    
    def get_tools_for_groups(self, group_names: List[str]) -> Set[str]:
        """获取指定工具组包含的所有工具名"""
        tools = set()
        for name in group_names:
            if name in self.groups:
                tools.update(self.groups[name].tools)
        return tools
    
    def match_groups_from_query(self, query: str) -> List[str]:
        """从查询中匹配工具组（通过别名）"""
        matched = []
        for name, group in self.groups.items():
            if group.matches_query(query):
                matched.append(name)
        return matched
    
    def get_all_group_descriptions(self) -> Dict[str, str]:
        """获取所有工具组的描述"""
        return {name: group.description for name, group in self.groups.items()}


def _get_config_path() -> Path:
    """获取工具分组配置文件路径"""
    return Path(__file__).parent.parent.parent / "config" / "tool_groups.json"


@lru_cache(maxsize=1)
def load_tool_groups_config(config_path: Optional[str] = None) -> ToolGroupConfig:
    """
    加载工具分组配置（带缓存）
    
    Args:
        config_path: 配置文件路径，为 None 则使用默认路径
        
    Returns:
        ToolGroupConfig 实例
    """
    path = Path(config_path) if config_path else _get_config_path()
    
    if not path.exists():
        # 返回默认配置
        return ToolGroupConfig(
            groups={},
            intent_mapping={
                "chat": [],
                "knowledge_qa": [],
                "tool_call": [],
                "complex_task": [],
            },
            default_tools=["FINISH"],
            always_include=["FINISH"],
        )
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 解析工具组
    groups = {}
    for name, group_data in data.get("groups", {}).items():
        groups[name] = ToolGroup(
            name=name,
            description=group_data.get("description", ""),
            tools=group_data.get("tools", []),
            aliases=group_data.get("aliases", []),
        )
    
    return ToolGroupConfig(
        groups=groups,
        intent_mapping=data.get("intent_mapping", {}),
        default_tools=data.get("default_tools", ["FINISH"]),
        always_include=data.get("always_include", ["FINISH"]),
    )


def reload_tool_groups_config() -> ToolGroupConfig:
    """重新加载配置（清除缓存）"""
    load_tool_groups_config.cache_clear()
    return load_tool_groups_config()


class ToolSelector:
    """
    工具选择器
    
    根据意图和查询动态选择相关工具
    
    使用方式:
        >>> selector = ToolSelector()
        >>> intent = Intent(type="tool_call", query="列出云盘文件")
        >>> selected = selector.select_tools(intent, all_tools)
    """
    
    def __init__(
        self,
        config: Optional[ToolGroupConfig] = None,
        matcher: Optional[Any] = None,  # ToolMatcher 实例
    ):
        """
        初始化
        
        Args:
            config: 工具分组配置，为 None 则从文件加载
            matcher: LLM 工具匹配器实例（可选）
        """
        self.config = config or load_tool_groups_config()
        self.matcher = matcher
    
    def set_matcher(self, matcher: Any) -> None:
        """设置 LLM 匹配器"""
        self.matcher = matcher
    
    def select_tools(
        self,
        intent: Intent,
        all_tools: List,
        use_llm_fallback: bool = True
    ) -> List:
        """
        根据意图选择工具
        
        选择流程：
        1. 从查询中识别工具组（通过别名匹配）- 优先级最高
        2. 如果没有从查询匹配到，使用意图类型关联的工具组
        3. 如果还是没有匹配到且启用 LLM，使用 LLM 智能匹配
        4. 过滤工具列表，保留匹配的工具
        5. 添加必须包含的工具（如 FINISH）
        
        Args:
            intent: 意图识别结果
            all_tools: 全部可用工具列表
            use_llm_fallback: 是否在规则匹配失败时使用 LLM
            
        Returns:
            筛选后的工具列表
        """
        # 1. 先从查询中识别工具组（优先级最高）
        query_matched = self.config.match_groups_from_query(intent.query)
        
        if query_matched:
            # 查询中有明确的别名匹配，只使用这些组
            groups = set(query_matched)
        else:
            # 2. 没有从查询匹配到，使用意图类型关联的工具组
            groups = set(self.config.get_groups_for_intent(intent.type))
        
        # 3. 如果没有匹配到且有 LLM 匹配器，尝试 LLM 匹配
        if not groups and use_llm_fallback and self.matcher:
            llm_matched = self.matcher.match_groups(intent.query)
            groups.update(llm_matched)
        
        # 4. 获取工具名集合
        if groups:
            tool_names = self.config.get_tools_for_groups(list(groups))
        else:
            # 如果还是没有匹配，返回所有工具（降级）
            tool_names = None
        
        # 5. 过滤工具列表
        selected = self._filter_tools(all_tools, tool_names)
        
        # 6. 添加必须包含的工具
        selected = self._add_always_include(selected, all_tools)
        
        # 记录选择结果
        group_names = list(groups) if groups else ["all"]
        print(f"🔧 工具选择: 意图={intent.type}, 匹配组={group_names}, 选中={len(selected)}个工具")
        
        return selected
    
    def _filter_tools(
        self,
        all_tools: List,
        tool_names: Optional[Set[str]]
    ) -> List:
        """
        过滤工具列表
        
        Args:
            all_tools: 全部工具
            tool_names: 要保留的工具名集合，为 None 则保留全部
            
        Returns:
            过滤后的工具列表
        """
        if tool_names is None:
            return list(all_tools)
        
        selected = []
        for tool in all_tools:
            tool_name = getattr(tool, 'name', str(tool))
            if tool_name in tool_names:
                selected.append(tool)
        
        return selected
    
    def _add_always_include(self, selected: List, all_tools: List) -> List:
        """
        添加必须包含的工具
        
        Args:
            selected: 已选择的工具
            all_tools: 全部工具
            
        Returns:
            添加后的工具列表
        """
        selected_names = {getattr(t, 'name', str(t)) for t in selected}
        
        for tool in all_tools:
            tool_name = getattr(tool, 'name', str(tool))
            if tool_name in self.config.always_include and tool_name not in selected_names:
                selected.append(tool)
        
        return selected
    
    def get_available_groups(self) -> Dict[str, str]:
        """获取所有可用的工具组及其描述"""
        return self.config.get_all_group_descriptions()
    
    def get_tools_in_group(self, group_name: str) -> List[str]:
        """获取指定工具组包含的工具名"""
        if group_name in self.config.groups:
            return self.config.groups[group_name].tools
        return []
