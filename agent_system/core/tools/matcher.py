"""
LLM 工具匹配器

当规则匹配失败时，使用 LLM 理解用户意图并匹配工具组
"""

import json
from typing import List, Optional, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .selector import ToolGroupConfig, load_tool_groups_config


# 工具组匹配提示词
TOOL_GROUP_MATCH_PROMPT = """你是一个工具选择助手。根据用户的查询，判断需要使用哪些工具组。

可用的工具组：
{groups_desc}

用户查询：{query}

请分析用户想要使用哪些工具组。
- 如果用户想查看文件、上传、下载、分享文件，选择 "cloud_drive"
- 如果用户想提取PDF文字、识别表格，选择 "ocr"
- 如果用户想处理法律文书、查询案例，选择 "legal"
- 如果用户想执行系统命令，选择 "system"
- 如果用户想操作项目表格或关联文件，选择 "project"
- 如果无法确定，返回空数组

请只返回工具组名称的JSON数组，例如：["cloud_drive"] 或 ["ocr", "legal"]
不要返回任何其他内容。

JSON数组："""


class ToolMatcher:
    """
    LLM 工具匹配器
    
    使用 LLM 理解用户查询并匹配到合适的工具组
    
    使用方式:
        >>> matcher = ToolMatcher(llm=llm)
        >>> groups = matcher.match_groups("列出云盘文件")
        >>> print(groups)  # ["cloud_drive"]
    """
    
    def __init__(
        self,
        llm: BaseChatModel,
        config: Optional[ToolGroupConfig] = None,
    ):
        """
        初始化
        
        Args:
            llm: 语言模型实例
            config: 工具分组配置
        """
        self.llm = llm
        self.config = config or load_tool_groups_config()
        self._prompt = PromptTemplate.from_template(TOOL_GROUP_MATCH_PROMPT)
    
    def match_groups(self, query: str) -> List[str]:
        """
        使用 LLM 匹配工具组
        
        Args:
            query: 用户查询
            
        Returns:
            匹配到的工具组名称列表
        """
        try:
            # 构建工具组描述
            groups_desc = self._build_groups_description()
            
            # 调用 LLM
            chain = self._prompt | self.llm | StrOutputParser()
            result = chain.invoke({
                "query": query,
                "groups_desc": groups_desc,
            })
            
            # 解析结果
            return self._parse_result(result)
            
        except Exception as e:
            print(f"⚠️ LLM 工具匹配失败: {e}")
            return []
    
    def _build_groups_description(self) -> str:
        """构建工具组描述文本"""
        lines = []
        for name, group in self.config.groups.items():
            tools_str = ", ".join(group.tools[:5])  # 只显示前5个工具
            if len(group.tools) > 5:
                tools_str += f" 等{len(group.tools)}个工具"
            lines.append(f"- {name}: {group.description} (包含: {tools_str})")
        return "\n".join(lines)
    
    def _parse_result(self, result: str) -> List[str]:
        """
        解析 LLM 返回的结果
        
        Args:
            result: LLM 返回的原始文本
            
        Returns:
            工具组名称列表
        """
        result = result.strip()
        
        # 尝试直接解析 JSON
        try:
            groups = json.loads(result)
            if isinstance(groups, list):
                # 验证工具组名称是否有效
                valid_groups = [g for g in groups if g in self.config.groups]
                return valid_groups
        except json.JSONDecodeError:
            pass
        
        # 尝试从文本中提取 JSON 数组
        import re
        match = re.search(r'\[([^\]]*)\]', result)
        if match:
            try:
                groups = json.loads(match.group(0))
                if isinstance(groups, list):
                    valid_groups = [g for g in groups if g in self.config.groups]
                    return valid_groups
            except json.JSONDecodeError:
                pass
        
        # 尝试从文本中提取工具组名称
        found = []
        for name in self.config.groups:
            if name in result.lower():
                found.append(name)
        
        return found
    
    def match_tools_directly(self, query: str, all_tools: List) -> List[str]:
        """
        直接匹配具体的工具名（不通过工具组）
        
        用于更精确的工具选择场景
        
        Args:
            query: 用户查询
            all_tools: 全部可用工具
            
        Returns:
            匹配到的工具名列表
        """
        # 构建工具描述
        tools_desc = "\n".join([
            f"- {getattr(t, 'name', str(t))}: {getattr(t, 'description', '')[:100]}"
            for t in all_tools
        ])
        
        prompt = f"""根据用户查询，从以下工具中选择最相关的工具。

可用工具：
{tools_desc}

用户查询：{query}

请返回最相关的工具名称列表（JSON数组），最多3个："""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content if hasattr(response, 'content') else str(response)
            
            # 解析结果
            import re
            match = re.search(r'\[([^\]]*)\]', result)
            if match:
                tools = json.loads(match.group(0))
                # 验证工具名是否存在
                tool_names = {getattr(t, 'name', str(t)) for t in all_tools}
                return [t for t in tools if t in tool_names]
        except Exception as e:
            print(f"⚠️ 直接工具匹配失败: {e}")
        
        return []
