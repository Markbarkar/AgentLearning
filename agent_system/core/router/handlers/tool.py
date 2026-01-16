"""
单工具调用处理器（已弃用）

注意：此 Handler 已弃用！
tool_call 意图现在由 ReActHandler 处理，以保留完整的 ReAct 循环能力。

保留此文件仅供参考和向后兼容。
"""

from typing import List, Optional, Any, Dict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from .base import BaseHandler
# from ..registry import register_handler  # 不再注册
from ...intent.models import Intent


# LLM 工具选择提示词
TOOL_SELECTION_PROMPT = """根据用户的查询，从以下可用工具中选择最合适的一个。

可用工具:
{tools_desc}

用户查询: {query}

请分析用户需求，选择最合适的工具。只返回工具名称，不要有任何其他内容。
如果用户只是在询问能否使用某功能（如"能调用云盘吗"），请选择能展示该功能的工具（如 file-list）。

工具名称:"""


# 注意：不再使用 @register_handler 装饰器，此 Handler 已弃用
# tool_call 意图由 ReActHandler 处理
class DirectToolHandler(BaseHandler):
    """
    单工具直接调用处理器
    
    当意图明确指向单个工具时，直接调用该工具
    优先使用 intent.selected_tools 中的精选工具
    """
    
    def __init__(
        self, 
        tools: Optional[List[BaseTool]] = None,
        llm: Optional[BaseChatModel] = None
    ):
        """
        初始化
        
        Args:
            tools: 工具列表（作为备用，优先使用 intent.selected_tools）
            llm: 语言模型实例（用于智能工具选择和参数提取）
        """
        self.tools = tools or []
        self.llm = llm
    
    def handle(self, intent: Intent) -> str:
        """
        处理单工具调用意图
        
        流程：
        1. 优先使用 intent.selected_tools（精选工具）
        2. 使用 LLM 智能选择最合适的工具
        3. 提取参数并调用工具
        
        Args:
            intent: 意图识别结果
            
        Returns:
            工具执行结果
        """
        query = intent.query
        
        # 1. 确定可用工具（优先使用精选工具）
        available_tools = intent.selected_tools if intent.selected_tools else self.tools
        
        if not available_tools:
            return "没有可用的工具。"
        
        # 构建工具索引
        tool_map: Dict[str, BaseTool] = {
            getattr(t, 'name', str(t)): t for t in available_tools
        }
        
        # 2. 尝试从元数据获取工具名
        tool_name = intent.metadata.get("tool_name")
        
        # 3. 如果没有明确工具名，使用 LLM 智能选择
        if not tool_name or tool_name not in tool_map:
            tool_name = self._smart_select_tool(query, available_tools)
        
        # 4. 如果还是没有，尝试简单匹配
        if not tool_name or tool_name not in tool_map:
            tool_name = self._match_tool_from_query(query, tool_map)
        
        if not tool_name or tool_name not in tool_map:
            # 列出可用工具供用户参考
            tool_names = ", ".join(tool_map.keys())
            matched_groups = intent.matched_groups if intent.matched_groups else ["未知"]
            return (
                f"已识别到工具组: {', '.join(matched_groups)}\n"
                f"可用工具: {tool_names}\n"
                f"请更明确地描述您想要执行的操作。"
            )
        
        # 5. 获取工具并执行
        tool = tool_map[tool_name]
        
        try:
            result = self._invoke_tool(tool, query)
            return result
        except Exception as e:
            return f"工具 '{tool_name}' 执行失败: {str(e)}"
    
    def _smart_select_tool(self, query: str, tools: List) -> Optional[str]:
        """
        使用 LLM 智能选择工具
        
        Args:
            query: 用户查询
            tools: 可用工具列表
            
        Returns:
            选择的工具名，失败返回 None
        """
        if not self.llm or not tools:
            return None
        
        # 构建工具描述
        tools_desc = "\n".join([
            f"- {getattr(t, 'name', str(t))}: {getattr(t, 'description', '无描述')[:100]}"
            for t in tools
        ])
        
        prompt = TOOL_SELECTION_PROMPT.format(
            tools_desc=tools_desc,
            query=query
        )
        
        try:
            response = self.llm.invoke(prompt)
            tool_name = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 验证工具名是否存在
            valid_names = {getattr(t, 'name', str(t)) for t in tools}
            if tool_name in valid_names:
                print(f"🎯 LLM 选择工具: {tool_name}")
                return tool_name
        except Exception as e:
            print(f"⚠️ LLM 工具选择失败: {e}")
        
        return None
    
    def _match_tool_from_query(self, query: str, tool_map: Dict[str, Any]) -> Optional[str]:
        """
        从查询中匹配工具名
        
        Args:
            query: 用户查询
            tool_map: 工具名到工具的映射
            
        Returns:
            匹配到的工具名，未匹配返回 None
        """
        query_lower = query.lower()
        
        for tool_name in tool_map:
            if tool_name.lower() in query_lower:
                return tool_name
        
        return None
    
    def _invoke_tool(self, tool: BaseTool, query: str) -> str:
        """
        调用工具
        
        Args:
            tool: 工具实例
            query: 用户查询
            
        Returns:
            工具执行结果
        """
        try:
            # 直接用查询作为输入调用工具
            result = tool.run(query)
            return str(result)
        except Exception as e:
            # 尝试提取参数后再调用
            if self.llm:
                return self._invoke_with_extracted_args(tool, query)
            raise e
    
    def _invoke_with_extracted_args(self, tool: BaseTool, query: str) -> str:
        """
        使用 LLM 提取参数后调用工具
        
        Args:
            tool: 工具实例
            query: 用户查询
            
        Returns:
            工具执行结果
        """
        if not self.llm:
            raise ValueError("需要 LLM 来提取工具参数")
        
        prompt = f"""从以下用户查询中提取工具所需的参数。

工具名称: {tool.name}
工具描述: {tool.description}
用户查询: {query}

请直接输出工具的输入参数（如果是文件路径，直接输出路径；如果不需要参数，输出空字符串）:"""
        
        try:
            response = self.llm.invoke(prompt)
            args = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 使用提取的参数调用工具
            if args:
                result = tool.run(args)
            else:
                result = tool.run("")
            return str(result)
            
        except Exception as e:
            raise ValueError(f"参数提取或工具调用失败: {e}")
    
    def can_handle(self, intent: Intent) -> bool:
        """检查是否可以处理"""
        # 优先检查精选工具
        if intent.selected_tools:
            return len(intent.selected_tools) > 0
        return len(self.tools) > 0
    
    def get_available_tools(self, intent: Optional[Intent] = None) -> List[str]:
        """获取可用工具列表"""
        if intent and intent.selected_tools:
            return [getattr(t, 'name', str(t)) for t in intent.selected_tools]
        return [getattr(t, 'name', str(t)) for t in self.tools]
