"""
ReAct 处理器

处理需要工具调用的任务，使用完整的 ReAct Agent 进行推理和执行
支持使用 intent.selected_tools 精选工具

同时处理 tool_call 和 complex_task 两种意图类型
"""

from typing import Optional, Any, List

from langchain_core.language_models import BaseChatModel

from .base import BaseHandler
from ..registry import register_handler
from ...intent.models import Intent


@register_handler("tool_call", priority=70, description="处理工具调用任务")
@register_handler("complex_task", priority=50, description="处理复杂任务")
class ReActHandler(BaseHandler):
    """
    ReAct Agent 处理器
    
    处理所有需要工具调用的任务，包括：
    - tool_call: 需要调用工具的任务
    - complex_task: 复杂的多步骤任务
    
    使用完整的 ReAct 循环（Think-Act-Observe）进行多轮推理
    优先使用 intent.selected_tools 中的精选工具创建临时 Agent
    """
    
    def __init__(
        self,
        agent: Optional[Any] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List] = None,
        knowledge_base: Optional[Any] = None
    ):
        """
        初始化
        
        Args:
            agent: ReAct Agent 实例（作为备用）
            llm: LLM 实例（用于创建临时 Agent）
            tools: 工具列表（作为备用）
            knowledge_base: 知识库实例（作为备用）
        """
        self.agent = agent
        self.llm = llm
        self.tools = tools or []
        self.knowledge_base = knowledge_base
    
    def handle(self, intent: Intent) -> str:
        """
        处理任务意图
        
        核心流程：
        1. 根据 intent.selected_tools 创建带精选工具的临时 Agent
        2. 调用 Agent.run() 进入完整的 ReAct 主循环
        3. Agent 进行多轮 Think-Act-Observe 推理
        4. 返回最终结果
        
        Args:
            intent: 意图识别结果
            
        Returns:
            Agent 处理结果
        """
        query = intent.query
        use_rag = intent.metadata.get("use_rag", True)
        
        try:
            # 获取或创建 Agent
            agent = self._get_or_create_agent(intent)
            
            if agent is None:
                return "Agent 未初始化，无法处理任务。请确保 LLM 已配置。"
            
            # 调用 ReAct Agent 的 run 方法（完整的主循环）
            print(f"🚀 进入 ReAct 主循环，意图类型: {intent.type}")
            result = agent.run(query, use_rag=use_rag)
            return result
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"任务处理失败: {str(e)}"
    
    def _get_or_create_agent(self, intent: Intent) -> Optional[Any]:
        """
        获取或创建 Agent 实例
        
        优先级：
        1. 使用精选工具创建临时 Agent（推荐）
        2. 使用备用 Agent
        3. 使用 LLM + 备用工具创建 Agent
        
        Args:
            intent: 意图
            
        Returns:
            Agent 实例
        """
        from ....core import Agent
        
        # 1. 如果有精选工具，创建带精选工具的临时 Agent
        if intent.selected_tools and self.llm:
            knowledge_base = intent.metadata.get("knowledge_base", self.knowledge_base)
            
            temp_agent = Agent(
                llm=self.llm,
                tools=intent.selected_tools,
                knowledge_base=knowledge_base,
                use_rag=intent.metadata.get("use_rag", True)
            )
            
            tool_names = intent.get_tool_names()
            print(f"🤖 创建临时 Agent，精选工具({len(tool_names)}个): {tool_names[:5]}{'...' if len(tool_names) > 5 else ''}")
            
            return temp_agent
        
        # 2. 使用备用 Agent
        if self.agent is not None:
            print("🤖 使用备用 Agent")
            return self.agent
        
        # 3. 使用 LLM + 备用工具创建 Agent
        if self.llm and self.tools:
            print(f"🤖 使用备用工具创建 Agent ({len(self.tools)}个工具)")
            return Agent(
                llm=self.llm,
                tools=self.tools,
                knowledge_base=self.knowledge_base,
                use_rag=True
            )
        
        return None
    
    def can_handle(self, intent: Intent) -> bool:
        """检查是否可以处理"""
        # 有以下任一条件即可处理：
        # 1. 有备用 Agent
        # 2. 有 LLM + 精选工具
        # 3. 有 LLM + 备用工具
        if self.agent is not None:
            return True
        if self.llm is not None:
            if intent.selected_tools:
                return True
            if self.tools:
                return True
        return False
