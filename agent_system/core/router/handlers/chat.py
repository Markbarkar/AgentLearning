"""
闲聊处理器

处理简单的闲聊对话，直接使用 LLM 回复
不需要工具调用或知识检索
"""

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from .base import BaseHandler
from ..registry import register_handler
from ...intent.models import Intent


# 闲聊提示词模板
CHAT_PROMPT = """你是一个友好的AI助手。请自然地回应用户的对话。

用户: {query}

请给出友好、自然的回复:"""


@register_handler("chat", priority=100, description="处理闲聊对话")
class ChatHandler(BaseHandler):
    """
    闲聊处理器
    
    直接使用 LLM 回复用户的闲聊内容
    无需工具调用或知识检索
    """
    
    def __init__(self, llm: BaseChatModel):
        """
        初始化
        
        Args:
            llm: 语言模型实例（由工厂自动注入）
        """
        self.llm = llm
        self._prompt = PromptTemplate.from_template(CHAT_PROMPT)
    
    def handle(self, intent: Intent) -> str:
        """
        处理闲聊意图
        
        Args:
            intent: 意图识别结果
            
        Returns:
            LLM 生成的回复
        """
        try:
            # 直接调用 LLM
            response = self.llm.invoke(self._prompt.format(query=intent.query))
            
            # 提取内容
            if hasattr(response, 'content'):
                return response.content
            return str(response)
            
        except Exception as e:
            return f"抱歉，我现在无法正常回复。错误: {str(e)}"
    
    def can_handle(self, intent: Intent) -> bool:
        """检查是否可以处理"""
        return self.llm is not None
