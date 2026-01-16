"""
知识问答处理器

使用 RAG 检索相关知识后回答问题
适用于需要查询知识库的问题
"""

from typing import Optional, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from .base import BaseHandler
from ..registry import register_handler
from ...intent.models import Intent


# 知识问答提示词模板
KNOWLEDGE_QA_PROMPT = """你是一个专业的AI助手。请根据以下参考资料回答用户的问题。

参考资料:
{context}

用户问题: {query}

请根据参考资料给出准确、专业的回答。如果参考资料中没有相关信息，请坦诚告知用户。

回答:"""

# 无知识时的提示词
NO_KNOWLEDGE_PROMPT = """你是一个专业的AI助手。用户提出了一个问题，但知识库中没有找到相关信息。

用户问题: {query}

请根据你的知识尽可能回答，并告知用户这是基于通用知识而非特定知识库的回答。

回答:"""


@register_handler("knowledge_qa", priority=80, description="处理知识问答")
class KnowledgeQAHandler(BaseHandler):
    """
    知识问答处理器
    
    使用 RAG 检索知识库后回答问题
    """
    
    def __init__(
        self, 
        llm: BaseChatModel,
        knowledge_base: Optional[Any] = None
    ):
        """
        初始化
        
        Args:
            llm: 语言模型实例
            knowledge_base: 知识库实例（可选）
        """
        self.llm = llm
        self.knowledge_base = knowledge_base
        self._qa_prompt = PromptTemplate.from_template(KNOWLEDGE_QA_PROMPT)
        self._no_knowledge_prompt = PromptTemplate.from_template(NO_KNOWLEDGE_PROMPT)
    
    def handle(self, intent: Intent) -> str:
        """
        处理知识问答意图
        
        流程：
        1. 从知识库检索相关内容
        2. 将检索结果作为上下文
        3. 使用 LLM 生成回答
        
        Args:
            intent: 意图识别结果
            
        Returns:
            基于知识库的回答
        """
        query = intent.query
        
        # 优先使用 metadata 中的知识库
        kb = intent.metadata.get("knowledge_base", self.knowledge_base)
        
        # 检索知识
        context = self._retrieve_knowledge(query, knowledge_base=kb)
        
        try:
            if context:
                # 有相关知识，使用知识回答
                prompt = self._qa_prompt.format(
                    context=context,
                    query=query
                )
            else:
                # 无相关知识，使用通用知识回答
                prompt = self._no_knowledge_prompt.format(query=query)
            
            response = self.llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                return response.content
            return str(response)
            
        except Exception as e:
            return f"知识问答处理失败: {str(e)}"
    
    def _retrieve_knowledge(self, query: str, top_k: int = 3, knowledge_base=None) -> str:
        """
        从知识库检索相关内容
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数
            knowledge_base: 知识库实例（可选，默认使用 self.knowledge_base）
            
        Returns:
            格式化的知识文本
        """
        kb = knowledge_base or self.knowledge_base
        if not kb:
            return ""
        
        try:
            # 调用知识库搜索
            results = kb.search(
                query=query,
                top_k=top_k,
                with_score=True
            )
            
            if not results:
                return ""
            
            # 格式化结果
            knowledge_parts = []
            for i, result in enumerate(results, 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                source = metadata.get('file_name', '未知来源')
                
                knowledge_parts.append(
                    f"[参考资料 {i}] 来源: {source}\n{content}"
                )
            
            return "\n\n".join(knowledge_parts)
            
        except Exception as e:
            print(f"知识检索失败: {e}")
            return ""
    
    def can_handle(self, intent: Intent) -> bool:
        """检查是否可以处理"""
        # 即使没有知识库也可以处理（使用通用知识）
        return self.llm is not None
