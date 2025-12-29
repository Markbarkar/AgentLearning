"""
RAG 检索工具

提供给 Agent 使用的知识库检索工具
"""

from typing import Optional
from langchain_core.tools import Tool


def create_rag_search_tool(knowledge_base) -> Tool:
    """
    创建 RAG 检索工具
    
    Args:
        knowledge_base: 知识库实例
        
    Returns:
        LangChain Tool 对象
    """
    
    def search_knowledge_base(query: str) -> str:
        """
        检索知识库
        
        Args:
            query: 查询文本
            
        Returns:
            格式化的检索结果
        """
        try:
            # 检索相关文档
            results = knowledge_base.search(
                query=query,
                top_k=3,
                with_score=True
            )
            
            if not results:
                return "未找到相关信息"
            
            # 格式化输出
            output = f"找到 {len(results)} 条相关信息：\n\n"
            
            for i, result in enumerate(results, 1):
                content = result['content']
                metadata = result['metadata']
                similarity = result.get('similarity', 0)
                
                output += f"【结果 {i}】（相似度: {similarity:.2f}）\n"
                output += f"来源: {metadata.get('file_name', '未知')}\n"
                output += f"内容: {content[:300]}...\n\n"
            
            return output
        
        except Exception as e:
            return f"检索失败: {str(e)}"
    
    # 创建工具
    tool = Tool(
        name="search_knowledge_base",
        description="""用于回想二次检索法律知识库，查找相关的法律文书和知识，用于辅助决策。
        输入参数：
        - query: 查询文本（必填），描述你要查找的信息

        返回：相关的法律文书片段和知识

        使用场景：
        - 需要查找相关法律案例或文书时
        - 需要了解特定法律概念或条款时
        - 需要参考历史案件信息时

        示例输入：合同纠纷的处理流程""",
                func=search_knowledge_base
            )
    
    return tool



