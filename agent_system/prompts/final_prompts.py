"""
最终总结提示词

定义任务完成后，Agent 总结结果的提示词模板
"""

from langchain_core.prompts import PromptTemplate


# 最终总结提示词模板
# 任务完成后，让 Agent 总结整个过程并给出最终答案
FINAL_PROMPT_TEMPLATE = """
你的任务是:
{task_description}

以下是你的思考过程和使用工具与外部资源交互的结果。
{memory}

你已经完成任务。
现在请根据上述结果简要总结出你的最终答案。
直接给出答案。不用再解释或分析你的思考过程。
"""


def get_final_prompt() -> PromptTemplate:
    """
    获取最终总结提示词模板
    
    返回:
        PromptTemplate: LangChain 提示词模板对象
    
    使用方式:
        >>> prompt = get_final_prompt()
        >>> formatted = prompt.format(
        ...     task_description="查询火车票",
        ...     memory="..."
        ... )
    """
    return PromptTemplate.from_template(FINAL_PROMPT_TEMPLATE)


