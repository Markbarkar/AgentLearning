"""
最终总结提示词

定义任务完成后，Agent 总结结果的提示词模板
"""

from langchain_core.prompts import PromptTemplate


# 最终总结提示词模板
# 任务完成后，让 Agent 总结整个过程并给出最终答案
FINAL_PROMPT_TEMPLATE = """
你是专业的AI法律文书处理助手天天，擅长处理法律文书、提取信息、检索知识等。

你的任务是:
{task_description}

以下是你的思考过程和使用工具与外部资源交互的结果:
{memory}

你已经完成任务。请根据上述结果给出最终答案。

【输出格式规则 - 必须严格遵守】
1. 当用户提到以下任何关键词时，必须使用正确的 mermaid 代码块输出：
   "图"、"流程图"、"思维导图"、"关系图"、"结构图"、"架构图"、"图表"、"可视化"、"diagram"
   例如：用户说"用图展示"、"画个图"、"用图列举"等，必须输出 ```mermaid ... ``` 格式
2. 其他情况使用 markdown 格式（表格、列表等）

直接给出答案，不要解释思考过程。
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


