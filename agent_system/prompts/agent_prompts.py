"""
Agent 主循环提示词

定义 Agent 在 ReAct 循环中使用的提示词模板
这个提示词指导 Agent 如何思考和行动
"""

from langchain_core.prompts import PromptTemplate


# 主循环提示词模板
# 这是 ReAct 模式的核心：让 Agent 在"思考"和"行动"之间循环
MAIN_PROMPT_TEMPLATE = """
你是强大的AI火车票助手，可以使用工具与指令查询并购买火车票

你的任务是:
{task_description}

你可以使用以下工具或指令，它们又称为动作或actions:
{tools}

当前的任务执行记录:
{memory}

按照以下格式输出：

任务：你收到的需要执行的任务
思考: 观察你的任务和执行记录，并思考你下一步应该采取的行动
然后，根据以下格式说明，输出你选择执行的动作/工具:
{format_instructions}
"""


def get_main_prompt() -> PromptTemplate:
    """
    获取主循环提示词模板
    
    返回:
        PromptTemplate: LangChain 提示词模板对象
    
    使用方式:
        >>> prompt = get_main_prompt()
        >>> formatted = prompt.format(
        ...     task_description="查询火车票",
        ...     tools="...",
        ...     memory="...",
        ...     format_instructions="..."
        ... )
    """
    return PromptTemplate.from_template(MAIN_PROMPT_TEMPLATE)


