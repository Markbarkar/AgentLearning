"""
通用工具

包含所有 Agent 通用的工具
"""

from langchain_core.tools import StructuredTool


def finish_action() -> None:
    """
    完成任务的占位符工具
    
    当 Agent 认为任务已完成时，会调用这个工具
    这个工具不执行任何实际操作，只是一个标记
    """
    return None


# FINISH 工具：特殊的占位符工具
# 当 Agent 认为任务已完成时，会选择调用这个工具
finish_tool = StructuredTool.from_function(
    func=finish_action,
    name="FINISH",
    description="用于表示任务完成的占位符工具。当你认为任务已经完成时，调用此工具。"
)


