"""
Agent 系统包

这是一个模块化的 Agent 系统实现
采用 ReAct (Reasoning and Acting) 模式
"""

__version__ = "1.0.0"
__author__ = "AgentLearning"

# 导出核心类，方便外部导入
# 使用方式: from agent_system import Agent, Action
from .core.agent import Agent
from .models.action import Action

__all__ = ["Agent", "Action"]


