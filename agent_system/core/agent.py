"""
Agent 核心类

实现 ReAct (Reasoning and Acting) 模式的 Agent
这是整个系统的核心
"""

import json
from typing import List, Optional, Tuple

from langchain.memory import ConversationBufferMemory
from langchain.tools.render import render_text_description
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from ..config.settings import MAX_THOUGHT_STEPS
from ..models.action import Action
from ..prompts.agent_prompts import get_main_prompt
from ..prompts.final_prompts import get_final_prompt
from .callbacks import MyPrintHandler


class Agent:
    """
    自定义 Agent 类
    
    核心流程（ReAct 循环）：
    1. 接收任务
    2. LLM 思考(Reasoning) → 决定下一步行动
    3. 执行工具(Acting) → 获取观察结果
    4. 更新记忆 → 记录思考和行动历史
    5. 回到步骤2，继续循环
    6. 直到任务完成（调用 FINISH 工具）
    
    使用方式:
        >>> from langchain_openai import ChatOpenAI
        >>> llm = ChatOpenAI(model="deepseek-chat")
        >>> agent = Agent(llm=llm, tools=[...])
        >>> result = agent.run("帮我查询火车票")
    """
    
    def __init__(
            self,
            llm: BaseChatModel,
            tools: List[BaseTool],
            max_thought_steps: Optional[int] = MAX_THOUGHT_STEPS,
    ):
        """
        初始化 Agent
        
        参数:
            llm: 大语言模型实例
            tools: 工具列表
            max_thought_steps: 最大思考步数，防止无限循环
        """
        # 核心组件
        self.llm = llm  # 大语言模型
        self.tools = tools  # 可用工具列表
        self.max_thought_steps = max_thought_steps  # 最多思考步数
        
        # 提示词模板
        self.main_prompt = get_main_prompt()  # 主循环提示词
        self.final_prompt = get_final_prompt()  # 最终总结提示词
        
        # 输出解析器：将 LLM 的文本输出解析为结构化的 Action 对象
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        
        # 初始化主循环的提示词（会注入工具描述和格式说明）
        self.prompt = self._init_prompt()
        
        # LCEL (LangChain Expression Language) 链
        # 构建处理流程：Prompt → LLM → 字符串解析
        self.llm_chain = self.prompt | self.llm | StrOutputParser()
        
        # 打印处理器：用于实时显示 LLM 的思考过程
        self.verbose_printer = MyPrintHandler()
    
    def _init_prompt(self):
        """
        初始化提示词模板
        
        将工具描述和输出格式说明注入到提示词中
        使用 partial 方法预填充部分变量
        
        返回:
            填充了工具信息和格式说明的提示词模板
        """
        return self.main_prompt.partial(
            # 渲染工具列表的描述（工具名称、参数、说明等）
            tools=render_text_description(self.tools),
            # 获取输出格式说明（告诉 LLM 如何输出 JSON 格式的 Action）
            format_instructions=self._chinese_friendly(
                self.output_parser.get_format_instructions(),
            )
        )
    
    def run(self, task_description: str) -> str:
        """
        Agent 主流程 - ReAct 循环的核心
        
        这是整个 Agent 的入口方法
        实现了完整的 Think-Act-Observe 循环
        
        参数:
            task_description: 任务描述（如"帮我查询24年6月1日早上去上海的火车票"）
            
        返回:
            最终答案字符串
            
        流程:
            1. 初始化记忆系统
            2. 进入 ReAct 循环：
               - Think: LLM 思考下一步行动
               - Act: 执行选择的工具
               - Observe: 获取工具执行结果
               - 更新记忆，进入下一轮
            3. 任务完成后，总结并返回最终答案
        """
        
        # 思考步数计数器
        thought_step_count = 0
        
        # 初始化记忆系统
        # ConversationBufferMemory 会保存 Agent 的所有思考和行动历史
        agent_memory = ConversationBufferMemory(
            return_messages=True,  # 以消息列表格式返回历史
        )
        # 初始化记忆，标记任务开始
        agent_memory.save_context(
            {"input": "\ninit"},
            {"output": "\n开始"}
        )
        
        # 开始 ReAct 循环
        # 每一轮循环代表一次"思考-行动"过程
        while thought_step_count < self.max_thought_steps:
            print(f">>>>Round: {thought_step_count}<<<<")
            
            # 【Think】步骤：让 LLM 思考并决定下一步行动
            action, response = self._step(
                task_description=task_description,
                memory=agent_memory
            )
            
            # 检查是否完成任务
            # 当 Agent 认为任务完成时，会选择 FINISH 工具
            if action.name == "FINISH":
                break
            
            # 【Act】步骤：执行 LLM 选择的工具
            observation = self._exec_action(action)
            print(f"----\nObservation:\n{observation}")
            
            # 【Update Memory】步骤：更新记忆
            # 将这一轮的思考(response)和观察结果(observation)存入记忆
            self._update_memory(agent_memory, response, observation)
            
            # 增加步数计数
            thought_step_count += 1
        
        # 循环结束，生成最终答案
        if thought_step_count >= self.max_thought_steps:
            # 如果达到最大步数限制，说明任务可能太复杂或陷入死循环
            reply = "抱歉，我没能完成您的任务。"
        else:
            # 任务正常完成，让 LLM 总结整个过程并给出最终答案
            final_chain = self.final_prompt | self.llm | StrOutputParser()
            reply = final_chain.invoke({
                "task_description": task_description,
                "memory": agent_memory  # 传入完整的思考和行动历史
            })
        
        return reply
    
    def _step(self, task_description: str, memory) -> Tuple[Action, str]:
        """
        执行一步思考（ReAct 循环中的 Think 步骤）
        
        这是 Agent 的"大脑"，负责：
        1. 将任务描述和历史记忆传递给 LLM
        2. LLM 进行推理，决定下一步行动
        3. 解析 LLM 的输出，提取出要执行的工具和参数
        
        参数:
            task_description: 原始任务描述
            memory: 历史记忆（包含之前的所有思考和行动）
            
        返回:
            action: 要执行的动作（工具名称+参数）
            response: LLM 的完整响应文本
        """
        response = ""
        
        # 使用流式调用 LLM
        # stream() 会逐个 token 返回结果，实现实时输出
        for s in self.llm_chain.stream({
            "task_description": task_description,  # 任务描述
            "memory": memory  # 历史记忆
        }, config={
            "callbacks": [
                self.verbose_printer  # 回调处理器，实时打印输出
            ]
        }):
            response += s  # 累积 LLM 的输出
        
        # 将 LLM 的文本输出解析为结构化的 Action 对象
        # 例如：{"name": "查询火车票", "args": {...}} → Action(name="查询火车票", args={...})
        action = self.output_parser.parse(response)
        return action, response
    
    def _exec_action(self, action: Action) -> str:
        """
        执行动作（ReAct 循环中的 Act 步骤）
        
        根据 LLM 决定的动作，调用对应的工具
        
        参数:
            action: LLM 决定的动作（包含工具名称和参数）
            
        返回:
            observation: 工具执行的结果（字符串格式）
        """
        observation = "没有找到工具"
        
        # 遍历所有可用工具，找到匹配的工具
        for tool in self.tools:
            if tool.name == action.name:
                try:
                    # 执行工具，传入 LLM 提供的参数
                    observation = tool.run(action.args)
                except ValidationError as e:
                    # 参数验证失败（类型不匹配、缺少必需参数等）
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    # 工具执行过程中的其他异常
                    observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
        
        return observation
    
    @staticmethod
    def _update_memory(agent_memory, response: str, observation: str):
        """
        更新记忆（ReAct 循环中的 Observe 步骤）
        
        将本轮的思考过程和观察结果保存到记忆中
        这些信息会在下一轮思考时提供给 LLM
        
        参数:
            agent_memory: 记忆对象
            response: LLM 的思考过程（文本）
            observation: 工具执行结果（观察）
        """
        agent_memory.save_context(
            {"input": response},  # 输入：LLM 的思考和决策
            {"output": "\n返回结果:\n" + str(observation)}  # 输出：工具的执行结果
        )
    
    @staticmethod
    def _chinese_friendly(string: str) -> str:
        """
        将输出格式说明转换为中文友好的格式
        
        Pydantic 的输出格式说明默认是英文的
        这个方法会将 JSON 示例转换为支持中文的格式（ensure_ascii=False）
        
        参数:
            string: 原始格式说明
            
        返回:
            中文友好的格式说明
        """
        lines = string.split('\n')
        for i, line in enumerate(lines):
            # 识别 JSON 行
            if line.startswith('{') and line.endswith('}'):
                try:
                    # 重新序列化 JSON，禁用 ASCII 转义
                    lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
                except:
                    pass  # 解析失败则保持原样
        return '\n'.join(lines)


