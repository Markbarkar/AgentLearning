"""
Agent 学习项目 - 火车票助手
==========================

这是一个基于 ReAct (Reasoning and Acting) 模式的 Agent 实现
Agent 会：
1. 接收任务描述
2. 通过思考(Reasoning)决定下一步行动
3. 执行工具(Acting)获取结果
4. 根据结果继续思考，形成循环
5. 直到任务完成

核心概念：
- LLM: 大语言模型，负责推理和决策
- Tools: Agent 可以调用的工具/函数
- Memory: 记录 Agent 的思考过程和执行历史
- Prompt: 指导 Agent 如何思考和行动的模板
"""

import json
import sys
from typing import List, Optional, Dict, Any, Tuple, Union
from uuid import UUID
from dotenv import load_dotenv

# 加载.env文件中的环境变量（包含API密钥）
load_dotenv()

# LangChain 核心组件导入
from langchain.memory import ConversationBufferMemory  # 对话记忆管理
from langchain.tools.render import render_text_description  # 工具描述渲染
from langchain_core.callbacks import BaseCallbackHandler  # 回调处理器基类
from langchain_core.language_models import BaseChatModel  # 聊天模型基类
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser  # 输出解析器
from langchain_core.outputs import GenerationChunk, ChatGenerationChunk, LLMResult  # LLM输出类型
from langchain_core.prompts import PromptTemplate  # 提示词模板
from langchain_core.tools import StructuredTool  # 结构化工具
from langchain_openai import ChatOpenAI  # OpenAI API 兼容的聊天模型

# Pydantic 用于数据验证和类型提示
from pydantic import BaseModel, Field, ValidationError

"""
=====================================
第一部分：工具(Tools)定义
=====================================
Tools 是 Agent 可以调用的外部功能
每个工具都是一个 Python 函数，Agent 通过 LLM 决定何时调用哪个工具
"""

def search_train_ticket(
        origin: str,
        destination: str,
        date: str,
        departure_time_start: str,
        departure_time_end: str
) -> List[dict[str, str]]:
    """
    工具1: 查询火车票
    
    功能：根据出发地、目的地、日期和时间范围查询可用的火车票
    参数：
        - origin: 出发地
        - destination: 目的地  
        - date: 出发日期
        - departure_time_start: 出发时间范围开始
        - departure_time_end: 出发时间范围结束
    返回：火车票列表（这里使用 mock 数据模拟）
    """

    # 这里使用 mock 数据模拟数据库查询
    # 在实际应用中，这里应该调用真实的火车票查询 API
    return [
        {
            "train_number": "G1234",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 8:00",
            "arrival_time": "2024-06-01 12:00",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G5678",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 18:30",
            "arrival_time": "2024-06-01 22:30",
            "price": "100.00",
            "seat_type": "商务座",
        },
        {
            "train_number": "G9012",
            "origin": "北京",
            "destination": "上海",
            "departure_time": "2024-06-01 19:00",
            "arrival_time": "2024-06-01 23:00",
            "price": "100.00",
            "seat_type": "商务座",
        }
    ]


def purchase_train_ticket(
        train_number: str,
) -> dict:
    """
    工具2: 购买火车票
    
    功能：根据车次号购买火车票
    参数：
        - train_number: 车次号
    返回：购买结果（包括座位号等信息）
    """
    # 模拟购买操作，返回购买成功的信息
    return {
        "result": "success",
        "message": "购买成功",
        "data": {
            "train_number": "G1234",
            "seat_type": "商务座",
            "seat_number": "7-17A"
        }
    }


# 将 Python 函数包装成 LangChain 的 StructuredTool
# StructuredTool 会自动：
# 1. 从函数签名提取参数信息
# 2. 生成参数的 JSON Schema
# 3. 验证 Agent 传入的参数是否合法

search_train_ticket_tool = StructuredTool.from_function(
    func=search_train_ticket,  # 实际执行的函数
    name="查询火车票",  # 工具名称，Agent 会看到这个名称
    description="查询指定日期可用的火车票。",  # 工具描述，帮助 Agent 理解何时使用这个工具
)

purchase_train_ticket_tool = StructuredTool.from_function(
    func=purchase_train_ticket,
    name="购买火车票",
    description="购买火车票。会返回购买结果(result), 和座位号(seat_number)",
)

# FINISH 工具：一个特殊的占位符工具
# 当 Agent 认为任务已完成时，会调用这个工具
finish_placeholder = StructuredTool.from_function(
    func=lambda: None,  # 不执行任何操作
    name="FINISH",
    description="用于表示任务完成的占位符工具"
)

# 工具列表：将所有工具组合在一起，提供给 Agent 使用
tools = [search_train_ticket_tool, purchase_train_ticket_tool, finish_placeholder]

"""
=====================================
第二部分：Prompt 提示词模板
=====================================
Prompt 是 Agent 的"大脑指令"，告诉 LLM：
1. 你是谁（角色设定）
2. 你能做什么（可用工具）
3. 你应该如何思考（推理格式）
4. 你应该如何输出（输出格式）
"""

# 主循环 Prompt：指导 Agent 每一步的思考和行动
# 这是 ReAct 模式的核心：让 Agent 在"思考"和"行动"之间循环
prompt_text = """
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

# 最终总结 Prompt：任务完成后，让 Agent 总结结果
# 这一步会整合所有的思考过程和工具执行结果，给出最终答案
final_prompt = """
你的任务是:
{task_description}

以下是你的思考过程和使用工具与外部资源交互的结果。
{memory}

你已经完成任务。
现在请根据上述结果简要总结出你的最终答案。
直接给出答案。不用再解释或分析你的思考过程。
"""

"""
=====================================
第三部分：数据模型和回调处理
=====================================
"""

class Action(BaseModel):
    """
    Action 数据模型
    
    定义 Agent 每一步要执行的动作
    使用 Pydantic 进行数据验证，确保 LLM 输出的格式正确
    
    属性：
        - name: 要调用的工具名称
        - args: 工具的参数（字典格式）
    
    示例：
    {
        "name": "查询火车票",
        "args": {
            "origin": "北京",
            "destination": "上海",
            "date": "2024-06-01",
            "departure_time_start": "08:00",
            "departure_time_end": "12:00"
        }
    }
    """
    name: str = Field(description="工具或指令名称")
    args: Optional[Dict[str, Any]] = Field(description="工具或指令参数，由参数名称和参数值组成")


class MyPrintHandler(BaseCallbackHandler):
    """
    自定义回调处理器
    
    作用：实时打印 LLM 的输出流
    在 Agent 思考过程中，可以看到 LLM 逐字输出的思考内容
    这对于调试和理解 Agent 的推理过程非常有帮助
    """
    def __init__(self):
        BaseCallbackHandler.__init__(self)

    def on_llm_new_token(
            self,
            token: str,  # LLM 生成的每一个 token（字符或词）
            *,
            chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        """每生成一个新 token 时调用，实现流式输出"""
        end = ""
        content = token + end
        sys.stdout.write(content)  # 立即输出到终端
        sys.stdout.flush()  # 刷新缓冲区，确保立即显示
        return token

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """LLM 输出完成时调用，添加换行符"""
        end = ""
        content = "\n" + end
        sys.stdout.write(content)
        sys.stdout.flush()
        return response

"""
=====================================
第四部分：Agent 核心类
=====================================
这是整个 Agent 的核心实现
采用 ReAct (Reasoning and Acting) 模式
"""

class MyAgent:
    """
    自定义 Agent 类
    
    核心流程（ReAct 循环）：
    1. 接收任务
    2. LLM 思考(Reasoning) → 决定下一步行动
    3. 执行工具(Acting) → 获取观察结果
    4. 更新记忆 → 记录思考和行动历史
    5. 回到步骤2，继续循环
    6. 直到任务完成（调用 FINISH 工具）
    """
    
    def __init__(
            self,
            llm: BaseChatModel = ChatOpenAI(
                model="deepseek-chat",  # 使用 DeepSeek-V3 模型
                temperature=0,  # 温度为0，输出更确定性，适合 Agent 任务
                base_url="https://api.deepseek.com",  # DeepSeek API 地址
                model_kwargs={
                    "seed": 42  # 固定随机种子，确保结果可复现
                },
            ),
            tools=None,  # Agent 可使用的工具列表
            prompt: str = "",  # 主循环的提示词模板
            final_prompt: str = "",  # 最终总结的提示词模板
            max_thought_steps: Optional[int] = 10,  # 最大思考步数，防止无限循环
    ):
        """
        初始化 Agent
        
        参数：
            llm: 大语言模型实例
            tools: 工具列表
            prompt: 主循环提示词
            final_prompt: 最终总结提示词
            max_thought_steps: 最大思考步数
        """
        if tools is None:
            tools = []
            
        # 核心组件
        self.llm = llm  # 大语言模型
        self.tools = tools  # 可用工具列表
        self.final_prompt = PromptTemplate.from_template(final_prompt)  # 最终总结模板
        self.max_thought_steps = max_thought_steps  # 最多思考步数，避免死循环
        
        # 输出解析器：将 LLM 的文本输出解析为结构化的 Action 对象
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        
        # 初始化主循环的提示词（会注入工具描述和格式说明）
        self.prompt = self.__init_prompt(prompt)
        
        # LCEL (LangChain Expression Language) 链
        # 构建处理流程：Prompt → LLM → 字符串解析
        # 使用 | 运算符串联各个组件
        self.llm_chain = self.prompt | self.llm | StrOutputParser()
        
        # 打印处理器：用于实时显示 LLM 的思考过程
        self.verbose_printer = MyPrintHandler()

    def __init_prompt(self, prompt):
        """
        初始化提示词模板
        
        将工具描述和输出格式说明注入到提示词中
        使用 partial 方法预填充部分变量
        
        返回：填充了工具信息和格式说明的提示词模板
        """
        return PromptTemplate.from_template(prompt).partial(
            # 渲染工具列表的描述（工具名称、参数、说明等）
            tools=render_text_description(self.tools),
            # 获取输出格式说明（告诉 LLM 如何输出 JSON 格式的 Action）
            format_instructions=self.__chinese_friendly(
                self.output_parser.get_format_instructions(),
            )
        )

    def run(self, task_description):
        """
        Agent 主流程 - ReAct 循环的核心
        
        这是整个 Agent 的入口方法
        实现了完整的 Think-Act-Observe 循环
        
        参数：
            task_description: 任务描述（如"帮我买24年6月1日早上去上海的火车票"）
            
        返回：
            最终答案字符串
            
        流程：
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
        # 这些历史会在每次思考时传递给 LLM，帮助 LLM 做出更好的决策
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
            action, response = self.__step(
                task_description=task_description,
                memory=agent_memory
            )

            # 检查是否完成任务
            # 当 Agent 认为任务完成时，会选择 FINISH 工具
            if action.name == "FINISH":
                break

            # 【Act】步骤：执行 LLM 选择的工具
            observation = self.__exec_action(action)
            print(f"----\nObservation:\n{observation}")

            # 【Update Memory】步骤：更新记忆
            # 将这一轮的思考(response)和观察结果(observation)存入记忆
            self.__update_memory(agent_memory, response, observation)

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

    def __step(self, task_description, memory) -> Tuple[Action, str]:
        """
        执行一步思考（ReAct 循环中的 Think 步骤）
        
        这是 Agent 的"大脑"，负责：
        1. 将任务描述和历史记忆传递给 LLM
        2. LLM 进行推理，决定下一步行动
        3. 解析 LLM 的输出，提取出要执行的工具和参数
        
        参数：
            task_description: 原始任务描述
            memory: 历史记忆（包含之前的所有思考和行动）
            
        返回：
            action: 要执行的动作（工具名称+参数）
            response: LLM 的完整响应文本
        """
        print("开始调用LLM...")
        response = ""
        try:
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
        except Exception as e:
            print(f"LLM调用出错: {e}")
            raise

        print("\n解析响应...")
        # 将 LLM 的文本输出解析为结构化的 Action 对象
        # 例如：{"name": "查询火车票", "args": {...}} → Action(name="查询火车票", args={...})
        action = self.output_parser.parse(response)
        return action, response

    def __exec_action(self, action: Action) -> str:
        """
        执行动作（ReAct 循环中的 Act 步骤）
        
        根据 LLM 决定的动作，调用对应的工具
        
        参数：
            action: LLM 决定的动作（包含工具名称和参数）
            
        返回：
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
    def __update_memory(agent_memory, response, observation):
        """
        更新记忆（ReAct 循环中的 Observe 步骤）
        
        将本轮的思考过程和观察结果保存到记忆中
        这些信息会在下一轮思考时提供给 LLM
        
        参数：
            agent_memory: 记忆对象
            response: LLM 的思考过程（文本）
            observation: 工具执行结果（观察）
        """
        agent_memory.save_context(
            {"input": response},  # 输入：LLM 的思考和决策
            {"output": "\n返回结果:\n" + str(observation)}  # 输出：工具的执行结果
        )

    @staticmethod
    def __chinese_friendly(string) -> str:
        """
        将输出格式说明转换为中文友好的格式
        
        Pydantic 的输出格式说明默认是英文的
        这个方法会将 JSON 示例转换为支持中文的格式（ensure_ascii=False）
        
        参数：
            string: 原始格式说明
            
        返回：
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

"""
=====================================
第五部分：主程序入口
=====================================
演示如何使用 Agent
"""

if __name__ == "__main__":
    # 创建 Agent 实例
    # 传入：工具列表、主循环提示词、最终总结提示词
    my_agent = MyAgent(
        tools=tools,  # 可用的工具列表
        prompt=prompt_text,  # 主循环提示词模板
        final_prompt=final_prompt,  # 最终总结提示词模板
    )

    # 定义任务
    # Agent 会自动：
    # 1. 理解任务需求
    # 2. 决定先查询火车票
    # 3. 根据查询结果选择合适的车次
    # 4. 调用购买工具
    # 5. 完成任务并返回结果
    task = "帮我查询24年6月1日早上去上海的火车票"
    
    # 运行 Agent
    reply = my_agent.run(task)
    
    # 打印最终答案
    print(reply)

"""
========================================
学习要点总结
========================================

1. **ReAct 模式**
   - Reasoning (思考): LLM 分析当前状态，决定下一步
   - Acting (行动): 执行选择的工具
   - Observing (观察): 获取工具结果，更新知识

2. **关键组件**
   - LLM: 提供推理能力
   - Tools: 扩展 Agent 的能力边界
   - Memory: 维护上下文，支持多步推理
   - Prompt: 引导 Agent 的行为

3. **工作流程**
   任务 → Think → Act → Observe → 更新记忆 → Think → ... → 完成

4. **扩展方向**
   - 添加更多工具（数据库查询、API 调用等）
   - 改进 Prompt 提升推理质量
   - 引入更复杂的记忆机制
   - 添加错误重试逻辑
   - 实现多 Agent 协作

5. **实际应用**
   - 客服机器人
   - 自动化运维
   - 数据分析助手
   - 代码生成和调试
   - 知识问答系统
"""