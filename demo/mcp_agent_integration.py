"""
MCP 服务器与 Agent 集成示例
==============================

展示如何将 MCP 服务器的工具集成到 Agent 中
这是一个完整的端到端示例

核心思想：
1. MCP 服务器提供标准化的工具接口
2. Agent 通过适配器调用 MCP 工具
3. 实现工具的热插拔和复用
"""

import sys
import io
import json
from typing import Dict, Any, List
from dotenv import load_dotenv

# 设置标准输出编码为 UTF-8，解决 Windows 控制台乱码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 加载环境变量
load_dotenv()

from langchain.memory import ConversationBufferMemory
from langchain.tools.render import render_text_description
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 导入 MCP 服务器
from mcp_server_demo import create_demo_mcp_server, MCPServer


# ==========================================
# MCP 工具适配器
# ==========================================

class MCPToolAdapter:
    """
    MCP 工具适配器
    
    作用：将 MCP 服务器的工具转换为 LangChain 工具
    这是连接 MCP 服务器和 Agent 的桥梁
    """
    
    def __init__(self, mcp_server: MCPServer):
        """
        初始化适配器
        
        参数：
            mcp_server: MCP 服务器实例
        """
        self.mcp_server = mcp_server
        print(f"🔌 MCP 适配器已连接到服务器: {mcp_server.name}")
    
    def create_langchain_tool(self, tool_info: Dict[str, Any]) -> StructuredTool:
        """
        将 MCP 工具转换为 LangChain 工具
        
        参数：
            tool_info: MCP 工具信息
            
        返回：
            LangChain StructuredTool 实例
        """
        tool_name = tool_info["name"]
        
        # 创建工具函数
        def tool_function(**kwargs) -> str:
            """调用 MCP 服务器的工具"""
            result = self.mcp_server.call_tool(tool_name, kwargs)
            
            if result["success"]:
                # 成功：返回结果的 JSON 字符串
                return json.dumps(result["result"], ensure_ascii=False)
            else:
                # 失败：返回错误信息
                return f"错误: {result.get('error', '未知错误')}"
        
        # 从 parameters 提取参数定义
        # 这里简化处理，实际应该完整转换 JSON Schema
        parameters = tool_info.get("parameters", {})
        properties = parameters.get("properties", {})
        
        # 构建函数参数（用于 LangChain 的类型提示）
        # 注意：这里简化了类型转换
        tool_function.__annotations__ = {
            param_name: str  # 简化处理，都用 str
            for param_name in properties.keys()
        }
        
        # 创建 LangChain 工具
        langchain_tool = StructuredTool.from_function(
            func=tool_function,
            name=tool_name,
            description=tool_info["description"]
        )
        
        return langchain_tool
    
    def get_all_tools(self) -> List[StructuredTool]:
        """
        获取所有 MCP 工具的 LangChain 版本
        
        返回：
            LangChain 工具列表
        """
        mcp_tools = self.mcp_server.list_tools()
        langchain_tools = []
        
        for tool_info in mcp_tools:
            try:
                lc_tool = self.create_langchain_tool(tool_info)
                langchain_tools.append(lc_tool)
                print(f"   ✅ 工具 '{tool_info['name']}' 已转换")
            except Exception as e:
                print(f"   ❌ 工具 '{tool_info['name']}' 转换失败: {e}")
        
        return langchain_tools


# ==========================================
# Agent 定义（简化版）
# ==========================================

class Action(BaseModel):
    """动作数据模型"""
    name: str = Field(description="工具或指令名称")
    args: Dict[str, Any] = Field(description="工具或指令参数")


class MCPAgent:
    """
    支持 MCP 的 Agent
    
    这是一个简化的 Agent 实现
    展示如何使用 MCP 服务器的工具
    """
    
    def __init__(
        self,
        mcp_server: MCPServer,
        llm: BaseChatModel = None,
        max_steps: int = 10
    ):
        """
        初始化 Agent
        
        参数：
            mcp_server: MCP 服务器实例
            llm: 语言模型
            max_steps: 最大思考步数
        """
        # 默认 LLM
        if llm is None:
            llm = ChatOpenAI(
                model="deepseek-chat",
                temperature=0,
                base_url="https://api.deepseek.com"
            )
        
        self.llm = llm
        self.max_steps = max_steps
        
        # 创建 MCP 适配器
        print("\n🔄 正在转换 MCP 工具...")
        self.adapter = MCPToolAdapter(mcp_server)
        self.tools = self.adapter.get_all_tools()
        
        # 添加 FINISH 工具
        finish_tool = StructuredTool.from_function(
            func=lambda: None,
            name="FINISH",
            description="任务完成时调用此工具"
        )
        self.tools.append(finish_tool)
        
        # 初始化其他组件
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self._init_prompts()
        
        print(f"\n✅ Agent 初始化完成，共加载 {len(self.tools)} 个工具\n")
    
    def _init_prompts(self):
        """初始化提示词"""
        prompt_text = """
你是一个智能助手，可以使用多种工具来帮助用户完成任务。

你的任务是: {task_description}

可用工具:
{tools}

历史记录:
{memory}

请按照以下格式输出：

任务：[复述任务]
思考：[分析当前情况，决定下一步]
动作：{format_instructions}
"""
        
        self.prompt = PromptTemplate.from_template(prompt_text).partial(
            tools=render_text_description(self.tools),
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        self.llm_chain = self.prompt | self.llm | StrOutputParser()
    
    def run(self, task: str) -> str:
        """
        运行 Agent
        
        参数：
            task: 任务描述
            
        返回：
            最终结果
        """
        print(f"🎯 任务: {task}\n")
        print("="*60)
        
        memory = ConversationBufferMemory(return_messages=True)
        memory.save_context({"input": "init"}, {"output": "开始"})
        
        for step in range(self.max_steps):
            print(f"\n🔄 Round {step}")
            print("-"*60)
            
            # 思考
            response = self.llm_chain.invoke({
                "task_description": task,
                "memory": memory
            })
            
            print(f"💭 思考:\n{response}\n")
            
            # 解析动作
            try:
                action = self.output_parser.parse(response)
            except Exception as e:
                print(f"❌ 解析失败: {e}")
                continue
            
            # 检查是否完成
            if action.name == "FINISH":
                print("✅ 任务完成！")
                break
            
            # 执行工具
            print(f"🛠️  执行工具: {action.name}")
            print(f"📝 参数: {json.dumps(action.args, ensure_ascii=False)}")
            
            observation = self._exec_tool(action)
            print(f"👀 观察结果:\n{observation}\n")
            
            # 更新记忆
            memory.save_context(
                {"input": response},
                {"output": f"结果: {observation}"}
            )
        
        print("="*60)
        return "任务执行完毕"
    
    def _exec_tool(self, action: Action) -> str:
        """执行工具"""
        for tool in self.tools:
            if tool.name == action.name:
                try:
                    return tool.run(action.args)
                except Exception as e:
                    return f"工具执行错误: {str(e)}"
        return f"未找到工具: {action.name}"


# ==========================================
# 演示和测试
# ==========================================

def demo_simple_tasks():
    """
    演示简单任务
    
    展示 MCP Agent 如何处理各种任务
    """
    print("\n" + "="*70)
    print("MCP Agent 演示 - 简单任务")
    print("="*70 + "\n")
    
    # 创建 MCP 服务器
    mcp_server = create_demo_mcp_server()
    
    # 创建 Agent
    agent = MCPAgent(mcp_server)
    
    # 任务列表
    tasks = [
        "查询一下上海的天气",
        "计算 (25 + 75) * 4 等于多少",
        "告诉我现在几点了",
        "搜索关于 MCP 的信息"
    ]
    
    for i, task in enumerate(tasks, 1):
        print(f"\n\n{'='*70}")
        print(f"任务 {i}/{len(tasks)}")
        print('='*70)
        agent.run(task)
        print("\n" + "⏸️  " + "-"*68)
        input("按回车继续下一个任务...")


def demo_complex_task():
    """
    演示复杂任务
    
    需要多步推理和多个工具协作
    """
    print("\n" + "="*70)
    print("MCP Agent 演示 - 复杂任务")
    print("="*70 + "\n")
    
    # 创建 MCP 服务器
    mcp_server = create_demo_mcp_server()
    
    # 创建 Agent
    agent = MCPAgent(mcp_server, max_steps=15)
    
    # 复杂任务
    task = """
    帮我做以下事情：
    1. 查询北京和上海的天气
    2. 计算两个城市温度的平均值（假设北京15度，上海18度）
    3. 搜索关于 'Python' 的信息
    4. 告诉我当前时间
    """
    
    agent.run(task)


def compare_traditional_vs_mcp():
    """
    对比传统工具 vs MCP 工具
    """
    print("\n" + "="*70)
    print("传统 Agent vs MCP Agent 对比")
    print("="*70 + "\n")
    
    print("📊 对比维度:\n")
    
    comparison = [
        ("工具定义", "硬编码在代码中", "独立的 MCP 服务器"),
        ("工具复用", "❌ 难以复用", "✅ 多个应用共享"),
        ("热插拔", "❌ 需要修改代码", "✅ 即插即用"),
        ("维护性", "❌ 耦合度高", "✅ 独立维护"),
        ("扩展性", "❌ 修改成本高", "✅ 易于扩展"),
        ("安全性", "⚠️  本地执行", "✅ 可隔离执行"),
        ("标准化", "❌ 各自实现", "✅ 统一协议"),
    ]
    
    print(f"{'维度':<12} | {'传统方式':<20} | {'MCP 方式':<20}")
    print("-" * 70)
    for dimension, traditional, mcp in comparison:
        print(f"{dimension:<12} | {traditional:<20} | {mcp:<20}")
    
    print("\n💡 结论:")
    print("   MCP 提供了更好的架构设计，特别适合：")
    print("   - 企业级应用（需要工具复用）")
    print("   - 多团队协作（工具标准化）")
    print("   - 长期维护项目（降低耦合）")


# ==========================================
# 主程序
# ==========================================

def main():
    """主菜单"""
    while True:
        print("\n" + "="*70)
        print("MCP 与 Agent 集成演示")
        print("="*70)
        print("\n请选择演示模式：")
        print("  1. 简单任务演示（多个独立任务）")
        print("  2. 复杂任务演示（需要多步推理）")
        print("  3. 传统 vs MCP 对比")
        print("  0. 退出")
        
        choice = input("\n请输入选项 (0-3): ").strip()
        
        if choice == "1":
            demo_simple_tasks()
        elif choice == "2":
            demo_complex_task()
        elif choice == "3":
            compare_traditional_vs_mcp()
        elif choice == "0":
            print("\n👋 再见！")
            break
        else:
            print("\n❌ 无效选项，请重新选择")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║                                                            ║
    ║           MCP (Model Context Protocol) 演示                ║
    ║                                                            ║
    ║  展示如何将 MCP 服务器集成到 Agent 中                      ║
    ║                                                            ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

