"""
Agent 系统主入口文件

演示如何使用模块化的 Agent 系统
"""

from langchain_openai import ChatOpenAI

# 从各个模块导入需要的组件
from agent_system.config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_SEED
)
from agent_system.tools import (
    search_train_ticket_tool,
    purchase_train_ticket_tool,
    finish_tool
)
from agent_system.core import Agent


def main():
    """
    主函数
    
    演示完整的 Agent 使用流程：
    1. 初始化 LLM
    2. 准备工具列表
    3. 创建 Agent 实例
    4. 运行任务
    5. 输出结果
    """
    
    print("=" * 60)
    print("🤖 Agent 系统启动")
    print("=" * 60)
    
    # 1. 初始化 LLM（大语言模型）
    print("\n📝 初始化 LLM...")
    llm = ChatOpenAI(
        model=LLM_MODEL,  # 使用 DeepSeek-V3 模型
        temperature=LLM_TEMPERATURE,  # 温度为0，输出更确定
        base_url=LLM_BASE_URL,  # DeepSeek API 地址
        model_kwargs={
            "seed": LLM_SEED  # 固定随机种子
        }
    )
    print(f"✅ LLM 初始化完成: {LLM_MODEL}")
    
    # 2. 准备工具列表
    print("\n🔧 准备工具...")
    tools = [
        search_train_ticket_tool,  # 查询火车票工具
        purchase_train_ticket_tool,  # 购买火车票工具
        finish_tool  # 完成任务工具
    ]
    print(f"✅ 工具准备完成，共 {len(tools)} 个工具")
    for tool in tools:
        print(f"   - {tool.name}")
    
    # 3. 创建 Agent 实例
    print("\n🤖 创建 Agent...")
    agent = Agent(llm=llm, tools=tools)
    print("✅ Agent 创建完成")
    
    # 4. 定义任务
    print("\n" + "=" * 60)
    print("📋 任务描述")
    print("=" * 60)
    task = "帮我查询24年6月1日早上去上海的火车票"
    print(f"任务: {task}")
    
    # 5. 运行 Agent
    print("\n" + "=" * 60)
    print("🚀 开始执行任务")
    print("=" * 60)
    print()
    
    result = agent.run(task)
    
    # 6. 输出最终结果
    print("\n" + "=" * 60)
    print("✨ 最终结果")
    print("=" * 60)
    print(result)
    print("\n" + "=" * 60)
    print("🎉 任务完成")
    print("=" * 60)


if __name__ == "__main__":
    main()


