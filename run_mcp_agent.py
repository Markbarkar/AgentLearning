"""
使用官方 MCP 服务器的 Agent 示例
"""

import sys
import io
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 设置 UTF-8 编码，windows默认使用gbk编码，所以需要设置为utf-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 加载环境变量
load_dotenv()

# 导入 Agent 和工具
from agent_system.core.agent import Agent
from agent_system.tools import finish_tool, create_filesystem_tools


def main():
    """主函数"""
    print("="*70)
    print("🤖 MCP Agent")
    print("="*70)
    print()
    
    # 用于存储 MCP 适配器（保持连接）
    mcp_adapter = None
    
    try:
        # 1. 创建 LLM
        print("📦 正在初始化 LLM...")
        llm = ChatOpenAI(
            model="deepseek-reasoner",
            temperature=0,
            base_url="https://api.deepseek.com"
        )
        print("✅ LLM 初始化完成\n")
        
        # 2. 创建 MCP 工具（文件系统）
        print("🔧 正在加载 MCP 工具...")
        mcp_tools, mcp_adapter = create_filesystem_tools(allowed_directory=".")
        print(f"✅ 已加载 {len(mcp_tools)} 个 MCP 工具\n")
        
        # 3. 组合所有工具
        all_tools = mcp_tools + [finish_tool]
        
        # 4. 创建 Agent
        print("🤖 正在创建 Agent...")
        agent = Agent(
            llm=llm,
            tools=all_tools,
            max_thought_steps=10
        )
        print("✅ Agent 创建完成\n")
        
        # 5. 运行任务
        print("="*70)
        print("🎯 开始执行任务")
        print("="*70)
        print()
        
        # task = "帮我创建一个mcp学习文档"
        task = "仅详细回答我怎么学习java？"

        result = agent.run(task)
        
        print("\n" + "="*70)
        print("✅ 任务完成")
        print("="*70)
        print(f"\n最终结果:\n{result}")
        
    finally:
        # 清理：关闭 MCP 连接
        if mcp_adapter:
            print("\n🔌 正在关闭 MCP 连接...")
            mcp_adapter.close()
            print("✅ 连接已关闭")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被用户中断")
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()