#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MCP 服务测试脚本

测试所有启用的 MCP 服务器配置是否能正常加载和运行

注意：此脚本独立于 agent_system，只依赖 mcp 模块
"""

import sys
import os
import json
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ==================== 配置加载（独立实现，避免导入依赖） ====================

def load_mcp_config_standalone(config_path: str = None) -> Dict[str, dict]:
    """
    独立加载 MCP 配置（不依赖 agent_system）
    """
    if config_path is None:
        config_path = os.path.join(
            PROJECT_ROOT, 
            "agent_system", "config", "mcp_servers.json"
        )
    
    if not os.path.exists(config_path):
        print(f"⚠️ 配置文件不存在: {config_path}")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("mcpServers", {})


def get_enabled_servers_standalone(config: dict) -> Dict[str, dict]:
    """获取所有启用的服务器"""
    return {
        name: cfg for name, cfg in config.items() 
        if cfg.get("enabled", True)
    }


def get_platform_command(command: str) -> str:
    """获取平台适配的命令"""
    if sys.platform == "win32" and command == "npx":
        return "npx.cmd"
    return command


# ==================== MCP 连接测试 ====================

class SimpleMCPClient:
    """简化的 MCP 客户端（用于测试）"""
    
    def __init__(self, server_name: str, command: str, args: List[str], env: dict = None):
        self.server_name = server_name
        self.command = get_platform_command(command)
        self.args = args
        self.env = env
        self.session = None
        self.tools_info = []
        self.loop = None
        self.thread = None
        self._stdio_context = None
        self._session_context = None
    
    def _run_event_loop(self, loop):
        """在独立线程中运行事件循环"""
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    async def _connect_async(self):
        """异步连接"""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        # 合并环境变量
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        
        server_params = StdioServerParameters(
            command=self.command,
            args=self.args,
            env=env if self.env else None
        )
        
        print(f"   🔌 正在连接 [{self.server_name}]...")
        print(f"      命令: {self.command} {' '.join(self.args)}")
        
        self._stdio_context = stdio_client(server_params)
        read, write = await self._stdio_context.__aenter__()
        
        self._session_context = ClientSession(read, write)
        self.session = await self._session_context.__aenter__()
        
        await self.session.initialize()
        
        # 获取工具列表
        tools_response = await self.session.list_tools()
        self.tools_info = [
            {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": tool.inputSchema
            }
            for tool in tools_response.tools
        ]
    
        print(f"   ✅ [{self.server_name}] 连接成功，发现 {len(self.tools_info)} 个工具")
    
    def connect(self):
        """同步连接"""
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(
            target=self._run_event_loop, 
            args=(self.loop,), 
            daemon=True
        )
        self.thread.start()
        
        future = asyncio.run_coroutine_threadsafe(self._connect_async(), self.loop)
        future.result(timeout=30)  # 30秒超时
        
        return self
    
    async def _call_tool_async(self, tool_name: str, arguments: dict) -> str:
        """异步调用工具"""
        result = await self.session.call_tool(tool_name, arguments)
        
        if hasattr(result, 'content'):
            content_list = result.content
            if content_list:
                return str(content_list[0].text if hasattr(content_list[0], 'text') else content_list[0])
        
        return json.dumps(result, ensure_ascii=False)
    
    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """同步调用工具"""
        if not self.session or not self.loop:
            raise RuntimeError("未连接到 MCP 服务器")
        
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(tool_name, arguments),
            self.loop
        )
        return future.result(timeout=30)
    
    def close(self):
        """关闭连接"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)


# ==================== 测试函数 ====================

def test_config_loading():
    """测试配置文件加载"""
    print("=" * 60)
    print("📋 测试 1: 配置文件加载")
    print("=" * 60)
    
    config = load_mcp_config_standalone()
    
    if not config:
        print("❌ 配置加载失败")
        return None
    
    print(f"\n✅ 配置文件加载成功")
    print(f"   总服务器数量: {len(config)}")
    
    print("\n📦 所有服务器配置:")
    for name, cfg in config.items():
        enabled = cfg.get("enabled", True)
        status = "🟢 启用" if enabled else "🔴 禁用"
        print(f"   [{status}] {name}")
        cmd = get_platform_command(cfg.get("command", "npx"))
        args = " ".join(cfg.get("args", []))
        print(f"       命令: {cmd} {args}")
        if cfg.get("description"):
            print(f"       描述: {cfg.get('description')}")
        if cfg.get("env"):
            env_keys = list(cfg.get("env").keys())
            print(f"       环境变量: {env_keys}")
    
    enabled = get_enabled_servers_standalone(config)
    print(f"\n🟢 启用的服务器: {list(enabled.keys())}")
    
    return enabled


def test_mcp_connection(enabled_servers: dict):
    """测试 MCP 服务器连接"""
    print("\n" + "=" * 60)
    print("🔌 测试 2: MCP 服务器连接")
    print("=" * 60)
    
    if not enabled_servers:
        print("⚠️ 没有启用的服务器，跳过连接测试")
        return {}
    
    clients = {}
    
    for name, cfg in enabled_servers.items():
        print(f"\n--- 连接服务器: {name} ---")
        
        try:
            client = SimpleMCPClient(
                server_name=name,
                command=cfg.get("command", "npx"),
                args=cfg.get("args", []),
                env=cfg.get("env")
            )
            client.connect()
            clients[name] = client
            
        except Exception as e:
            print(f"   ❌ 连接失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 连接结果: 成功 {len(clients)}/{len(enabled_servers)}")
    
    return clients


def test_tools_info(clients: dict):
    """测试工具信息"""
    print("\n" + "=" * 60)
    print("🔧 测试 3: 工具信息")
    print("=" * 60)
    
    if not clients:
        print("⚠️ 没有连接的客户端，跳过工具测试")
        return
    
    total_tools = 0
    
    for server_name, client in clients.items():
        tools = client.tools_info
        total_tools += len(tools)
        
        print(f"\n📦 [{server_name}] 共 {len(tools)} 个工具:")
        
        for i, tool in enumerate(tools, 1):
            desc = tool.get("description", "无描述")
            if len(desc) > 60:
                desc = desc[:57] + "..."
            print(f"   {i}. {tool['name']}")
            print(f"      {desc}")
    
    print(f"\n📊 总计: {total_tools} 个工具")


def test_tool_execution(clients: dict):
    """测试工具执行（仅对安全的只读操作）"""
    print("\n" + "=" * 60)
    print("▶️  测试 4: 工具执行（只读操作）")
    print("=" * 60)
    
    if not clients:
        print("⚠️ 没有连接的客户端，跳过执行测试")
        return
    
    for server_name, client in clients.items():
        print(f"\n--- 测试服务器: {server_name} ---")
        
        # 查找只读工具
        safe_tools = []
        for tool in client.tools_info:
            name_lower = tool["name"].lower()
            if any(kw in name_lower for kw in ['list', 'read', 'get', 'search']):
                safe_tools.append(tool)
        
        if not safe_tools:
            print("   ⚠️ 没有找到安全的只读工具")
            continue
        
        # 测试第一个安全工具
        for tool in safe_tools:
            tool_name = tool["name"]
            
            # 尝试推断参数
            input_schema = tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            
            # 对于 list_directory 类工具，使用当前目录
            if "path" in properties:
                args = {"path": "."}
                print(f"\n   测试: {tool_name}(path='.')")
                
                try:
                    result = client.call_tool(tool_name, args)
                    print(f"   ✅ 执行成功!")
                    
                    # 截断结果
                    if len(result) > 500:
                        result = result[:500] + "\n   ... (结果已截断)"
                    print(f"   结果:\n   {result}")
                    
                except Exception as e:
                    print(f"   ❌ 执行失败: {e}")
                
                break  # 只测试一个工具


def cleanup(clients: dict):
    """清理资源"""
    print("\n" + "=" * 60)
    print("🧹 清理资源")
    print("=" * 60)
    
    if clients:
        for name, client in clients.items():
            try:
                client.close()
                print(f"   ✅ 已关闭: {name}")
            except Exception as e:
                print(f"   ⚠️ 关闭 {name} 时出错: {e}")
    else:
        print("   ℹ️ 无需清理")


def test_agent_integration():
    """测试 Agent 集成 MCP 工具"""
    print("\n" + "=" * 60)
    print("🤖 测试 5: Agent 集成 MCP 工具")
    print("=" * 60)
    
    try:
        # 导入必要模块
        from langchain_openai import ChatOpenAI
        from agent_system.config.settings import (
            LLM_MODEL, LLM_TEMPERATURE, LLM_BASE_URL, LLM_SEED
        )
        from agent_system.tools import (
            create_tools_from_config,
            close_all_adapters,
            finish_tool
        )
        from agent_system.core import Agent
        
        print("\n📦 加载 MCP 工具...")
        mcp_tools, adapters = create_tools_from_config()
        
        if not mcp_tools:
            print("⚠️ 没有加载到 MCP 工具，跳过 Agent 测试")
            return None
        
        # 添加 finish 工具
        all_tools = mcp_tools + [finish_tool]
        
        print(f"\n🔧 创建 Agent...")
        print(f"   模型: {LLM_MODEL}")
        print(f"   工具数量: {len(all_tools)}")
        
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            model_kwargs={"seed": LLM_SEED}
        )
        
        agent = Agent(
            llm=llm,
            tools=all_tools,
            use_rag=False  # MCP 测试不需要 RAG
        )
        
        # 测试任务
        test_task = "列出当前目录下的所有文件和文件夹"
        
        print(f"\n▶️  执行测试任务: {test_task}")
        print("-" * 40)
        
        result = agent.run(test_task)
        
        print("-" * 40)
        print(f"\n✅ Agent 执行完成!")
        print(f"结果:\n{result}")
        
        return adapters
        
    except ImportError as e:
        print(f"\n⚠️ 导入失败，跳过 Agent 测试: {e}")
        print("   提示: 请确保已安装 langchain_openai 和配置 API Key")
        return None
    except Exception as e:
        print(f"\n❌ Agent 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_usage_guide():
    """打印 Agent 集成使用指南"""
    print("\n" + "=" * 60)
    print("📖 Agent 集成 MCP 工具使用指南")
    print("=" * 60)
    
    guide = '''
┌─────────────────────────────────────────────────────────────┐
│  方式一：从配置文件加载所有启用的 MCP 工具                    │
└─────────────────────────────────────────────────────────────┘

```python
from langchain_openai import ChatOpenAI
from agent_system.tools import (
    create_tools_from_config,
    close_all_adapters,
    finish_tool
)
from agent_system.core import Agent

# 1. 加载 MCP 工具
mcp_tools, adapters = create_tools_from_config()

# 2. 合并其他工具
all_tools = mcp_tools + [finish_tool]

# 3. 创建 Agent
llm = ChatOpenAI(model="deepseek-chat")
agent = Agent(llm=llm, tools=all_tools, use_rag=False)

# 4. 执行任务
result = agent.run("列出当前目录的文件")

# 5. 清理连接（重要！）
close_all_adapters(adapters)
```

┌─────────────────────────────────────────────────────────────┐
│  方式二：加载指定的 MCP 服务器                               │
└─────────────────────────────────────────────────────────────┘

```python
from agent_system.tools import create_tools_by_server_name

# 只加载 filesystem 服务器
tools, adapter = create_tools_by_server_name("filesystem")

# 或者指定多个服务器
tools, adapters = create_tools_from_config(
    server_names=["filesystem", "sqlite"]
)
```

┌─────────────────────────────────────────────────────────────┐
│  方式三：在 API 中集成（推荐生产环境使用）                    │
└─────────────────────────────────────────────────────────────┘

在 agent_system/api/dependencies.py 中修改 create_qwen_vl_tools():

```python
def create_qwen_vl_tools(user_id: Optional[str] = None):
    # ... 现有工具 ...
    
    # 添加 MCP 工具
    try:
        from ..tools import create_tools_from_config
        mcp_tools, _ = create_tools_from_config()
        tools.extend(mcp_tools)
        print(f"✓ 已加载 {len(mcp_tools)} 个 MCP 工具")
    except Exception as e:
        print(f"⚠️ MCP 工具加载失败: {e}")
    
    return tools
```

┌─────────────────────────────────────────────────────────────┐
│  配置文件位置                                                │
└─────────────────────────────────────────────────────────────┘

agent_system/config/mcp_servers.json

添加新服务器只需编辑此文件，无需修改代码！
'''
    print(guide)


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("🚀 MCP 服务测试脚本")
    print("=" * 60)
    
    clients = {}
    agent_adapters = None
    
    try:
        # 测试 1: 配置加载
        enabled_servers = test_config_loading()
        
        if enabled_servers:
            # 测试 2: MCP 连接
            clients = test_mcp_connection(enabled_servers)
            
            if clients:
                # 测试 3: 工具信息
                test_tools_info(clients)
                
                # 测试 4: 工具执行
                test_tool_execution(clients)
    
        # 清理简单客户端
        cleanup(clients)
        clients = {}  # 已清理
        
        # 测试 5: Agent 集成（可选，需要 langchain）
        print("\n" + "-" * 60)
        user_input = input("是否测试 Agent 集成? (y/N): ").strip().lower()
        
        if user_input == 'y':
            agent_adapters = test_agent_integration()
        
        # 打印使用指南
        print_usage_guide()
        
    except KeyboardInterrupt:
        print("\n\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理资源
        cleanup(clients)
        
        # 清理 Agent 的 MCP 适配器
        if agent_adapters:
            print("\n🧹 清理 Agent MCP 连接...")
            try:
                from agent_system.tools import close_all_adapters
                close_all_adapters(agent_adapters)
            except:
                pass
    
    print("\n" + "=" * 60)
    print("✅ 测试完成")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
