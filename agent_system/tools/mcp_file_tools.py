"""
MCP 官方服务器工具适配器
将官方 MCP 服务器的工具转换为 LangChain 工具

支持两种使用方式：
1. 直接调用函数创建指定服务器的工具
2. 从 mcp_servers.json 配置文件加载所有启用的服务器工具
"""

import asyncio
import json
import sys
import threading
import inspect
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..config.mcp_config import (
    load_mcp_config,
    get_enabled_servers,
    get_server_config,
    MCPServerConfig
)


# 根据操作系统选择正确的 npx 命令
NPX_COMMAND = "npx.cmd" if sys.platform == "win32" else "npx"


class MCPClientAdapter:
    """MCP 客户端适配器 - 连接官方 MCP 服务器"""
    
    def __init__(
        self, 
        server_command: str, 
        server_args: List[str],
        server_env: Optional[Dict[str, str]] = None,
        server_name: str = "unknown"
    ):
        """
        初始化 MCP 客户端
        
        参数:
            server_command: 服务器命令 (如 "npx.cmd")
            server_args: 服务器参数 (如 ["-y", "@modelcontextprotocol/server-filesystem", "."])
            server_env: 环境变量字典（可选）
            server_name: 服务器名称（用于日志）
        """
        self.server_name = server_name
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args,
            env=server_env
        )
        self.session: Optional[ClientSession] = None
        self.tools_info: List[Dict[str, Any]] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._stdio_context = None
        self._session_context = None
    
    @classmethod
    def from_config(cls, config: MCPServerConfig) -> "MCPClientAdapter":
        """
        从配置对象创建适配器
        
        参数:
            config: MCP 服务器配置对象
            
        返回:
            MCPClientAdapter 实例
        """
        return cls(
            server_command=config.get_command(),
            server_args=config.args,
            server_env=config.get_env() if config.env else None,
            server_name=config.name
        )
        
    def _run_event_loop(self, loop):
        """在独立线程中运行事件循环"""
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    async def _connect_async(self):
        """异步连接到 MCP 服务器"""
        print(f"🔌 正在连接到 MCP 服务器 [{self.server_name}]...")
        
        # 使用 async with 正确管理上下文
        self._stdio_context = stdio_client(self.server_params)
        read, write = await self._stdio_context.__aenter__()
        
        self._session_context = ClientSession(read, write)
        self.session = await self._session_context.__aenter__()
        
        # 初始化会话
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
        
        print(f"✅ [{self.server_name}] 已连接，发现 {len(self.tools_info)} 个工具")
        
    def connect(self):
        """同步方式连接到 MCP 服务器"""
        # 创建新的事件循环在独立线程中运行
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, args=(self.loop,), daemon=True)
        self.thread.start()
        
        # 在新线程的事件循环中执行连接
        future = asyncio.run_coroutine_threadsafe(self._connect_async(), self.loop)
        future.result()  # 等待连接完成
        
        return self
    
    def call_tool_sync(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        同步方式调用 MCP 工具
        
        参数:
            tool_name: 工具名称
            arguments: 工具参数
            
        返回:
            工具执行结果（JSON 字符串）
        """
        if not self.session or not self.loop:
            raise RuntimeError("未连接到 MCP 服务器")
        
        # 在事件循环线程中执行异步调用
        future = asyncio.run_coroutine_threadsafe(
            self._call_tool_async(tool_name, arguments),
            self.loop
        )
        return future.result()
    
    async def _call_tool_async(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """异步调用工具"""
        result = await self.session.call_tool(tool_name, arguments)
        
        # 将结果转换为字符串
        if hasattr(result, 'content'):
            # 处理 MCP 响应格式
            content_list = result.content
            if content_list:
                return str(content_list[0].text if hasattr(content_list[0], 'text') else content_list[0])
        
        return json.dumps(result, ensure_ascii=False)
    
    def get_tools_info(self) -> List[Dict[str, Any]]:
        """获取所有工具信息"""
        return self.tools_info
    
    def close(self):
        """关闭连接"""
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread:
            self.thread.join(timeout=5)


def create_mcp_langchain_tools(
    server_command: str = None,
    server_args: List[str] = None
) -> Tuple[List[StructuredTool], MCPClientAdapter]:
    """
    创建 MCP LangChain 工具
    
    参数:
        server_command: MCP 服务器命令（默认自动检测：Windows 用 npx.cmd，其他用 npx）
        server_args: MCP 服务器参数
        
    返回:
        (LangChain 工具列表, MCP 适配器实例)
    """
    if server_command is None:
        server_command = NPX_COMMAND
    
    if server_args is None:
        # 默认：文件系统服务器，访问当前目录
        server_args = ["-y", "@modelcontextprotocol/server-filesystem", "."]
    
    # 创建适配器并连接
    adapter = MCPClientAdapter(server_command, server_args)
    adapter.connect()
    
    # 转换为 LangChain 工具
    langchain_tools = []
    
    for tool_info in adapter.get_tools_info():
        tool_name = tool_info["name"]
        
        # 从 inputSchema 提取参数信息
        input_schema = tool_info.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])
        
        # 使用 Pydantic 动态创建参数模型
        # 这是正确的方式：为每个参数创建明确的字段
        field_definitions = {}
        #参数名和参数类型（path和string等）
        for param_name, param_schema in properties.items():
            #参数描述
            param_desc = param_schema.get("description", "")
            is_required = param_name in required_params
            
            # 简化处理：所有参数都用 str 类型
            if is_required:
                # ...表示这是一个必需字段，没有默认值
                field_definitions[param_name] = (str, Field(..., description=param_desc))
            else:
                # Optional[str]表示这个参数是可选的，类型是str，默认值是None
                field_definitions[param_name] = (Optional[str], Field(None, description=param_desc))
        
        # 创建 Pydantic 模型，用于构建Langchain格式的工具
        ArgsSchema = create_model(
            f"{tool_name}_args",
            **field_definitions
        )
        
        # 创建工具函数（使用闭包捕获 tool_name 和 adapter）
        def make_tool_func(name: str, adp: MCPClientAdapter):
            def tool_func(**kwargs) -> str:
                """调用 MCP 工具"""
                # 过滤掉 None 值，即值为None的字段直接不传入
                arguments = {k: v for k, v in kwargs.items() if v is not None}
                return adp.call_tool_sync(name, arguments)
            return tool_func
        
        tool_func = make_tool_func(tool_name, adapter)
        
        # 创建 LangChain 工具（显式指定 args_schema）
        lc_tool = StructuredTool(
            name=tool_name,
            description=tool_info["description"],
            func=tool_func,
            args_schema=ArgsSchema
        )
        
        langchain_tools.append(lc_tool)
        print(f"   ✅ 工具 '{tool_name}' 已加载")
    
    return langchain_tools, adapter


# 便捷函数：创建文件系统工具
def create_filesystem_tools(allowed_directory: str = ".") -> Tuple[List[StructuredTool], MCPClientAdapter]:
    """
    创建文件系统 MCP 工具
    
    参数:
        allowed_directory: 允许访问的目录路径
        
    返回:
        (LangChain 工具列表, MCP 适配器实例)
    """
    return create_mcp_langchain_tools(
        server_command=NPX_COMMAND,
        server_args=["-y", "@modelcontextprotocol/server-filesystem", allowed_directory]
    )


# 便捷函数：创建 SQLite 工具
def create_sqlite_tools(database_path: str) -> Tuple[List[StructuredTool], MCPClientAdapter]:
    """
    创建 SQLite MCP 工具
    
    参数:
        database_path: 数据库文件路径
        
    返回:
        (LangChain 工具列表, MCP 适配器实例)
    """
    return create_mcp_langchain_tools(
        server_command=NPX_COMMAND,
        server_args=["-y", "@modelcontextprotocol/server-sqlite", database_path]
    )


# ==================== 配置文件驱动的工具创建 ====================

def create_tools_from_config(
    config_path: Optional[str] = None,
    server_names: Optional[List[str]] = None,
    servers: Optional[Dict[str, MCPServerConfig]] = None
) -> Tuple[List[StructuredTool], Dict[str, MCPClientAdapter]]:
    """
    从配置文件创建 MCP LangChain 工具
    
    遍历配置文件中所有启用的服务器，创建连接并生成工具
    
    参数:
        config_path: 配置文件路径（默认使用 agent_system/config/mcp_servers.json）
        server_names: 要加载的服务器名称列表（默认加载所有启用的服务器）
        servers: 预加载的服务器配置字典（用于用户级配置覆盖场景）
        
    返回:
        (所有工具的列表, 服务器名称到适配器的字典)
        
    示例:
        # 加载所有启用的服务器
        tools, adapters = create_tools_from_config()
        
        # 只加载指定的服务器
        tools, adapters = create_tools_from_config(server_names=["filesystem", "github"])
        
        # 使用预合并的用户配置
        tools, adapters = create_tools_from_config(servers=merged_user_servers)
    """
    all_tools: List[StructuredTool] = []
    adapters: Dict[str, MCPClientAdapter] = {}
    
    # 获取配置（优先使用传入的 servers 参数）
    if servers is not None:
        # 使用传入的服务器配置（用于用户级配置）
        # 过滤出启用的服务器
        servers = {name: cfg for name, cfg in servers.items() if cfg.enabled}
    elif server_names:
        # 加载指定的服务器
        all_servers = load_mcp_config(config_path)
        servers = {name: all_servers[name] for name in server_names if name in all_servers}
    else:
        # 加载所有启用的服务器
        servers = get_enabled_servers(config_path)
    
    if not servers:
        print("⚠️ 没有找到启用的 MCP 服务器配置")
        return all_tools, adapters
    
    print(f"📦 准备加载 {len(servers)} 个 MCP 服务器...")
    
    for name, config in servers.items():
        if not config.enabled and name not in (server_names or []):
            continue
            
        try:
            print(f"\n--- 加载服务器: {name} ---")
            if config.description:
                print(f"   描述: {config.description}")
            
            # 从配置创建适配器
            adapter = MCPClientAdapter.from_config(config)
            adapter.connect()
            
            # 转换为 LangChain 工具
            tools = _convert_adapter_to_tools(adapter)
            
            all_tools.extend(tools)
            adapters[name] = adapter
            
            print(f"   ✅ 服务器 '{name}' 加载成功，共 {len(tools)} 个工具")
            
        except Exception as e:
            print(f"   ❌ 服务器 '{name}' 加载失败: {e}")
            continue
    
    print(f"\n📊 总计加载 {len(all_tools)} 个工具，来自 {len(adapters)} 个服务器")
    return all_tools, adapters


def create_tools_by_server_name(
    server_name: str,
    config_path: Optional[str] = None
) -> Tuple[List[StructuredTool], MCPClientAdapter]:
    """
    根据服务器名称从配置文件创建工具
    
    参数:
        server_name: 配置文件中的服务器名称
        config_path: 配置文件路径
        
    返回:
        (LangChain 工具列表, MCP 适配器实例)
        
    异常:
        ValueError: 服务器名称不存在
    """
    config = get_server_config(server_name, config_path)
    
    if config is None:
        raise ValueError(f"服务器 '{server_name}' 在配置文件中不存在")
    
    adapter = MCPClientAdapter.from_config(config)
    adapter.connect()
    
    tools = _convert_adapter_to_tools(adapter)
    
    return tools, adapter


def _convert_adapter_to_tools(adapter: MCPClientAdapter) -> List[StructuredTool]:
    """
    将 MCP 适配器的工具信息转换为 LangChain 工具
    
    这是一个内部函数，用于复用工具转换逻辑
    """
    langchain_tools = []
    
    for tool_info in adapter.get_tools_info():
        tool_name = tool_info["name"]
        
        # 从 inputSchema 提取参数信息
        input_schema = tool_info.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])
        
        # 使用 Pydantic 动态创建参数模型
        field_definitions = {}
        for param_name, param_schema in properties.items():
            param_desc = param_schema.get("description", "")
            is_required = param_name in required_params
            
            if is_required:
                field_definitions[param_name] = (str, Field(..., description=param_desc))
            else:
                field_definitions[param_name] = (Optional[str], Field(None, description=param_desc))
        
        # 创建 Pydantic 模型
        ArgsSchema = create_model(
            f"{tool_name}_args",
            **field_definitions
        )
        
        # 创建工具函数
        def make_tool_func(name: str, adp: MCPClientAdapter):
            def tool_func(**kwargs) -> str:
                arguments = {k: v for k, v in kwargs.items() if v is not None}
                return adp.call_tool_sync(name, arguments)
            return tool_func
        
        tool_func = make_tool_func(tool_name, adapter)
        
        # 创建 LangChain 工具
        lc_tool = StructuredTool(
            name=tool_name,
            description=tool_info["description"],
            func=tool_func,
            args_schema=ArgsSchema
        )
        
        langchain_tools.append(lc_tool)
        print(f"      ✅ 工具 '{tool_name}' 已加载")
    
    return langchain_tools


def close_all_adapters(adapters: Dict[str, MCPClientAdapter]):
    """
    关闭所有 MCP 适配器连接
    
    参数:
        adapters: 服务器名称到适配器的字典
    """
    for name, adapter in adapters.items():
        try:
            adapter.close()
            print(f"✅ 已关闭服务器 '{name}' 的连接")
        except Exception as e:
            print(f"⚠️ 关闭服务器 '{name}' 时出错: {e}")
