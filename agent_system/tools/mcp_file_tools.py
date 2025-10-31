"""
MCP 官方服务器工具适配器
将官方 MCP 服务器的工具转换为 LangChain 工具
"""

import asyncio
import json
import threading
import inspect
from typing import Dict, Any, List, Optional
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClientAdapter:
    """MCP 客户端适配器 - 连接官方 MCP 服务器"""
    
    def __init__(self, server_command: str, server_args: List[str]):
        """
        初始化 MCP 客户端
        
        参数:
            server_command: 服务器命令 (如 "npx.cmd")
            server_args: 服务器参数 (如 ["-y", "@modelcontextprotocol/server-filesystem", "."])
        """
        self.server_params = StdioServerParameters(
            command=server_command,
            args=server_args
        )
        self.session: Optional[ClientSession] = None
        self.tools_info: List[Dict[str, Any]] = []
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.thread: Optional[threading.Thread] = None
        self._stdio_context = None
        self._session_context = None
        
    def _run_event_loop(self, loop):
        """在独立线程中运行事件循环"""
        asyncio.set_event_loop(loop)
        loop.run_forever()
    
    async def _connect_async(self):
        """异步连接到 MCP 服务器"""
        print(f"🔌 正在连接到 MCP 服务器...")
        
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
        
        print(f"✅ 已连接，发现 {len(self.tools_info)} 个工具")
        
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
    server_command: str = "npx.cmd",
    server_args: List[str] = None
) -> tuple[List[StructuredTool], MCPClientAdapter]:
    """
    创建 MCP LangChain 工具
    
    参数:
        server_command: MCP 服务器命令
        server_args: MCP 服务器参数
        
    返回:
        (LangChain 工具列表, MCP 适配器实例)
    """
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
def create_filesystem_tools(allowed_directory: str = ".") -> tuple[List[StructuredTool], MCPClientAdapter]:
    """
    创建文件系统 MCP 工具
    
    参数:
        allowed_directory: 允许访问的目录路径
        
    返回:
        (LangChain 工具列表, MCP 适配器实例)
    """
    return create_mcp_langchain_tools(
        server_command="npx.cmd",
        server_args=["-y", "@modelcontextprotocol/server-filesystem", allowed_directory]
    )


# 便捷函数：创建 SQLite 工具
def create_sqlite_tools(database_path: str) -> tuple[List[StructuredTool], MCPClientAdapter]:
    """
    创建 SQLite MCP 工具
    
    参数:
        database_path: 数据库文件路径
        
    返回:
        (LangChain 工具列表, MCP 适配器实例)
    """
    return create_mcp_langchain_tools(
        server_command="npx.cmd",
        server_args=["-y", "@modelcontextprotocol/server-sqlite", database_path]
    )
