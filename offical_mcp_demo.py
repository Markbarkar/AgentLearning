import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def connect_to_filesystem_server():
    # 配置服务器参数
    # Windows 上需要使用 npx.cmd
    server_params = StdioServerParameters(
        command="npx.cmd",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."]
    )
    
    # 连接到服务器
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            
            # 列出可用工具
            tools = await session.list_tools()
            print("Available tools:", tools)
            
            # 调用工具（例如读取文件）
            result = await session.call_tool("read_file", {
                "path": "MCP_资源汇总.md"
            })
            print("File content:", result)

# 运行
asyncio.run(connect_to_filesystem_server())