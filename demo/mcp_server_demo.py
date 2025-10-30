"""
MCP 服务器演示
================

这是一个简单的 MCP 服务器实现，提供天气查询和计算器功能
展示如何将 MCP 服务器集成到 Agent 中

MCP (Model Context Protocol) 优势：
1. 标准化接口：统一的工具调用方式
2. 独立部署：服务器可以独立运行和维护
3. 多应用共享：一个 MCP 服务器可以被多个 Agent 使用
4. 安全隔离：敏感操作在独立进程中执行
"""

import sys
import io
from typing import Any, List, Dict
import json
from datetime import datetime

# 设置标准输出编码为 UTF-8，解决 Windows 控制台乱码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


class MCPServer:
    """
    简单的 MCP 服务器实现
    
    MCP 服务器主要职责：
    1. 注册可用的工具
    2. 接收工具调用请求
    3. 执行工具并返回结果
    4. 提供工具列表和描述
    """
    
    def __init__(self, name: str):
        """
        初始化 MCP 服务器
        
        参数：
            name: 服务器名称
        """
        self.name = name
        self.tools: Dict[str, Dict[str, Any]] = {}
        print(f"✅ MCP 服务器 '{name}' 已启动")
    
    def register_tool(
        self, 
        name: str, 
        description: str,
        parameters: Dict[str, Any],
        handler: callable
    ):
        """
        注册工具到 MCP 服务器
        
        参数：
            name: 工具名称
            description: 工具描述
            parameters: 参数定义（JSON Schema 格式）
            handler: 处理函数
        """
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }
        print(f"📝 工具 '{name}' 已注册")
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        列出所有可用工具
        
        返回：工具列表（不包含处理函数）
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        调用工具
        
        这是 MCP 服务器的核心方法
        模拟标准的 MCP 协议调用
        
        参数：
            tool_name: 工具名称
            arguments: 工具参数
            
        返回：
            {
                "success": bool,
                "result": Any,
                "error": str (可选)
            }
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "error": f"工具 '{tool_name}' 不存在"
            }
        
        tool = self.tools[tool_name]
        try:
            # 调用工具的处理函数
            result = tool["handler"](**arguments)
            return {
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"工具执行失败: {str(e)}"
            }
    
    def get_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        return {
            "name": self.name,
            "version": "1.0.0",
            "tool_count": len(self.tools),
            "tools": self.list_tools()
        }


# ==========================================
# 工具实现示例
# ==========================================

def get_weather(city: str, date: str = None) -> Dict[str, Any]:
    """
    天气查询工具
    
    模拟天气查询 API
    在实际应用中，这里会调用真实的天气 API
    """
    # 模拟数据
    weather_data = {
        "北京": {"temperature": "15°C", "condition": "晴天", "humidity": "45%"},
        "上海": {"temperature": "18°C", "condition": "多云", "humidity": "60%"},
        "广州": {"temperature": "25°C", "condition": "阴天", "humidity": "70%"},
        "深圳": {"temperature": "26°C", "condition": "晴天", "humidity": "65%"},
    }
    
    if city not in weather_data:
        return {
            "city": city,
            "message": "暂无该城市的天气数据"
        }
    
    date_str = date or datetime.now().strftime("%Y-%m-%d")
    
    return {
        "city": city,
        "date": date_str,
        "weather": weather_data[city]
    }


def calculate(expression: str) -> Dict[str, Any]:
    """
    计算器工具
    
    安全地计算数学表达式
    """
    try:
        # 安全计算（只允许基本运算）
        # 实际应用中应该使用更安全的表达式解析器
        allowed_chars = set("0123456789+-*/.()")
        if not all(c in allowed_chars or c.isspace() for c in expression):
            return {
                "error": "表达式包含不允许的字符"
            }
        
        result = eval(expression)
        return {
            "expression": expression,
            "result": result
        }
    except Exception as e:
        return {
            "error": f"计算失败: {str(e)}"
        }


def get_current_time(timezone: str = "Asia/Shanghai") -> Dict[str, Any]:
    """
    获取当前时间
    
    参数：
        timezone: 时区（这里简化处理）
    """
    now = datetime.now()
    return {
        "timezone": timezone,
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": int(now.timestamp())
    }


def search_database(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    数据库搜索工具（模拟）
    
    模拟在数据库中搜索信息
    """
    # 模拟数据库
    database = [
        {"id": 1, "title": "Python 教程", "content": "学习 Python 编程"},
        {"id": 2, "title": "AI Agent 开发", "content": "构建智能 Agent"},
        {"id": 3, "title": "MCP 协议", "content": "Model Context Protocol"},
        {"id": 4, "title": "LangChain 框架", "content": "LangChain 使用指南"},
        {"id": 5, "title": "DeepSeek 模型", "content": "DeepSeek 模型介绍"},
    ]
    
    # 简单的关键词匹配
    results = [
        item for item in database
        if query.lower() in item["title"].lower() or query.lower() in item["content"].lower()
    ]
    
    return {
        "query": query,
        "total": len(results),
        "results": results[:limit]
    }


# ==========================================
# 创建并配置 MCP 服务器
# ==========================================

def create_demo_mcp_server() -> MCPServer:
    """
    创建演示用的 MCP 服务器
    
    注册多个工具，展示 MCP 服务器的能力
    """
    # 创建服务器实例
    server = MCPServer(name="DemoMCPServer")
    
    # 注册天气查询工具
    server.register_tool(
        name="get_weather",
        description="查询指定城市的天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，如：北京、上海"
                },
                "date": {
                    "type": "string",
                    "description": "日期（可选），格式：YYYY-MM-DD"
                }
            },
            "required": ["city"]
        },
        handler=get_weather
    )
    
    # 注册计算器工具
    server.register_tool(
        name="calculate",
        description="计算数学表达式，支持加减乘除和括号",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "数学表达式，如：(10 + 20) * 3"
                }
            },
            "required": ["expression"]
        },
        handler=calculate
    )
    
    # 注册时间查询工具
    server.register_tool(
        name="get_current_time",
        description="获取当前时间",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，默认：Asia/Shanghai"
                }
            },
            "required": []
        },
        handler=get_current_time
    )
    
    # 注册数据库搜索工具
    server.register_tool(
        name="search_database",
        description="在数据库中搜索信息",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词"
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量限制，默认5"
                }
            },
            "required": ["query"]
        },
        handler=search_database
    )
    
    return server


# ==========================================
# 演示和测试
# ==========================================

def demo_mcp_usage():
    """
    演示 MCP 服务器的基本使用
    """
    print("\n" + "="*60)
    print("MCP 服务器演示")
    print("="*60 + "\n")
    
    # 创建 MCP 服务器
    server = create_demo_mcp_server()
    
    print("\n" + "-"*60)
    print("📋 服务器信息")
    print("-"*60)
    info = server.get_info()
    print(f"名称: {info['name']}")
    print(f"版本: {info['version']}")
    print(f"工具数量: {info['tool_count']}")
    
    print("\n" + "-"*60)
    print("🛠️  可用工具列表")
    print("-"*60)
    for i, tool in enumerate(server.list_tools(), 1):
        print(f"\n{i}. {tool['name']}")
        print(f"   描述: {tool['description']}")
        print(f"   参数: {json.dumps(tool['parameters'], ensure_ascii=False, indent=6)}")
    
    print("\n" + "-"*60)
    print("🎯 工具调用演示")
    print("-"*60)
    
    # 测试1: 查询天气
    print("\n1️⃣  查询北京天气")
    result1 = server.call_tool("get_weather", {"city": "北京"})
    print(f"   结果: {json.dumps(result1, ensure_ascii=False, indent=3)}")
    
    # 测试2: 计算表达式
    print("\n2️⃣  计算 (100 + 50) * 2")
    result2 = server.call_tool("calculate", {"expression": "(100 + 50) * 2"})
    print(f"   结果: {json.dumps(result2, ensure_ascii=False, indent=3)}")
    
    # 测试3: 获取当前时间
    print("\n3️⃣  获取当前时间")
    result3 = server.call_tool("get_current_time", {})
    print(f"   结果: {json.dumps(result3, ensure_ascii=False, indent=3)}")
    
    # 测试4: 搜索数据库
    print("\n4️⃣  搜索关键词 'Agent'")
    result4 = server.call_tool("search_database", {"query": "Agent", "limit": 3})
    print(f"   结果: {json.dumps(result4, ensure_ascii=False, indent=3)}")
    
    # 测试5: 错误处理
    print("\n5️⃣  测试错误处理 - 调用不存在的工具")
    result5 = server.call_tool("non_existent_tool", {})
    print(f"   结果: {json.dumps(result5, ensure_ascii=False, indent=3)}")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60 + "\n")
    
    return server


if __name__ == "__main__":
    # 运行演示
    mcp_server = demo_mcp_usage()
    
    print("\n💡 提示：")
    print("   - 这个 MCP 服务器现在可以被 Agent 使用")
    print("   - 查看 'mcp_agent_integration.py' 了解如何集成到 Agent")
    print("   - MCP 服务器可以独立运行，被多个应用共享")

