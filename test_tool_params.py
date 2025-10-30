"""测试 LangChain StructuredTool 的参数传递"""

from langchain_core.tools import StructuredTool
from typing import Dict, Any

# 方式 1：使用 **kwargs（你当前的方式）
def tool_func_kwargs(**kwargs) -> str:
    """使用 kwargs 接收参数"""
    print(f"接收到的 kwargs: {kwargs}")
    return str(kwargs)

# 设置注解
tool_func_kwargs.__annotations__ = {
    "path": str,
    "head": int
}

# 创建工具
tool1 = StructuredTool.from_function(
    func=tool_func_kwargs,
    name="test_tool_kwargs",
    description="测试工具（kwargs 方式）"
)

print("="*60)
print("方式 1: 使用 **kwargs")
print("="*60)
print(f"工具参数 schema: {tool1.args_schema.schema()}")
print()

# 方式 2：显式定义参数（正确方式）
def tool_func_explicit(path: str, head: int = 100) -> str:
    """显式定义参数"""
    print(f"path: {path}, head: {head}")
    return f"读取文件: {path}, 前 {head} 行"

tool2 = StructuredTool.from_function(
    func=tool_func_explicit,
    name="test_tool_explicit",
    description="测试工具（显式参数）"
)

print("="*60)
print("方式 2: 显式定义参数")
print("="*60)
print(f"工具参数 schema: {tool2.args_schema.schema()}")
print()

# 测试调用
print("="*60)
print("测试调用")
print("="*60)

try:
    print("\n测试 tool1 (kwargs):")
    result1 = tool1.run({"path": "test.txt", "head": 100})
    print(f"结果: {result1}")
except Exception as e:
    print(f"错误: {e}")

try:
    print("\n测试 tool2 (explicit):")
    result2 = tool2.run({"path": "test.txt", "head": 100})
    print(f"结果: {result2}")
except Exception as e:
    print(f"错误: {e}")


