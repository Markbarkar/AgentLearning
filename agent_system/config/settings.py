"""
系统配置文件

集中管理所有配置项，包括：
- LLM 模型配置
- Agent 运行参数
- API 密钥等
"""

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# ==================== LLM 配置 ====================
LLM_MODEL = "deepseek-chat"  # DeepSeek-V3 模型
LLM_TEMPERATURE = 0  # 温度为0，输出更确定
LLM_BASE_URL = "https://api.deepseek.com"  # DeepSeek API 地址
LLM_API_KEY = os.getenv("OPENAI_API_KEY")  # 从环境变量获取 API 密钥
LLM_SEED = 42  # 随机种子，确保结果可复现

# ==================== Agent 配置 ====================
MAX_THOUGHT_STEPS = 10  # 最大思考步数，防止死循环
AGENT_VERBOSE = True  # 是否打印详细日志

# ==================== 记忆配置 ====================
MEMORY_TYPE = "buffer"  # 记忆类型: buffer, token_buffer, summary
MEMORY_RETURN_MESSAGES = True  # 是否以消息格式返回历史

# ==================== 工具配置 ====================
TOOL_TIMEOUT = 30  # 工具执行超时时间（秒）
TOOL_RETRY_TIMES = 3  # 工具执行失败重试次数


