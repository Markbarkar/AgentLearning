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

# ==================== RAG 配置 ====================
# Qwen Embedding 模型配置
QWEN_EMBEDDING_MODEL = "text-embedding-v3"  # Qwen Embedding 模型名称
QWEN_EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY")  # DashScope API Key
QWEN_EMBEDDING_DIMENSION = 1024  # 向量维度

# Chroma 向量数据库配置
CHROMA_PERSIST_DIR = "./data/chroma_db"  # Chroma 持久化存储路径
CHROMA_COLLECTION_NAME = "legal_documents"  # 集合名称（已废弃，仅保留向后兼容）

# 多用户知识库配置
PUBLIC_COLLECTION_NAME = "public_knowledge_base"  # 公共知识库（无user_id时使用）
ENABLE_USER_ISOLATION = True  # 是否启用用户隔离

# RAG 检索配置
RAG_TOP_K = 3  # 检索返回的最相关文档数量
RAG_SIMILARITY_THRESHOLD = 0.3  # 相似度阈值(默认0.7)
RAG_CHUNK_SIZE = 800  # 文本分块大小
RAG_CHUNK_OVERLAP = 100  # 文本分块重叠大小

# 知识库文档路径
KNOWLEDGE_BASE_DOCS_DIR = "./data/knowledge_base"  # 知识库文档目录


