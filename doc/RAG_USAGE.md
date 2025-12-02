# RAG 功能使用指南

本文档介绍如何使用 Qwen2.5-VL Agent 系统的 RAG（检索增强生成）功能。

## 目录

1. [功能概述](#功能概述)
2. [环境配置](#环境配置)
3. [安装依赖](#安装依赖)
4. [构建知识库](#构建知识库)
5. [使用 RAG Agent](#使用-rag-agent)
6. [API 接口说明](#api-接口说明)
7. [测试验证](#测试验证)
8. [常见问题](#常见问题)

## 功能概述

RAG（Retrieval-Augmented Generation）通过检索相关知识来增强 Agent 的回答能力：

- **自动知识检索**：任务开始前自动检索相关知识
- **工具调用检索**：Agent 可主动调用检索工具
- **多格式支持**：支持 PDF、Word、文本等多种文档格式
- **智能分块**：自动将文档分块以提高检索精度
- **Qwen Embedding**：使用 Qwen 的高质量中文 Embedding 模型

## 环境配置

### 1. 配置 API Key

在项目根目录创建或编辑 `.env` 文件：

```bash
# DeepSeek API Key（用于 LLM）
OPENAI_API_KEY=your_deepseek_api_key

# DashScope API Key（用于 Qwen Embedding）
DASHSCOPE_API_KEY=your_dashscope_api_key
```

### 2. 获取 API Key

- **DeepSeek API Key**: https://platform.deepseek.com/
- **DashScope API Key**: https://dashscope.console.aliyun.com/

## 安装依赖

```bash
# 安装所有依赖
pip install -r requirements.txt
```

主要依赖包括：
- `chromadb`: 向量数据库
- `langchain-chroma`: LangChain 集成
- `dashscope`: 阿里云 DashScope SDK
- `pypdf`: PDF 处理
- `python-docx`: Word 文档处理

## 构建知识库

### 方法 1: 使用命令行脚本

```bash
# 1. 准备文档
# 将法律文书、案例等文档放入 data/knowledge_base/ 目录

# 2. 构建知识库（使用默认目录）
python build_knowledge_base.py

# 3. 从指定目录构建
python build_knowledge_base.py --directory /path/to/docs

# 4. 清空已有数据后重新构建
python build_knowledge_base.py --clear

# 5. 只处理 PDF 文件
python build_knowledge_base.py --extensions .pdf

# 6. 查看帮助
python build_knowledge_base.py --help
```

### 方法 2: 使用 API 接口

```bash
# 启动 API 服务
python api_qwen_agent.py

# 使用 API 构建知识库
curl -X POST "http://localhost:8003/agent/knowledge_base/build" \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "./data/knowledge_base",
    "clear_existing": false,
    "recursive": true
  }'
```

### 方法 3: 使用 Python 代码

```python
from agent_system.rag import KnowledgeBase
from agent_system.tools import Qwen25VLTools

# 初始化
vl_tools = Qwen25VLTools()
kb = KnowledgeBase(vl_tools=vl_tools)

# 从目录构建
result = kb.build_from_directory(
    directory="./data/knowledge_base",
    recursive=True,
    clear_existing=False
)

print(f"构建完成，共 {result['total_count']} 个文档块")
```

## 使用 RAG Agent

### 方法 1: 通过 API

```bash
# 启动服务
python api_qwen_agent.py

# 使用 RAG 处理任务
curl -X POST "http://localhost:8003/agent/process_task" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "请告诉我合同纠纷案件的处理流程",
    "use_rag": true
  }'
```

### 方法 2: 使用 Python 代码

```python
from langchain_openai import ChatOpenAI
from agent_system.core import Agent
from agent_system.rag import KnowledgeBase
from agent_system.tools import Qwen25VLTools, finish_tool, create_rag_search_tool
from langchain_core.tools import Tool

# 初始化 LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    base_url="https://api.deepseek.com"
)

# 初始化知识库
vl_tools = Qwen25VLTools()
kb = KnowledgeBase(vl_tools=vl_tools)

# 创建工具
extract_tool = Tool(
    name="extract_legal_key_info",
    description="提取法律文书关键信息",
    func=lambda path: str(vl_tools.extract_key_info(file_path=path))
)
rag_tool = create_rag_search_tool(kb)
tools = [extract_tool, rag_tool, finish_tool]

# 创建 RAG Agent
agent = Agent(
    llm=llm,
    tools=tools,
    knowledge_base=kb,
    use_rag=True  # 启用 RAG
)

# 执行任务
result = agent.run("请告诉我合同纠纷案件的处理流程")
print(result)
```

## API 接口说明

### 1. 构建知识库

**POST** `/agent/knowledge_base/build`

请求体：
```json
{
  "directory": "./data/knowledge_base",  // 可选，文档目录
  "file_paths": ["file1.pdf", "file2.txt"],  // 可选，文件列表
  "clear_existing": false,  // 是否清空已有数据
  "recursive": true  // 是否递归处理子目录
}
```

### 2. 查询知识库信息

**GET** `/agent/knowledge_base/info`

响应：
```json
{
  "success": true,
  "collection_name": "legal_documents",
  "total_documents": 150,
  "embedding_model": "text-embedding-v3"
}
```

### 3. 检索知识库

**POST** `/agent/knowledge_base/search`

请求体：
```json
{
  "query": "合同纠纷",
  "top_k": 3
}
```

### 4. 清空知识库

**DELETE** `/agent/knowledge_base/clear`

### 5. 处理任务（支持 RAG）

**POST** `/agent/process_task`

请求体：
```json
{
  "task": "请告诉我合同纠纷案件的处理流程",
  "use_rag": true,  // 是否使用 RAG
  "temperature": 0.0
}
```

## 测试验证

运行测试脚本验证 RAG 功能：

```bash
python test_rag_agent.py
```

测试内容包括：
1. 知识库基础功能测试
2. RAG Agent 功能测试
3. 对比测试（有/无 RAG）

## 配置说明

在 `agent_system/config/settings.py` 中可以调整 RAG 相关配置：

```python
# Qwen Embedding 模型配置
QWEN_EMBEDDING_MODEL = "text-embedding-v3"
QWEN_EMBEDDING_DIMENSION = 1024

# Chroma 向量数据库配置
CHROMA_PERSIST_DIR = "./data/chroma_db"
CHROMA_COLLECTION_NAME = "legal_documents"

# RAG 检索配置
RAG_TOP_K = 3  # 检索返回的文档数量
RAG_SIMILARITY_THRESHOLD = 0.7  # 相似度阈值
RAG_CHUNK_SIZE = 800  # 文本分块大小
RAG_CHUNK_OVERLAP = 100  # 文本分块重叠大小
```

## 常见问题

### Q1: 知识库为空怎么办？

A: 请先将文档放入 `data/knowledge_base/` 目录，然后运行：
```bash
python build_knowledge_base.py
```

### Q2: DASHSCOPE_API_KEY 未设置

A: 在 `.env` 文件中添加：
```
DASHSCOPE_API_KEY=your_api_key
```

### Q3: PDF 文本提取失败

A: 系统会自动尝试两种方式：
1. 使用 Qwen2.5-VL 的 OCR（推荐，效果更好）
2. 使用 pypdf 提取（备用方案）

确保 Qwen2.5-VL 服务正常运行。

### Q4: 如何提高检索准确度？

A: 可以调整以下参数：
- 增加 `RAG_TOP_K`：返回更多相关文档
- 调整 `RAG_CHUNK_SIZE`：更小的块可能更精确
- 提高 `RAG_SIMILARITY_THRESHOLD`：只返回高相似度结果

### Q5: 知识库更新后需要重启服务吗？

A: 不需要。知识库是持久化的，更新后立即生效。

### Q6: 如何查看知识库中有哪些文档？

A: 使用 API 接口：
```bash
curl http://localhost:8003/agent/knowledge_base/info
```

## 工作原理

### RAG 流程

1. **任务接收**：用户提交任务
2. **知识检索**：根据任务描述检索相关文档（top_k=3）
3. **上下文注入**：将检索到的知识注入到 Agent 的记忆中
4. **增强推理**：LLM 基于任务和相关知识进行推理
5. **工具执行**：执行必要的工具调用
6. **结果返回**：返回最终答案

### 两种使用方式

1. **自动检索**（推荐）
   - 任务开始前自动检索
   - 无需 Agent 主动调用
   - 适合大多数场景

2. **工具调用**
   - Agent 可主动调用 `search_knowledge_base` 工具
   - 更灵活，可多次检索
   - 适合复杂任务

## 性能优化建议

1. **文档质量**：确保文档内容清晰、结构化
2. **合理分块**：根据文档特点调整 chunk_size
3. **定期更新**：及时添加新的案例和文档
4. **监控效果**：通过对比测试评估 RAG 效果

## 技术架构

```
┌─────────────────────────────────────────────────────────┐
│                      用户任务                            │
└─────────────────────┬───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│                   RAG Agent                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  1. 知识检索 (Qwen Embedding)                     │  │
│  │  2. 上下文增强                                     │  │
│  │  3. LLM 推理 (DeepSeek)                           │  │
│  │  4. 工具执行                                       │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
   ┌────────┐   ┌─────────┐   ┌────────┐
   │ Chroma │   │  Qwen   │   │  工具  │
   │ 向量库 │   │ Embed   │   │  集合  │
   └────────┘   └─────────┘   └────────┘
```

## 更多资源

- [LangChain 文档](https://python.langchain.com/)
- [Chroma 文档](https://docs.trychroma.com/)
- [DashScope 文档](https://help.aliyun.com/zh/dashscope/)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)

---

如有问题或建议，请提交 Issue 或 Pull Request。



