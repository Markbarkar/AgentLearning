# Qwen2.5-VL Agent 系统

基于 Qwen2.5-VL 和 DeepSeek-V3 的智能法律文书处理 Agent 系统，支持 RAG（检索增强生成）功能。

## 🌟 主要特性

### 核心功能

- ✅ **法律文书处理**: 提取关键信息、OCR 识别、表单填写
- ✅ **RAG 增强**: 从历史文档中检索相关知识，提升回答质量
- ✅ **多模态理解**: 基于 Qwen2.5-VL 的图文理解能力
- ✅ **智能推理**: 使用 DeepSeek-V3 进行复杂推理
- ✅ **工具调用**: 丰富的工具集，支持多种任务
- ✅ **API 服务**: 完整的 HTTP API 接口

### RAG 功能（新增）

- 🔍 **智能检索**: 自动从知识库检索相关文档
- 📚 **多格式支持**: PDF、Word、文本、Markdown
- 🧠 **上下文增强**: 将检索结果注入到 Agent 推理过程
- 🎯 **高精度**: 使用 Qwen Embedding 模型，优秀的中文支持
- 💾 **持久化**: 基于 Chroma 向量数据库，本地存储

## 🚀 快速开始

### 1. 环境配置

创建 `.env` 文件：

```bash
# DeepSeek API Key
OPENAI_API_KEY=sk-your-deepseek-key

# DashScope API Key（用于 Qwen Embedding）
DASHSCOPE_API_KEY=sk-your-dashscope-key
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 构建知识库（可选，启用 RAG 功能）

```bash
# 将文档放入 data/knowledge_base/ 目录
mkdir -p data/knowledge_base
cp your_docs/* data/knowledge_base/

# 构建知识库
python build_knowledge_base.py
```

### 4. 启动服务

```bash
python api_qwen_agent.py
```

服务将在 `http://localhost:8003` 启动。

### 5. 使用 API

```bash
# 处理任务（使用 RAG）
curl -X POST http://localhost:8003/agent/process_task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "请告诉我合同纠纷案件的处理流程",
    "use_rag": true
  }'

# 查询知识库信息
curl http://localhost:8003/agent/knowledge_base/info

# 检索知识库
curl -X POST http://localhost:8003/agent/knowledge_base/search \
  -H "Content-Type: application/json" \
  -d '{"query": "合同纠纷", "top_k": 3}'
```

## 📖 文档

- **快速开始**: [RAG_QUICKSTART.md](RAG_QUICKSTART.md) - 5 分钟快速上手
- **使用指南**: [RAG_USAGE.md](RAG_USAGE.md) - 详细的 RAG 功能说明
- **实施总结**: [RAG_IMPLEMENTATION_SUMMARY.md](RAG_IMPLEMENTATION_SUMMARY.md) - 技术实现细节

## 🛠️ 项目结构

```
AgentLearning/
├── agent_system/              # Agent 核心系统
│   ├── config/               # 配置管理
│   ├── core/                 # Agent 核心（支持 RAG）
│   ├── models/               # 数据模型
│   ├── prompts/              # 提示词模板
│   ├── tools/                # 工具集（包含 RAG 工具）
│   ├── rag/                  # RAG 模块（新增）
│   │   ├── embeddings.py    # Qwen Embedding 封装
│   │   ├── vector_store.py  # 向量数据库管理
│   │   ├── document_processor.py  # 文档处理
│   │   └── knowledge_base.py      # 知识库管理
│   └── utils/                # 工具函数
├── data/                     # 数据目录
│   ├── knowledge_base/       # 知识库文档
│   └── chroma_db/           # 向量数据库（自动创建）
├── api_qwen_agent.py         # API 服务（支持 RAG）
├── build_knowledge_base.py   # 知识库构建脚本
├── test_rag_agent.py         # RAG 测试脚本
├── requirements.txt          # 依赖列表
└── README.md                 # 本文档
```

## 🔧 核心工具

### 法律文书处理工具

1. **extract_legal_key_info**: 提取关键信息（案号、法院、当事人等）
2. **extract_document_text**: OCR 提取全文
3. **annotate_legal_pdf**: 标注文档，提取字段位置
4. **recognize_form**: 识别表单和表格

### RAG 工具（新增）

5. **search_knowledge_base**: 检索知识库，查找相关文档

## 📊 API 接口

### 任务处理

- `POST /agent/process_task` - 处理用户任务
  - 参数: `task`, `file_path`, `temperature`, `use_rag`

### 知识库管理（新增）

- `POST /agent/knowledge_base/build` - 构建/更新知识库
- `GET /agent/knowledge_base/info` - 查询知识库信息
- `POST /agent/knowledge_base/search` - 检索知识库
- `DELETE /agent/knowledge_base/clear` - 清空知识库

### 工具查询

- `GET /agent/tools` - 获取可用工具列表

## 🧪 测试

```bash
# 运行 RAG 功能测试
python test_rag_agent.py
```

测试包括：
1. 知识库基础功能
2. RAG Agent 功能
3. 对比测试（有/无 RAG）

## ⚙️ 配置

在 `agent_system/config/settings.py` 中配置：

```python
# LLM 配置
LLM_MODEL = "deepseek-chat"
LLM_TEMPERATURE = 0
LLM_BASE_URL = "https://api.deepseek.com"

# RAG 配置
RAG_TOP_K = 3                    # 检索文档数量
RAG_SIMILARITY_THRESHOLD = 0.7   # 相似度阈值
RAG_CHUNK_SIZE = 800             # 文本分块大小
RAG_CHUNK_OVERLAP = 100          # 分块重叠大小

# Qwen Embedding 配置
QWEN_EMBEDDING_MODEL = "text-embedding-v3"
```

## 🎯 使用场景

### 1. 法律文书信息提取

```python
# 提取 PDF 中的关键信息
curl -X POST http://localhost:8003/agent/process_task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "请提取文档中的案号、法院和当事人信息",
    "file_path": "/path/to/document.pdf"
  }'
```

### 2. 基于知识的问答（RAG）

```python
# 询问法律问题，自动检索相关案例
curl -X POST http://localhost:8003/agent/process_task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "合同纠纷案件的诉讼时效是多久？",
    "use_rag": true
  }'
```

### 3. 表单识别和填写

```python
# 识别并填写法律表单
curl -X POST http://localhost:8003/agent/process_task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "识别这个案件受理表并提取信息",
    "file_path": "/path/to/form.pdf"
  }'
```

## 🔍 RAG 工作原理

```
用户任务
   ↓
任务分析
   ↓
知识检索 (Qwen Embedding + Chroma)
   ↓
上下文增强（注入相关知识）
   ↓
LLM 推理 (DeepSeek)
   ↓
工具执行（如需要）
   ↓
返回结果
```

## 📈 性能指标

- **检索速度**: <1秒
- **知识库构建**: ~5-10秒/页（PDF OCR）
- **向量化**: ~0.1秒/文本块
- **并发支持**: 是
- **持久化**: 自动

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) - 多模态理解
- [DeepSeek-V3](https://platform.deepseek.com/) - 大语言模型
- [LangChain](https://python.langchain.com/) - Agent 框架
- [Chroma](https://www.trychroma.com/) - 向量数据库

## 📞 联系方式

如有问题或建议，请提交 Issue。

---

**版本**: 1.0.0 (支持 RAG)  
**更新日期**: 2025-11-18



