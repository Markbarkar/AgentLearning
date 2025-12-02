# 知识库构建 API 系统

一个完整的法律文书知识库构建、管理和查询系统，提供 RESTful API 接口和可视化管理界面。

## 📋 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [文件说明](#文件说明)
- [使用指南](#使用指南)
- [API 文档](#api-文档)
- [常见问题](#常见问题)

---

## ✨ 功能特性

### 核心功能

- ✅ **多种构建方式**
  - 从目录批量构建
  - 从文件列表构建
  - 上传文件构建
  
- ✅ **同步/异步执行**
  - 同步模式：小批量文档，即时返回结果
  - 异步模式：大批量文档，后台处理，任务状态追踪
  
- ✅ **实时进度展示**
  - 任务状态查询
  - 构建进度追踪
  - 详细统计信息
  
- ✅ **知识库管理**
  - 查看知识库信息
  - 检索知识库内容
  - 清空知识库数据
  
- ✅ **可视化界面**
  - Web 管理面板
  - 实时状态展示
  - 交互式操作

### 技术特性

- 🚀 基于 FastAPI，高性能异步处理
- 🔍 使用 ChromaDB 向量数据库
- 🤖 集成 Qwen2.5-VL 进行 PDF OCR
- 📊 支持 BGE 大规模中文嵌入模型
- 🌐 RESTful API 设计
- 📱 响应式 Web 界面
- 🔒 CORS 跨域支持

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    客户端层                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │ Web 管理面板│  │测试客户端    │  │  HTTP 客户端 │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST API
┌────────────────────▼────────────────────────────────────┐
│                  API 服务层                              │
│  ┌──────────────────────────────────────────────────┐  │
│  │  FastAPI Application (api_knowledge_base.py)     │  │
│  │  - 构建端点 | 查询端点 | 管理端点                  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  业务逻辑层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  KnowledgeBase│  │DocumentProcessor│  │VectorStore │  │
│  │  知识库管理   │  │  文档处理器    │  │  向量存储   │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                  数据存储层                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │
│  │  ChromaDB    │  │Qwen2.5-VL API│  │  BGE Model  │  │
│  │  向量数据库   │  │  OCR 服务    │  │  嵌入模型   │  │
│  └──────────────┘  └──────────────┘  └─────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 1. 环境准备

**安装依赖**：
```bash
pip install fastapi uvicorn python-multipart
pip install chromadb langchain-community
pip install requests
```

**确认服务**：
- Qwen2.5-VL 服务已启动
- BGE 嵌入模型可访问

### 2. 准备文档

将要构建知识库的文档放入：
```bash
/projects/AgentLearning/data/knowledge_base/
```

支持格式：PDF, TXT, DOCX, Markdown

### 3. 启动 API 服务

**方式一：使用启动脚本**
```bash
./start_kb_api.sh
```

**方式二：直接运行**
```bash
python3 api_knowledge_base.py
```

**方式三：指定端口**
```bash
uvicorn api_knowledge_base:app --host 0.0.0.0 --port 8004
```

### 4. 访问服务

- **API 服务**: http://127.0.0.1:8004
- **API 文档**: http://127.0.0.1:8004/docs
- **管理面板**: 打开 `kb_dashboard.html`

### 5. 构建知识库

**使用 Web 管理面板**：
1. 打开 `kb_dashboard.html`
2. 查看知识库信息
3. 点击"从目录构建"
4. 查看构建进度和结果

**使用测试客户端**：
```bash
python3 test_kb_api.py
```

**使用 curl**：
```bash
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": false}'
```

---

## 📁 文件说明

### 核心文件

| 文件 | 说明 |
|------|------|
| `api_knowledge_base.py` | API 服务主程序 |
| `build_knowledge_base.py` | 命令行构建工具（原始） |
| `kb_dashboard.html` | Web 管理面板 |
| `test_kb_api.py` | 交互式测试客户端 |
| `start_kb_api.sh` | 服务启动脚本 |

### 文档文件

| 文件 | 说明 |
|------|------|
| `KB_API_GUIDE.md` | 详细 API 使用指南 |
| `KB_API_README.md` | 系统总览（本文件） |
| `RAG_USAGE.md` | RAG 系统使用文档 |
| `RAG_QUICKSTART.md` | RAG 快速入门 |

### 数据目录

| 目录 | 说明 |
|------|------|
| `data/knowledge_base/` | 文档存放目录 |
| `data/chroma_db/` | 向量数据库存储 |

---

## 📖 使用指南

### 方式一：Web 管理面板（推荐）

1. **启动服务**
   ```bash
   python3 api_knowledge_base.py
   ```

2. **打开管理面板**
   - 双击打开 `kb_dashboard.html`
   - 或在浏览器中打开该文件

3. **操作知识库**
   - 查看当前状态和统计信息
   - 配置构建参数
   - 执行构建任务
   - 检索知识库内容

### 方式二：测试客户端

1. **启动客户端**
   ```bash
   python3 test_kb_api.py
   ```

2. **选择操作**
   ```
   可用命令:
     1. 健康检查 (health)
     2. 获取知识库信息 (info)
     3. 获取详细统计 (stats)
     4. 从目录构建 (build_dir)
     5. 从文件列表构建 (build_files)
     6. 上传文件构建 (upload)
     7. 检索知识库 (search)
     8. 查看任务状态 (task)
     9. 列出所有任务 (tasks)
    10. 清空知识库 (clear)
    11. 获取配置 (config)
     0. 退出 (exit)
   ```

3. **按提示操作**

### 方式三：编程调用

**Python 示例**：
```python
import requests

BASE_URL = "http://127.0.0.1:8004"

# 构建知识库
response = requests.post(
    f"{BASE_URL}/knowledge_base/build/directory",
    json={"clear_existing": False}
)
print(response.json())

# 检索知识库
response = requests.post(
    f"{BASE_URL}/knowledge_base/search",
    json={
        "query": "沈俊华的案件信息",
        "top_k": 3
    }
)
print(response.json())
```

**JavaScript 示例**：
```javascript
// 构建知识库
fetch('http://127.0.0.1:8004/knowledge_base/build/directory', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({clear_existing: false})
})
.then(r => r.json())
.then(data => console.log(data));

// 检索知识库
fetch('http://127.0.0.1:8004/knowledge_base/search', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: "沈俊华的案件信息",
    top_k: 3
  })
})
.then(r => r.json())
.then(data => console.log(data));
```

---

## 🔌 API 文档

### 主要端点

#### 基础信息
- `GET /` - 服务信息
- `GET /health` - 健康检查
- `GET /config` - 配置信息

#### 知识库构建
- `POST /knowledge_base/build/directory` - 从目录构建
- `POST /knowledge_base/build/files` - 从文件列表构建
- `POST /knowledge_base/build/upload` - 上传文件构建

#### 任务管理
- `GET /knowledge_base/tasks/{task_id}` - 查询任务状态
- `GET /knowledge_base/tasks` - 列出所有任务

#### 知识库查询
- `GET /knowledge_base/info` - 知识库信息
- `GET /knowledge_base/stats` - 详细统计
- `POST /knowledge_base/search` - 检索知识库

#### 知识库管理
- `DELETE /knowledge_base/clear` - 清空知识库

### 详细文档

完整的 API 文档请查看：
- **交互式文档**: http://127.0.0.1:8004/docs
- **详细指南**: [KB_API_GUIDE.md](KB_API_GUIDE.md)

---

## 💡 最佳实践

### 1. 构建策略

**首次构建**：
```bash
# 清空已有数据，全新构建
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": true}'
```

**增量更新**：
```bash
# 不清空，添加新文档
curl -X POST http://127.0.0.1:8004/knowledge_base/build/files \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/path/to/new_file.pdf"],
    "clear_existing": false
  }'
```

**大批量构建**：
```bash
# 使用异步模式
curl -X POST "http://127.0.0.1:8004/knowledge_base/build/directory?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"directory": "/large/dir"}'
```

### 2. 文档组织

**推荐目录结构**：
```
data/knowledge_base/
├── 民事案件/
│   ├── 2023/
│   └── 2024/
├── 刑事案件/
└── 行政案件/
```

### 3. 检索优化

- 使用具体、描述性的查询语句
- 根据需求调整 `top_k` 参数
- 设置合理的相似度阈值

### 4. 性能优化

- 批量处理文件而非逐个处理
- 大任务使用异步模式
- 定期清理无用数据

---

## ❓ 常见问题

### Q1: 服务启动失败？

**A**: 检查依赖是否安装完整：
```bash
pip install fastapi uvicorn python-multipart chromadb
```

### Q2: 构建知识库失败？

**A**: 可能原因：
1. 文档目录不存在或为空
2. Qwen2.5-VL 服务未启动
3. 文档格式不支持

解决：
- 检查文档目录和文件
- 确认 VL 服务可用
- 查看服务日志

### Q3: 检索无结果？

**A**: 尝试：
1. 降低相似度阈值
2. 优化查询语句
3. 确认知识库有数据

### Q4: 如何监控构建进度？

**A**: 
- 使用异步模式构建
- 通过 `/knowledge_base/tasks/{task_id}` 查询状态
- 使用 Web 管理面板实时查看

### Q5: 如何备份知识库？

**A**: 备份 ChromaDB 数据目录：
```bash
cp -r data/chroma_db data/chroma_db_backup
```

### Q6: 如何重置知识库？

**A**: 
```bash
curl -X DELETE http://127.0.0.1:8004/knowledge_base/clear
```

或删除数据目录：
```bash
rm -rf data/chroma_db/*
```

---

## 🔧 配置说明

### 环境变量

在 `agent_system/config/settings.py` 中配置：

```python
# 知识库文档目录
KNOWLEDGE_BASE_DOCS_DIR = "/projects/AgentLearning/data/knowledge_base"

# ChromaDB 配置
CHROMA_PERSIST_DIR = "/projects/AgentLearning/data/chroma_db"
CHROMA_COLLECTION_NAME = "legal_documents"

# RAG 配置
RAG_TOP_K = 3
RAG_SIMILARITY_THRESHOLD = 0.5
```

### API 服务配置

在 `api_knowledge_base.py` 中修改：

```python
# 服务端口
uvicorn.run(app, host="0.0.0.0", port=8004)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 使用流程

```
1. 准备文档
   ↓
2. 启动 API 服务
   ↓
3. 构建知识库
   ├─ 同步模式（小批量）
   └─ 异步模式（大批量）
   ↓
4. 监控构建进度
   ├─ 查看任务状态
   └─ 查看知识库信息
   ↓
5. 使用知识库
   ├─ 检索查询
   └─ Agent 集成
   ↓
6. 维护管理
   ├─ 增量更新
   └─ 清空重建
```

---

## 🎯 功能对比

| 功能 | 命令行工具 | API 服务 | Web 管理面板 |
|------|-----------|---------|-------------|
| 构建知识库 | ✅ | ✅ | ✅ |
| 异步执行 | ❌ | ✅ | ✅ |
| 进度监控 | ❌ | ✅ | ✅ |
| 知识库检索 | ❌ | ✅ | ✅ |
| 远程调用 | ❌ | ✅ | ✅ |
| 可视化界面 | ❌ | ❌ | ✅ |
| 编程集成 | ❌ | ✅ | ❌ |

---

## 📚 相关文档

- [KB_API_GUIDE.md](KB_API_GUIDE.md) - 详细 API 使用指南
- [RAG_USAGE.md](RAG_USAGE.md) - RAG 系统使用文档
- [RAG_QUICKSTART.md](RAG_QUICKSTART.md) - RAG 快速入门
- API 文档: http://127.0.0.1:8004/docs

---

## 🤝 贡献

欢迎提交问题和改进建议！

---

## 📄 许可证

本项目遵循 MIT 许可证。

---

**最后更新**: 2023-12-02
**版本**: 1.0.0

