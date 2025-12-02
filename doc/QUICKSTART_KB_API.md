# 知识库 API 快速启动指南

⚡ 5分钟快速上手知识库构建 API

---

## 🎯 目标

通过本指南，您将学会：
1. 启动知识库 API 服务
2. 构建您的第一个知识库
3. 检索知识库内容

---

## 📋 前置条件

- ✅ Python 3.8+
- ✅ 已安装依赖包
- ✅ Qwen2.5-VL 服务运行中

---

## 🚀 三步启动

### 第 1 步：准备文档（30秒）

将文档放入知识库目录：

```bash
# 创建目录（如果不存在）
mkdir -p /projects/AgentLearning/data/knowledge_base

# 复制您的文档到该目录
cp /path/to/your/documents/*.pdf /projects/AgentLearning/data/knowledge_base/
```

支持的文件格式：
- PDF 文件
- TXT 文本文件
- DOCX Word 文档
- Markdown 文件

### 第 2 步：启动服务（30秒）

```bash
# 进入项目目录
cd /projects/AgentLearning

# 启动 API 服务
python3 api_knowledge_base.py
```

看到这个表示成功：
```
======================================================================
🚀 启动知识库构建 API 服务
======================================================================

📁 默认文档目录: /projects/AgentLearning/data/knowledge_base
💾 向量数据库目录: /projects/AgentLearning/data/chroma_db
📦 集合名称: legal_documents

🌐 API 服务地址:
   http://127.0.0.1:8004
   http://0.0.0.0:8004

📚 API 文档:
   http://127.0.0.1:8004/docs
   http://127.0.0.1:8004/redoc

======================================================================
```

### 第 3 步：构建知识库（2-5分钟）

**选项 A：使用 Web 管理面板**（推荐）

1. 双击打开 `kb_dashboard.html`
2. 查看知识库状态
3. 点击"从目录构建"按钮
4. 等待构建完成

**选项 B：使用测试客户端**

```bash
# 新开一个终端
python3 test_kb_api.py

# 选择 4 (build_dir)
# 按提示操作
```

**选项 C：使用 curl**

```bash
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory": null,
    "recursive": true,
    "clear_existing": false
  }'
```

---

## 🎉 成功！现在可以...

### 查看知识库信息

**Web 面板**：
- 打开 `kb_dashboard.html`
- 查看"知识库信息"卡片

**curl**：
```bash
curl http://127.0.0.1:8004/knowledge_base/info
```

**预期输出**：
```json
{
  "collection_name": "legal_documents",
  "persist_directory": "/projects/AgentLearning/data/chroma_db",
  "total_documents": 25,
  "embedding_model": "bge-large-zh-v1.5",
  "status": "ready"
}
```

### 检索知识库

**Web 面板**：
1. 在"知识库检索"卡片中输入查询
2. 点击"检索"按钮
3. 查看结果

**curl**：
```bash
curl -X POST http://127.0.0.1:8004/knowledge_base/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "沈俊华的案件信息",
    "top_k": 3
  }'
```

**预期输出**：
```json
{
  "success": true,
  "query": "沈俊华的案件信息",
  "total_results": 3,
  "results": [
    {
      "content": "沈俊华，男，1975年生...",
      "metadata": {
        "source": "/path/to/document.pdf",
        "page": 1
      },
      "similarity": 0.92
    }
  ]
}
```

---

## 📝 常用操作

### 增量添加文档

```bash
# 1. 添加新文档到目录
cp /path/to/new_document.pdf /projects/AgentLearning/data/knowledge_base/

# 2. 增量构建（不清空已有数据）
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": false}'
```

### 重建知识库

```bash
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": true}'
```

### 上传单个文件

```bash
curl -X POST http://127.0.0.1:8004/knowledge_base/build/upload \
  -F "files=@/path/to/document.pdf"
```

---

## 🔧 故障排查

### 问题 1：服务启动失败

**错误**：`ModuleNotFoundError: No module named 'fastapi'`

**解决**：
```bash
pip install fastapi uvicorn python-multipart
```

### 问题 2：知识库构建失败

**错误**：`目录不存在`

**解决**：
```bash
mkdir -p /projects/AgentLearning/data/knowledge_base
# 然后添加文档
```

### 问题 3：检索无结果

**原因**：知识库可能为空

**检查**：
```bash
curl http://127.0.0.1:8004/knowledge_base/info
```

如果 `total_documents` 为 0，需要先构建知识库。

### 问题 4：端口被占用

**错误**：`Address already in use`

**解决**：
```bash
# 修改 api_knowledge_base.py 中的端口
# 将 port=8004 改为其他端口，如 port=8005
```

---

## 📚 下一步

- 📖 阅读 [KB_API_GUIDE.md](KB_API_GUIDE.md) 了解所有 API
- 🔍 访问 http://127.0.0.1:8004/docs 查看交互式文档
- 🛠️ 集成到您的应用中
- 🤖 配合 Agent 使用 RAG 功能

---

## 💡 小贴士

1. **大批量文档**：使用异步模式（`async_mode=true`）
2. **查询优化**：使用具体、描述性的查询语句
3. **定期备份**：备份 `data/chroma_db` 目录
4. **监控状态**：使用 Web 面板实时查看

---

## 🆘 需要帮助？

- 查看完整文档：[KB_API_README.md](KB_API_README.md)
- 查看 API 文档：http://127.0.0.1:8004/docs
- 使用测试客户端：`python3 test_kb_api.py`

---

**祝您使用愉快！** 🎉

