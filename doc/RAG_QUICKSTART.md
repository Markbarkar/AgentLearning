# RAG 功能快速开始指南

## 5 分钟快速上手

### 第 1 步：配置环境变量

创建 `.env` 文件：

```bash
# DeepSeek API Key（用于 LLM）
OPENAI_API_KEY=sk-your-deepseek-key

# DashScope API Key（用于 Qwen Embedding）
DASHSCOPE_API_KEY=sk-your-dashscope-key
```

### 第 2 步：安装依赖

```bash
pip install -r requirements.txt
```

### 第 3 步：准备文档

将法律文书、案例等文档放入 `data/knowledge_base/` 目录：

```bash
mkdir -p data/knowledge_base
# 将你的 PDF、Word、文本文件复制到这个目录
```

### 第 4 步：构建知识库

```bash
python build_knowledge_base.py
```

预期输出：
```
==============================================================
开始构建知识库: data/knowledge_base
==============================================================

找到 5 个文件待处理
[1/5] 处理文件: 案例1.pdf
  ✓ 生成 12 个文档块
[2/5] 处理文件: 案例2.pdf
  ✓ 生成 8 个文档块
...

==============================================================
知识库构建完成
==============================================================
文档块数量: 50
向量数据库总数: 50
==============================================================
```

### 第 5 步：启动 API 服务

```bash
python api_qwen_agent.py
```

### 第 6 步：测试 RAG 功能

#### 方法 1：使用测试脚本

```bash
python test_rag_agent.py
```

#### 方法 2：使用 API

```bash
# 查询知识库信息
curl http://localhost:8003/agent/knowledge_base/info

# 检索知识
curl -X POST http://localhost:8003/agent/knowledge_base/search \
  -H "Content-Type: application/json" \
  -d '{"query": "合同纠纷", "top_k": 3}'

# 使用 RAG 处理任务
curl -X POST http://localhost:8003/agent/process_task \
  -H "Content-Type: application/json" \
  -d '{
    "task": "请告诉我合同纠纷案件的处理流程",
    "use_rag": true
  }'
```

## 完成！

现在你的 Agent 已经具备了 RAG 能力，可以：

✅ 自动检索相关知识
✅ 基于历史案例回答问题
✅ 提供更准确的法律建议

## 下一步

- 📖 阅读完整文档：[RAG_USAGE.md](RAG_USAGE.md)
- 🔧 调整配置参数：`agent_system/config/settings.py`
- 📝 添加更多文档到知识库
- 🧪 运行更多测试验证效果

## 常见问题

**Q: 知识库构建很慢？**
A: PDF OCR 需要时间，这是正常的。可以先用少量文档测试。

**Q: 检索不到相关内容？**
A: 检查知识库是否成功构建，调整相似度阈值参数。

**Q: API Key 错误？**
A: 确保 `.env` 文件在项目根目录，且 Key 正确无误。

## 技术支持

遇到问题？查看详细文档或提交 Issue。



