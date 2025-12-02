# 知识库 API 使用说明

## 📦 新增文件清单

本次开发为您创建了以下文件：

### 🔧 核心文件

1. **`api_knowledge_base.py`** - API 服务主程序
   - RESTful API 服务
   - 支持同步/异步构建
   - 提供 12 个 API 端点
   - 启动方式: `python3 api_knowledge_base.py`

2. **`test_kb_api.py`** - 交互式测试客户端
   - 命令行交互界面
   - 测试所有 API 功能
   - 启动方式: `python3 test_kb_api.py`

3. **`kb_dashboard.html`** - Web 可视化管理面板
   - 实时状态展示
   - 可视化操作界面
   - 使用方式: 双击打开或浏览器打开

4. **`start_kb_api.sh`** - 服务启动脚本
   - 自动检查依赖
   - 一键启动服务
   - 使用方式: `./start_kb_api.sh`

### 📚 文档文件

5. **`KB_API_README.md`** - 系统总览文档
   - 功能特性介绍
   - 系统架构说明
   - 完整使用指南
   - 最佳实践和常见问题

6. **`KB_API_GUIDE.md`** - 详细 API 使用指南
   - 所有 API 端点详细文档
   - 请求/响应示例
   - Python/JavaScript/curl 代码示例

7. **`QUICKSTART_KB_API.md`** - 快速启动指南
   - 5分钟快速上手
   - 三步启动流程
   - 常用操作示例
   - 故障排查指南

8. **`KB_API_SUMMARY.md`** - 开发总结文档
   - 功能特性总结
   - 技术架构说明
   - 代码实现细节

9. **`知识库API使用说明.md`** - 本文件
   - 文件清单
   - 快速开始指南

---

## 🚀 快速开始（3步）

### 第 1 步：准备文档

将您的文档（PDF、TXT、DOCX）放入：

```bash
/projects/AgentLearning/data/knowledge_base/
```

### 第 2 步：启动服务

```bash
cd /projects/AgentLearning
python3 api_knowledge_base.py
```

看到以下信息表示启动成功：

```
🚀 启动知识库构建 API 服务
🌐 API 服务地址: http://127.0.0.1:8004
📚 API 文档: http://127.0.0.1:8004/docs
```

### 第 3 步：构建知识库

**选择一种方式：**

#### 方式 A：Web 管理面板（最简单）

1. 双击打开 `kb_dashboard.html`
2. 点击"从目录构建"按钮
3. 等待构建完成

#### 方式 B：测试客户端

```bash
# 新开一个终端
python3 test_kb_api.py
# 选择 4 (build_dir)
```

#### 方式 C：API 调用

```bash
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": false}'
```

---

## 🎯 主要功能

### 1. 构建知识库

- ✅ 从目录批量构建
- ✅ 从文件列表构建
- ✅ 上传文件构建
- ✅ 支持同步/异步执行

### 2. 查询知识库

- ✅ 语义检索
- ✅ 相似度评分
- ✅ 灵活的参数配置

### 3. 管理知识库

- ✅ 查看知识库信息
- ✅ 查看详细统计
- ✅ 清空知识库
- ✅ 任务状态追踪

---

## 📱 使用界面

### Web 管理面板

打开 `kb_dashboard.html` 可以看到：

```
┌─────────────────────────────────────────┐
│  📚 知识库构建管理面板                    │
│  法律文书知识库构建、管理和查询系统        │
│  ✓ 服务正常                              │
├─────────────────────────────────────────┤
│  📊 知识库信息  │  ⚙️ 系统配置  │  🔨 构建操作  │
│  - 集合名称     │  - 文档目录   │  - 文档目录     │
│  - 文档数量: 25 │  - 数据库目录 │  ☑ 递归子目录   │
│  - 嵌入模型     │  - 集合名称   │  ☐ 清空已有数据 │
│  - 状态: 就绪   │              │  ☐ 异步执行     │
│                │              │  [🔨 从目录构建] │
├─────────────────────────────────────────┤
│  🔍 知识库检索                            │
│  查询文本: [___________________________] │
│  结果数量: [3___]                        │
│  [🔍 检索]  [🗑️ 清空知识库]              │
└─────────────────────────────────────────┘
```

### 测试客户端界面

运行 `python3 test_kb_api.py` 可以看到：

```
══════════════════════════════════════════════
知识库 API 测试客户端
══════════════════════════════════════════════

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

请输入命令（数字或名称）: 
```

---

## 🔌 API 端点一览

### 基础信息
- `GET /` - 服务信息
- `GET /health` - 健康检查
- `GET /config` - 配置信息

### 知识库构建
- `POST /knowledge_base/build/directory` - 从目录构建
- `POST /knowledge_base/build/files` - 从文件列表构建
- `POST /knowledge_base/build/upload` - 上传文件构建

### 任务管理
- `GET /knowledge_base/tasks/{task_id}` - 查询任务状态
- `GET /knowledge_base/tasks` - 列出所有任务

### 知识库查询
- `GET /knowledge_base/info` - 知识库信息
- `GET /knowledge_base/stats` - 详细统计
- `POST /knowledge_base/search` - 检索知识库

### 知识库管理
- `DELETE /knowledge_base/clear` - 清空知识库

---

## 💡 使用示例

### Python 示例

```python
import requests

BASE_URL = "http://127.0.0.1:8004"

# 1. 构建知识库
response = requests.post(
    f"{BASE_URL}/knowledge_base/build/directory",
    json={"clear_existing": False}
)
print(response.json())

# 2. 检索知识库
response = requests.post(
    f"{BASE_URL}/knowledge_base/search",
    json={
        "query": "沈俊华的案件信息",
        "top_k": 3
    }
)
print(response.json())

# 3. 查看知识库信息
response = requests.get(f"{BASE_URL}/knowledge_base/info")
print(response.json())
```

### curl 示例

```bash
# 1. 健康检查
curl http://127.0.0.1:8004/health

# 2. 构建知识库
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": false}'

# 3. 检索知识库
curl -X POST http://127.0.0.1:8004/knowledge_base/search \
  -H "Content-Type: application/json" \
  -d '{"query": "沈俊华", "top_k": 3}'

# 4. 查看信息
curl http://127.0.0.1:8004/knowledge_base/info
```

---

## 📚 文档阅读顺序

**新手推荐**：
1. 先读本文件（`知识库API使用说明.md`）- 了解整体
2. 再读 `QUICKSTART_KB_API.md` - 快速上手
3. 查阅 `KB_API_GUIDE.md` - 深入学习

**进阶用户**：
1. `KB_API_README.md` - 系统总览
2. `KB_API_GUIDE.md` - API 详细文档
3. `KB_API_SUMMARY.md` - 技术实现

**在线文档**：
- http://127.0.0.1:8004/docs - Swagger UI（交互式）
- http://127.0.0.1:8004/redoc - ReDoc（精美文档）

---

## ❓ 常见问题

### Q1: 如何启动服务？

**A**: 三种方式任选一种：

```bash
# 方式 1: 使用启动脚本
./start_kb_api.sh

# 方式 2: 直接运行
python3 api_knowledge_base.py

# 方式 3: 使用 uvicorn
uvicorn api_knowledge_base:app --host 0.0.0.0 --port 8004
```

### Q2: 如何查看构建进度？

**A**: 

1. **同步模式**：立即返回结果
2. **异步模式**：
   - Web 面板自动刷新
   - 调用 `GET /knowledge_base/tasks/{task_id}`
   - 使用测试客户端实时监控

### Q3: 支持哪些文件格式？

**A**: 
- PDF 文件（通过 OCR）
- TXT 文本文件
- DOCX Word 文档
- Markdown 文件

### Q4: 如何增量添加文档？

**A**: 

```bash
# 设置 clear_existing=false
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": false}'
```

### Q5: 如何清空知识库？

**A**: 

```bash
# 方式 1: API 调用
curl -X DELETE http://127.0.0.1:8004/knowledge_base/clear

# 方式 2: Web 面板
# 点击"清空知识库"按钮

# 方式 3: 测试客户端
# 选择命令 10 (clear)
```

### Q6: 端口被占用怎么办？

**A**: 修改 `api_knowledge_base.py` 中的端口号：

```python
# 找到这一行
uvicorn.run(app, host="0.0.0.0", port=8004)

# 改为其他端口，如
uvicorn.run(app, host="0.0.0.0", port=8005)
```

---

## 🛠️ 故障排查

### 服务无法启动

1. 检查依赖是否安装：
   ```bash
   pip install fastapi uvicorn python-multipart
   ```

2. 检查端口是否被占用：
   ```bash
   lsof -i :8004
   ```

### 构建失败

1. 检查文档目录是否存在
2. 确认 Qwen2.5-VL 服务运行正常
3. 查看服务日志错误信息

### 检索无结果

1. 确认知识库有数据：
   ```bash
   curl http://127.0.0.1:8004/knowledge_base/info
   ```

2. 降低相似度阈值
3. 优化查询语句

---

## 🎯 下一步

1. **启动服务**
   ```bash
   python3 api_knowledge_base.py
   ```

2. **打开管理面板**
   ```bash
   # 双击打开
   kb_dashboard.html
   ```

3. **构建知识库**
   - 按照面板提示操作

4. **开始使用**
   - 检索知识库
   - 集成到您的应用

---

## 📖 详细文档

- **快速入门**: `QUICKSTART_KB_API.md`
- **完整指南**: `KB_API_README.md`
- **API 文档**: `KB_API_GUIDE.md`
- **技术总结**: `KB_API_SUMMARY.md`
- **在线文档**: http://127.0.0.1:8004/docs

---

## 💬 技术支持

如需帮助，请：
1. 查看相关文档
2. 访问在线 API 文档
3. 使用测试客户端验证

---

**祝您使用愉快！** 🎉

---

**创建日期**: 2023-12-02  
**版本**: 1.0.0

