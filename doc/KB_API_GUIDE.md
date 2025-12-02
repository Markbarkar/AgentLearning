# 知识库构建 API 使用指南

这是一个用于构建和管理法律文书知识库的 RESTful API 服务。

## 目录

- [快速开始](#快速开始)
- [API 端点](#api-端点)
- [使用示例](#使用示例)
- [客户端工具](#客户端工具)
- [最佳实践](#最佳实践)

---

## 快速开始

### 1. 启动服务

**方式一：使用启动脚本**
```bash
./start_kb_api.sh
```

**方式二：直接运行**
```bash
python3 api_knowledge_base.py
```

服务启动后可以访问：
- **API 服务**: http://127.0.0.1:8004
- **交互式文档**: http://127.0.0.1:8004/docs
- **文档（ReDoc）**: http://127.0.0.1:8004/redoc

### 2. 准备文档

将要构建知识库的文档放入目录：
```bash
/projects/AgentLearning/data/knowledge_base/
```

支持的文件格式：
- PDF 文件（通过 OCR 提取）
- TXT 文本文件
- DOCX Word 文档
- Markdown 文件

### 3. 构建知识库

**使用测试客户端**（推荐）：
```bash
python3 test_kb_api.py
```

**使用 curl**：
```bash
curl -X POST "http://127.0.0.1:8004/knowledge_base/build/directory" \
  -H "Content-Type: application/json" \
  -d '{
    "directory": null,
    "recursive": true,
    "clear_existing": false
  }'
```

---

## API 端点

### 基础信息

#### GET `/`
获取服务基本信息

**响应示例**：
```json
{
  "service": "知识库构建 API",
  "version": "1.0.0",
  "description": "法律文书知识库构建、管理和查询服务",
  "docs": "/docs"
}
```

#### GET `/health`
健康检查

**响应示例**：
```json
{
  "status": "healthy",
  "knowledge_base": "initialized",
  "total_documents": 15
}
```

#### GET `/config`
获取当前配置信息

**响应示例**：
```json
{
  "knowledge_base_docs_dir": "/projects/AgentLearning/data/knowledge_base",
  "chroma_persist_dir": "/projects/AgentLearning/data/chroma_db",
  "chroma_collection_name": "legal_documents"
}
```

---

### 知识库构建

#### POST `/knowledge_base/build/directory`
从目录构建知识库

**请求参数**：
- `async_mode` (query, boolean): 是否异步执行（默认 false）

**请求体**：
```json
{
  "directory": "/path/to/documents",  // null 则使用默认目录
  "recursive": true,                   // 是否递归子目录
  "file_extensions": [".pdf", ".txt"], // null 则处理所有支持的格式
  "clear_existing": false              // 是否清空已有数据
}
```

**响应示例（同步模式）**：
```json
{
  "success": true,
  "message": "知识库构建成功",
  "chunks_added": 25,
  "total_count": 50
}
```

**响应示例（异步模式）**：
```json
{
  "success": true,
  "message": "构建任务已创建",
  "task_id": "task_20231202_143022_123456"
}
```

---

#### POST `/knowledge_base/build/files`
从文件列表构建知识库

**请求参数**：
- `async_mode` (query, boolean): 是否异步执行（默认 false）

**请求体**：
```json
{
  "file_paths": [
    "/path/to/file1.pdf",
    "/path/to/file2.txt"
  ],
  "clear_existing": false
}
```

**响应示例**：
```json
{
  "success": true,
  "message": "知识库构建成功",
  "chunks_added": 10,
  "total_count": 60
}
```

---

#### POST `/knowledge_base/build/upload`
上传文件并构建知识库

**请求参数**：
- `clear_existing` (query, boolean): 是否清空已有数据（默认 false）
- `async_mode` (query, boolean): 是否异步执行（默认 false）

**请求体**：
- `files` (multipart/form-data): 文件列表

**使用 curl 示例**：
```bash
curl -X POST "http://127.0.0.1:8004/knowledge_base/build/upload" \
  -F "files=@/path/to/file1.pdf" \
  -F "files=@/path/to/file2.pdf"
```

**响应示例**：
```json
{
  "success": true,
  "message": "上传成功，构建任务已创建",
  "task_id": "task_20231202_143530_789012",
  "files_uploaded": 2
}
```

---

### 任务管理

#### GET `/knowledge_base/tasks/{task_id}`
获取构建任务状态

**路径参数**：
- `task_id` (string): 任务ID

**响应示例（进行中）**：
```json
{
  "task_id": "task_20231202_143022_123456",
  "type": "build_from_directory",
  "status": "processing",
  "message": "正在处理文档...",
  "created_at": "2023-12-02T14:30:22.123456",
  "params": {
    "directory": "/projects/AgentLearning/data/knowledge_base",
    "recursive": true,
    "clear_existing": false
  }
}
```

**响应示例（已完成）**：
```json
{
  "task_id": "task_20231202_143022_123456",
  "type": "build_from_directory",
  "status": "completed",
  "message": "构建成功",
  "created_at": "2023-12-02T14:30:22.123456",
  "completed_at": "2023-12-02T14:31:45.678901",
  "result": {
    "success": true,
    "message": "知识库构建成功",
    "chunks_added": 25,
    "total_count": 50
  }
}
```

**响应示例（失败）**：
```json
{
  "task_id": "task_20231202_143022_123456",
  "type": "build_from_directory",
  "status": "failed",
  "message": "构建失败: 目录不存在",
  "created_at": "2023-12-02T14:30:22.123456",
  "completed_at": "2023-12-02T14:30:25.123456",
  "error": "目录不存在"
}
```

---

#### GET `/knowledge_base/tasks`
列出所有构建任务

**响应示例**：
```json
{
  "total": 3,
  "tasks": [
    {
      "task_id": "task_20231202_143022_123456",
      "type": "build_from_directory",
      "status": "completed",
      "message": "构建成功",
      "created_at": "2023-12-02T14:30:22.123456"
    },
    {
      "task_id": "task_20231202_150000_234567",
      "type": "build_from_files",
      "status": "processing",
      "message": "正在处理 5 个文件...",
      "created_at": "2023-12-02T15:00:00.234567"
    }
  ]
}
```

---

### 知识库信息

#### GET `/knowledge_base/info`
获取知识库基本信息

**响应示例**：
```json
{
  "collection_name": "legal_documents",
  "persist_directory": "/projects/AgentLearning/data/chroma_db",
  "total_documents": 50,
  "embedding_model": "bge-large-zh-v1.5",
  "status": "ready"
}
```

**状态说明**：
- `empty`: 知识库为空
- `initializing`: 知识库初始化中（文档数 < 10）
- `ready`: 知识库就绪

---

#### GET `/knowledge_base/stats`
获取知识库详细统计信息

**响应示例**：
```json
{
  "success": true,
  "collection_name": "legal_documents",
  "persist_directory": "/projects/AgentLearning/data/chroma_db",
  "total_documents": 50,
  "embedding_model": "bge-large-zh-v1.5",
  "storage_info": {
    "persist_directory": "/projects/AgentLearning/data/chroma_db",
    "collection_name": "legal_documents"
  }
}
```

---

### 知识库检索

#### POST `/knowledge_base/search`
检索知识库

**请求体**：
```json
{
  "query": "沈俊华的案件信息",
  "top_k": 3,
  "with_score": true,
  "score_threshold": 0.5
}
```

**参数说明**：
- `query` (string, 必填): 查询文本
- `top_k` (integer, 1-20): 返回结果数量，默认 3
- `with_score` (boolean): 是否返回相似度分数，默认 true
- `score_threshold` (float, 0-1): 相似度阈值，默认 0.5

**响应示例**：
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
        "page": 1,
        "chunk_id": "chunk_001"
      },
      "similarity": 0.92,
      "distance": 0.087
    },
    {
      "content": "案号：(2023)浙01民初1234号...",
      "metadata": {
        "source": "/path/to/document.pdf",
        "page": 2,
        "chunk_id": "chunk_002"
      },
      "similarity": 0.85,
      "distance": 0.176
    }
  ]
}
```

---

### 知识库管理

#### DELETE `/knowledge_base/clear`
清空知识库

**响应示例**：
```json
{
  "success": true,
  "message": "知识库已清空"
}
```

---

## 使用示例

### Python 客户端示例

```python
import requests

# 基础配置
BASE_URL = "http://127.0.0.1:8004"

# 1. 健康检查
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. 从默认目录构建知识库（同步）
response = requests.post(
    f"{BASE_URL}/knowledge_base/build/directory",
    json={
        "directory": None,  # 使用默认目录
        "recursive": True,
        "clear_existing": False
    }
)
print(response.json())

# 3. 从指定文件构建（异步）
response = requests.post(
    f"{BASE_URL}/knowledge_base/build/files",
    json={
        "file_paths": ["/path/to/file1.pdf", "/path/to/file2.pdf"],
        "clear_existing": False
    },
    params={"async_mode": True}
)
result = response.json()
task_id = result.get("task_id")

# 4. 监控任务状态
import time
while True:
    response = requests.get(f"{BASE_URL}/knowledge_base/tasks/{task_id}")
    status = response.json()
    print(f"状态: {status['status']} - {status['message']}")
    
    if status["status"] in ["completed", "failed"]:
        break
    
    time.sleep(2)

# 5. 检索知识库
response = requests.post(
    f"{BASE_URL}/knowledge_base/search",
    json={
        "query": "沈俊华的案件信息",
        "top_k": 3,
        "with_score": True
    }
)
print(response.json())

# 6. 获取知识库信息
response = requests.get(f"{BASE_URL}/knowledge_base/info")
print(response.json())
```

---

### JavaScript/Fetch 示例

```javascript
const BASE_URL = "http://127.0.0.1:8004";

// 1. 健康检查
fetch(`${BASE_URL}/health`)
  .then(response => response.json())
  .then(data => console.log(data));

// 2. 构建知识库
fetch(`${BASE_URL}/knowledge_base/build/directory`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    directory: null,
    recursive: true,
    clear_existing: false
  })
})
  .then(response => response.json())
  .then(data => console.log(data));

// 3. 检索知识库
fetch(`${BASE_URL}/knowledge_base/search`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: "沈俊华的案件信息",
    top_k: 3
  })
})
  .then(response => response.json())
  .then(data => console.log(data));

// 4. 上传文件
const formData = new FormData();
formData.append('files', fileInput.files[0]);
formData.append('files', fileInput.files[1]);

fetch(`${BASE_URL}/knowledge_base/build/upload?async_mode=true`, {
  method: 'POST',
  body: formData
})
  .then(response => response.json())
  .then(data => console.log(data));
```

---

### curl 示例

```bash
# 1. 健康检查
curl http://127.0.0.1:8004/health

# 2. 从默认目录构建
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory": null,
    "recursive": true,
    "clear_existing": false
  }'

# 3. 从文件列表构建（异步）
curl -X POST "http://127.0.0.1:8004/knowledge_base/build/files?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/path/to/file1.pdf", "/path/to/file2.pdf"],
    "clear_existing": false
  }'

# 4. 上传文件
curl -X POST http://127.0.0.1:8004/knowledge_base/build/upload \
  -F "files=@/path/to/file1.pdf" \
  -F "files=@/path/to/file2.pdf"

# 5. 查询任务状态
curl http://127.0.0.1:8004/knowledge_base/tasks/task_20231202_143022_123456

# 6. 检索知识库
curl -X POST http://127.0.0.1:8004/knowledge_base/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "沈俊华的案件信息",
    "top_k": 3
  }'

# 7. 获取知识库信息
curl http://127.0.0.1:8004/knowledge_base/info

# 8. 清空知识库
curl -X DELETE http://127.0.0.1:8004/knowledge_base/clear
```

---

## 客户端工具

### 交互式测试客户端

项目提供了一个交互式测试客户端 `test_kb_api.py`：

```bash
python3 test_kb_api.py
```

**功能**：
- 健康检查
- 获取知识库信息和统计
- 从目录/文件构建知识库
- 上传文件构建
- 检索知识库
- 查看任务状态
- 清空知识库

**使用方法**：
1. 启动客户端
2. 根据菜单选择操作
3. 按提示输入参数
4. 查看结果

---

## 最佳实践

### 1. 构建策略

**首次构建**：
```bash
# 清空已有数据，从默认目录全新构建
curl -X POST http://127.0.0.1:8004/knowledge_base/build/directory \
  -H "Content-Type: application/json" \
  -d '{"clear_existing": true}'
```

**增量更新**：
```bash
# 不清空，仅添加新文档
curl -X POST http://127.0.0.1:8004/knowledge_base/build/files \
  -H "Content-Type: application/json" \
  -d '{
    "file_paths": ["/path/to/new_file.pdf"],
    "clear_existing": false
  }'
```

**大批量构建**：
```bash
# 使用异步模式，避免请求超时
curl -X POST "http://127.0.0.1:8004/knowledge_base/build/directory?async_mode=true" \
  -H "Content-Type: application/json" \
  -d '{"directory": "/large/document/dir"}'
```

### 2. 文档组织

**推荐目录结构**：
```
data/knowledge_base/
├── 民事案件/
│   ├── 2023/
│   │   ├── 案件001.pdf
│   │   └── 案件002.pdf
│   └── 2024/
├── 刑事案件/
└── 行政案件/
```

**文件命名规范**：
- 使用有意义的文件名
- 包含关键信息（案号、当事人、日期等）
- 避免使用特殊字符

### 3. 检索优化

**使用合适的 top_k**：
- 简单查询：top_k = 3
- 复杂查询：top_k = 5-10
- 详细分析：top_k = 10-20

**调整相似度阈值**：
- 高精度：score_threshold = 0.7-0.9
- 平衡：score_threshold = 0.5-0.7
- 高召回：score_threshold = 0.3-0.5

**优化查询语句**：
```python
# 不好的查询
query = "沈"

# 好的查询
query = "沈俊华的案件信息"

# 更好的查询
query = "沈俊华 民事诉讼 杭州中院"
```

### 4. 性能优化

**批量处理**：
```python
# 不推荐：逐个文件调用
for file in files:
    requests.post(f"{BASE_URL}/knowledge_base/build/files", 
                  json={"file_paths": [file]})

# 推荐：批量处理
requests.post(f"{BASE_URL}/knowledge_base/build/files", 
              json={"file_paths": files})
```

**异步处理大任务**：
```python
# 处理大量文档时使用异步模式
response = requests.post(
    f"{BASE_URL}/knowledge_base/build/directory",
    json={"directory": "/large/dir"},
    params={"async_mode": True}
)
```

### 5. 错误处理

```python
import requests

def build_knowledge_base(directory):
    try:
        response = requests.post(
            f"{BASE_URL}/knowledge_base/build/directory",
            json={"directory": directory},
            timeout=300  # 设置超时
        )
        response.raise_for_status()  # 检查 HTTP 状态
        
        result = response.json()
        if result.get("success"):
            print(f"构建成功: {result['chunks_added']} 个文档块")
        else:
            print(f"构建失败: {result['message']}")
            
    except requests.exceptions.Timeout:
        print("请求超时，建议使用异步模式")
    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
    except Exception as e:
        print(f"未知错误: {e}")
```

---

## 常见问题

### 1. 服务启动失败

**问题**：`ImportError: No module named 'fastapi'`

**解决**：
```bash
pip install fastapi uvicorn python-multipart
```

### 2. 知识库为空

**问题**：构建成功但文档数为 0

**原因**：
- 文档目录为空或路径错误
- 文档格式不支持
- OCR 处理失败

**解决**：
1. 检查目录路径和文件
2. 查看服务日志
3. 确认 Qwen2.5-VL 服务可用

### 3. 检索无结果

**问题**：检索返回空结果

**原因**：
- 知识库为空
- 查询语句不匹配
- 相似度阈值过高

**解决**：
1. 确认知识库有数据：`GET /knowledge_base/info`
2. 降低相似度阈值：`score_threshold: 0.3`
3. 优化查询语句

### 4. 上传文件失败

**问题**：文件上传返回错误

**原因**：
- 文件过大
- 格式不支持
- 磁盘空间不足

**解决**：
1. 检查文件大小（建议 < 50MB）
2. 确认文件格式支持
3. 检查磁盘空间

---

## 技术支持

如有问题或建议，请查看：
- API 文档：http://127.0.0.1:8004/docs
- 项目 README：./README.md
- RAG 使用指南：./RAG_USAGE.md

---

**最后更新**: 2023-12-02

