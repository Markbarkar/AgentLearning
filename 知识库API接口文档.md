# 知识库 API 接口文档

## 基础信息

- **Base URL**: `http://www.zktmai.com:8001`
- **Content-Type**: 
  - 构建知识库接口: `multipart/form-data` (文件上传) 或 `application/json` (路径模式)
  - 其他接口: `application/json`
- **所有接口都使用 POST 方法**

---

## 1. 构建知识库

**接口**: `POST /agent/knowledge_base/build`

### 方式一: 文件上传模式（推荐）

**Content-Type**: `multipart/form-data`

- `files`: 文件列表 (必填，支持多个文件)
- `user_id`: 用户ID (必填，公共知识库仅限管理员操作)
- `clear_existing`: 是否清空已有数据 (可选，默认false)
- `recursive`: 是否递归处理子目录 (可选，默认true)

**响应示例**:
```json
{
  "success": true,
  "message": "知识库构建成功",
  "user_id": "user_123",
  "chunks_added": 25,              // 新增文档块数量
  "total_count": 50,               // 总文档块数量
  "uploaded_files": ["doc1.pdf", "doc2.pdf"],  // 上传的文件列表
  "file_count": 2                  // 上传的文件数量
}
```

### 方式二: 路径模式（管理员）

**Content-Type**: `multipart/form-data` 或 `application/json`

**请求参数**:
- `user_id`: 用户ID (可选，不传则使用公共库)
- `directory`: 文档目录路径 (可选，指定服务器上的目录)
- `file_paths`: 文件路径列表JSON字符串 (可选，指定服务器上的文件)
- `clear_existing`: 是否清空已有数据 (可选，默认false)
- `recursive`: 是否递归处理子目录 (可选，默认true)

**响应示例**:
```json
{
  "success": true,
  "message": "知识库构建成功",
  "user_id": "user_123",
  "chunks_added": 25,              // 新增文档块数量
  "total_count": 50                // 总文档块数量
}
```

- **说明**:
- **文件上传模式**: 用户直接上传文件，临时存储后自动构建并清理
- **路径模式**: 适用于管理员指定服务器上已存在的文件或目录
- `user_id` 为必填字段且必须是有效用户ID，以避免误操作公共知识库
- 上传的文件在构建完成后会自动删除，仅保留向量数据

---

## 2. 获取知识库信息

**接口**: `POST /agent/knowledge_base/info`

**请求参数**:
```json
{
  "user_id": "user_123"  // 可选，用户ID（不传则查询公共库）
}
```

**响应示例**:
```json
{
  "success": true,
  "user_id": "user_123",
  "collection_name": "user_user_123_documents",
  "persist_directory": "./data/chroma_db",
  "total_documents": 50,           // 文档块数量（注意：不是文件数）
  "embedding_model": "text-embedding-v3"
}
```

**注意**: 
- `total_documents` 是文档块（chunks）数量，不是文件数量
- 1个PDF文件可能被分割成多个chunks

---

## 3. 检索知识库

**接口**: `POST /agent/knowledge_base/search`

**请求参数**:
```json
{
  "user_id": "user_123",    // 可选，用户ID
  "query": "查询内容",      // 必填，查询文本
  "top_k": 3                // 可选，返回结果数量（1-10）
}
```

**响应示例**:
```json
{
  "success": true,
  "user_id": "user_123",
  "query": "查询内容",
  "results": [
    {
      "content": "文档内容片段...",
      "metadata": {
        "source": "/path/to/file.pdf",
        "page": 1,
        "chunk_id": "chunk_001"
      },
      "similarity": 0.92,    // 相似度（0-1）
      "distance": 0.087      // 距离值
    }
  ]
}
```

---

## 4. 清空知识库

**接口**: `POST /agent/knowledge_base/clear`

**请求参数**:
```json
{
  "user_id": "user_123"  // 可选，用户ID
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "知识库已清空",
  "user_id": "user_123"
}
```

**警告**: 此操作不可恢复！

---

## 5. Agent 任务处理（支持RAG）

**接口**: `POST /agent/process_task`

**请求参数**:
```json
{
  "task": "请分析这个案件",          // 必填，任务描述
  "user_id": "user_123",            // 可选，使用该用户的知识库
  "file_path": "/path/to/file.pdf", // 可选，文件路径
  "temperature": 0.0,               // 可选，LLM温度（0-1）
  "use_rag": true                   // 可选，是否使用RAG
}
```

**响应示例**:
```json
{
  "success": true,
  "result": "Agent处理结果...",
  "error": null
}
```

---

## 用户隔离说明

### user_id 规则

- **不传 user_id**: 使用公共知识库 `public_knowledge_base`
- **传 user_id**: 使用用户专属知识库 `user_{user_id}_documents`
- **字符限制**: 只保留字母、数字、下划线，最长32字符

### 示例

```javascript
// 公共知识库
fetch('/agent/knowledge_base/info', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({})  // 不传user_id
})

// 用户专属知识库
fetch('/agent/knowledge_base/info', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    user_id: 'user_123'
  })
})
```

---

## 快速测试

### curl 示例

```bash
# 1. 构建知识库 - 文件上传模式（推荐）
curl -X POST http://www.zktmai.com:8001/agent/knowledge_base/build \
  -F "files=@/path/to/document1.pdf" \
  -F "files=@/path/to/document2.pdf" \
  -F "user_id=test_user" \
  -F "clear_existing=false"

# 1b. 构建知识库 - 路径模式（管理员）
curl -X POST http://www.zktmai.com:8001/agent/knowledge_base/build \
  -F "user_id=test_user" \
  -F "directory=/path/to/docs" \
  -F "recursive=true"

# 2. 查看知识库信息
curl -X POST http://www.zktmai.com:8001/agent/knowledge_base/info \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}'

# 3. 检索知识库
curl -X POST http://www.zktmai.com:8001/agent/knowledge_base/search \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "query": "沈俊华",
    "top_k": 3
  }'

# 4. 清空知识库
curl -X POST http://www.zktmai.com:8001/agent/knowledge_base/clear \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user"}'
```

### JavaScript 示例

```javascript
const BASE_URL = 'http://www.zktmai.com:8001';

// 构建知识库 - 文件上传模式（推荐）
async function buildKBWithFiles(userId, files) {
  const formData = new FormData();
  
  // 添加文件
  for (let file of files) {
    formData.append('files', file);
  }
  
  // 添加其他参数
  if (userId) formData.append('user_id', userId);
  formData.append('clear_existing', 'false');
  formData.append('recursive', 'true');
  
  const response = await fetch(`${BASE_URL}/agent/knowledge_base/build`, {
    method: 'POST',
    body: formData
    // 注意: 不要设置 Content-Type header，浏览器会自动设置
  });
  return await response.json();
}

// 使用示例 - 从文件输入获取文件
// HTML: <input type="file" id="fileInput" multiple>
document.getElementById('fileInput').addEventListener('change', async (e) => {
  const files = e.target.files;
  const result = await buildKBWithFiles('user_123', files);
  console.log('构建结果:', result);
  console.log(`上传了 ${result.file_count} 个文件`);
  console.log(`新增 ${result.chunks_added} 个文档块`);
});

// 构建知识库 - 路径模式（管理员）
async function buildKBWithPath(userId, directory) {
  const formData = new FormData();
  if (userId) formData.append('user_id', userId);
  if (directory) formData.append('directory', directory);
  formData.append('clear_existing', 'false');
  formData.append('recursive', 'true');
  
  const response = await fetch(`${BASE_URL}/agent/knowledge_base/build`, {
    method: 'POST',
    body: formData
  });
  return await response.json();
}

// 获取信息
async function getKBInfo(userId) {
  const response = await fetch(`${BASE_URL}/agent/knowledge_base/info`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({user_id: userId})
  });
  return await response.json();
}

// 检索
async function searchKB(userId, query) {
  const response = await fetch(`${BASE_URL}/agent/knowledge_base/search`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      user_id: userId,
      query: query,
      top_k: 3
    })
  });
  return await response.json();
}

// 使用示例
const userId = 'user_123';
const info = await getKBInfo(userId);
console.log(`知识库文档块数: ${info.total_documents}`);

const results = await searchKB(userId, '沈俊华');
console.log(`找到 ${results.results.length} 个结果`);
```

---

## 错误处理

所有接口失败时返回：
```json
{
  "success": false,
  "message": "错误描述",
  "user_id": "user_123"
}
```

HTTP 状态码：
- `200`: 成功
- `500`: 服务器错误

---

## 注意事项

1. **文档块 vs 文件数**: 
   - API 返回的 `total_documents` 是文档块数量
   - 一个 PDF 文件会被分割成多个块（默认每块800字符）

2. **知识库存储**:
   - 原始文件：`./data/knowledge_base/`
   - 向量数据：`./data/chroma_db/`
   - Collection 名称是逻辑名称，数据存储在 UUID 文件夹中

3. **用户隔离**:
   - 每个 `user_id` 对应独立的 collection
   - 数据完全隔离，互不影响

4. **性能优化**:
   - 知识库实例会被缓存
   - 首次访问会较慢（需要初始化）
   - 后续访问使用缓存实例

---

## API 文档地址

启动服务后访问：
- Swagger UI: http://www.zktmai.com:8001/docs
- ReDoc: http://www.zktmai.com:8001/redoc

---

**最后更新**: 2024-12-03

