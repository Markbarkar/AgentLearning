"""
Qwen2.5-VL Agent API 服务

提供 HTTP API 接口，接收用户任务并返回 Agent 处理结果
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import shutil
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

# 从各个模块导入需要的组件
from agent_system.config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_SEED,
    KNOWLEDGE_BASE_DOCS_DIR
)
from agent_system.tools import Qwen25VLTools, finish_tool, create_rag_search_tool
from agent_system.core import Agent
from agent_system.rag import KnowledgeBase
from fastapi.middleware.cors import CORSMiddleware


# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen2.5-VL Agent API",
    description="法律文书智能处理 Agent API",
    version="1.0.0"
)

# 配置CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://192.168.2.106:6789",
#         "http://localhost:6789",
#         "http://www.zktmai.com:9093",
#         "http://www.zktmai.com:8001",
#         "http://www.zktmai.com:6789",
#         "http://120.237.13.172:9093",
#         "http://120.237.13.172:8001",
#         "http://120.237.13.172:6789",
#     ],
#     allow_credentials=True,  # 允许凭证
#     allow_methods=["*"],  # 允许所有方法
#     allow_headers=["*"],  # 允许所有头
#     expose_headers=["Content-Length", "Content-Range"],
#     max_age=3600,  # 预检请求缓存时间
# )

# 请求模型
class TaskRequest(BaseModel):
    """任务请求模型"""
    task: str = Field(..., description="用户任务描述", example="请提取 /Users/linzaizai/Desktop/Agent/doc/附件九.pdf 的关键信息")
    file_path: Optional[str] = Field(None, description="文件路径（可选）", example="/Users/linzaizai/Desktop/Agent/doc/附件九.pdf")
    temperature: Optional[float] = Field(None, description="LLM 温度参数（可选，默认使用配置值）", ge=0.0, le=1.0)
    use_rag: Optional[bool] = Field(True, description="是否使用 RAG 增强（默认启用）")
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="user_123")


class KnowledgeBaseBuildRequest(BaseModel):
    """知识库构建请求模型"""
    directory: Optional[str] = Field(None, description="文档目录路径")
    file_paths: Optional[list] = Field(None, description="文件路径列表")
    clear_existing: bool = Field(False, description="是否清空已有数据")
    recursive: bool = Field(True, description="是否递归处理子目录")
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="user_123")


class KnowledgeBaseSearchRequest(BaseModel):
    """知识库检索请求模型"""
    query: str = Field(..., description="查询文本")
    top_k: int = Field(3, description="返回的文档数量", ge=1, le=10)
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="user_123")


class KnowledgeBaseInfoRequest(BaseModel):
    """知识库信息查询请求模型"""
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="user_123")


class KnowledgeBaseClearRequest(BaseModel):
    """知识库清空请求模型"""
    user_id: Optional[str] = Field(None, description="用户ID（用于知识库隔离）", example="user_123")


# 响应模型
class TaskResponse(BaseModel):
    """任务响应模型"""
    success: bool = Field(..., description="任务是否成功")
    result: str = Field(..., description="Agent 处理结果")
    error: Optional[str] = Field(None, description="错误信息（如果有）")


# 全局变量：缓存实例
_agent_instances = {}  # {user_id: Agent实例}
_vl_tools_instance = None
_knowledge_base_instances = {}  # {user_id: KnowledgeBase实例}，支持多用户隔离


def get_vl_tools():
    """获取 Qwen2.5-VL 工具集实例（单例模式）"""
    global _vl_tools_instance
    if _vl_tools_instance is None:
        _vl_tools_instance = Qwen25VLTools()
    return _vl_tools_instance


def get_knowledge_base(user_id: Optional[str] = None):
    """
    获取知识库实例（多用户模式）
    
    Args:
        user_id: 用户ID，用于实现多用户知识库隔离
        
    Returns:
        对应用户的知识库实例
    """
    global _knowledge_base_instances
    
    # 使用 user_id 或 "public" 作为缓存key
    cache_key = user_id or "public"
    
    if cache_key not in _knowledge_base_instances:
        try:
            vl_tools = get_vl_tools()
            _knowledge_base_instances[cache_key] = KnowledgeBase(
                user_id=user_id,
                vl_tools=vl_tools
            )
            user_label = f"用户 {user_id}" if user_id else "公共"
            print(f"✓ {user_label}知识库初始化成功")
        except Exception as e:
            print(f"✗ 知识库初始化失败: {str(e)}")
            return None
    
    return _knowledge_base_instances[cache_key]


def annotate_pdf_wrapper(vl_tools, input_str):
    """标注 PDF 的包装函数"""
    try:
        parts = input_str.split(',', 1)
        if len(parts) != 2:
            return "错误：输入格式应为 'file_path,key_fields'，例如：'/path/to/file.pdf,案号;法院;当事人'"
        
        file_path = parts[0].strip()
        key_fields = parts[1].strip().replace(';', ',')
        
        result = vl_tools.annotate_legal_pdf(
            file_path=file_path,
            key_fields=key_fields,
            temperature=0.1
        )
        
        return str(result)
    except Exception as e:
        return f"错误：{str(e)}"


def recognize_form_wrapper(vl_tools, input_str):
    """识别表单的包装函数"""
    try:
        parts = input_str.split(',')
        file_path = parts[0].strip()
        table_type = parts[1].strip() if len(parts) > 1 else "custom"
        
        result = vl_tools.recognize_form(
            file_path=file_path,
            table_type=table_type
        )
        
        return str(result)
    except Exception as e:
        return f"错误：{str(e)}"


def create_qwen_vl_tools(user_id: Optional[str] = None):
    """
    创建 Qwen2.5-VL 相关的 LangChain 工具
    
    Args:
        user_id: 用户ID，用于创建用户专属的RAG工具
    """
    vl_tools = get_vl_tools()
    
    # 1. 提取关键信息工具
    extract_key_info_tool = Tool(
        name="extract_legal_key_info",
        description=
        """提取法律文书的关键信息。
        
            输入参数：
            - file_path: 文件路径（必填）

            返回信息包括：
            - title: 文件标题
            - case_number: 案号
            - case_reason: 案由
            - court: 法院
            - hearing_time: 开庭时间
            - hearing_place: 开庭地点
            - case_type: 案件类型
            - lead_lawyer: 主办律师
            - client_name: 委托人
            - parties: 当事人

            使用场景：快速了解法律文书的核心信息
            示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf""",
        func=lambda file_path: str(vl_tools.extract_key_info(file_path=file_path))
    )
    
    # 2. 提取文档全文工具
    extract_text_tool = Tool(
        name="extract_document_text",
        description=
        """提取文档的全部文字内容（OCR）。

            输入参数：
            - file_path: 文件路径（必填）

            返回：文档每一页的文字内容列表

            使用场景：需要获取文档完整文本内容时使用
            示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf""",
        func=lambda file_path: str(vl_tools.extract_text(file_path=file_path))
    )
    
    # 3. 标注法律文书工具（PDF）
    annotate_pdf_tool = Tool(
        name="annotate_legal_pdf",
        description="""标注 PDF 法律文书，提取指定字段及其在文档中的位置。

        输入参数格式（用逗号分隔）：
        file_path,key_fields

        其中：
        - file_path: 文件路径
        - key_fields: 要提取的字段，用分号分隔，如 "案号;主办律师;法院"

        支持的字段：
        - 案号、案件编号
        - 主办律师、承办律师
        - 协办律师
        - 文件类型、案件类型
        - 当事人、原告、被告
        - 委托人
        - 法院
        - 案由
        - 开庭时间、开庭地点
        - 落款日期、落款地点、落款人

        返回：每个字段的内容和位置信息（边界框坐标）

        使用场景：需要知道信息在文档中的具体位置时使用
        示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf,案号;法院;当事人""",
        func=lambda input_str: annotate_pdf_wrapper(vl_tools, input_str)
    )
    
    # 4. 识别表单工具
    recognize_form_tool = Tool(
        name="recognize_form",
        description="""识别法律案件表单/表格信息。

        输入参数格式（用逗号分隔）：
        file_path,table_type

        其中：
        - file_path: 文件路径
        - table_type: 表单类型，可选值：
        * main: 案件主信息表
        * task: 任务与期限明细表
        * fee: 案件收费明细表
        * asset: 财产保全明细表
        * custom: 自定义表单（默认）

        返回：结构化的表单数据

        使用场景：识别结构化的表格信息
        示例输入：/Users/linzaizai/Desktop/Agent/doc/附件九.pdf,custom""",
        func=lambda input_str: recognize_form_wrapper(vl_tools, input_str)
    )
    
    # 5. RAG 检索工具（如果知识库可用）
    tools = [
        extract_key_info_tool,
        extract_text_tool,
        annotate_pdf_tool,
        recognize_form_tool,
    ]
    
    # 尝试添加 RAG 工具（使用用户专属知识库）
    try:
        kb = get_knowledge_base(user_id)
        if kb and kb.vector_store.get_collection_count() > 0:
            rag_tool = create_rag_search_tool(kb)
            tools.append(rag_tool)
            user_label = f"用户 {user_id}" if user_id else "公共"
            print(f"✓ {user_label} RAG 工具已添加")
    except Exception as e:
        print(f"RAG 工具初始化失败: {str(e)}")
    
    tools.append(finish_tool)
    return tools


def get_agent(user_id: Optional[str] = None, temperature: Optional[float] = None, use_rag: bool = True):
    """
    获取 Agent 实例（多用户模式）
    
    Args:
        user_id: 用户ID，用于实现多用户知识库隔离
        temperature: LLM 温度参数，如果为 None 则使用默认配置
        use_rag: 是否启用 RAG 增强
    
    Returns:
        对应用户的 Agent 实例
    """
    global _agent_instances
    
    # 使用 user_id 或 "public" 作为缓存key
    cache_key = user_id or "public"
    
    # 如果指定了温度参数，或者还没有创建该用户的实例，则创建新实例
    if temperature is not None or cache_key not in _agent_instances:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            model_kwargs={"seed": LLM_SEED}
        )
        
        # 创建用户专属工具（包括用户专属RAG工具）
        tools = create_qwen_vl_tools(user_id)
        
        # 获取用户专属知识库（如果启用 RAG）
        knowledge_base = None
        if use_rag:
            try:
                knowledge_base = get_knowledge_base(user_id)
            except Exception as e:
                print(f"知识库加载失败: {str(e)}")
        
        _agent_instances[cache_key] = Agent(
            llm=llm,
            tools=tools,
            knowledge_base=knowledge_base,
            use_rag=use_rag
        )
    
    return _agent_instances[cache_key]


@app.get("/agent/tools")
async def get_tools_info():
    """获取可用工具列表"""
    return {
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in create_qwen_vl_tools()
        ]
    }


@app.post("/agent/knowledge_base/build")
async def build_knowledge_base(
    files: Optional[List[UploadFile]] = File(None),
    user_id: str = Form(...),
    directory: Optional[str] = Form(None),
    file_paths: Optional[str] = Form(None),  # JSON字符串
    clear_existing: bool = Form(False),
    recursive: bool = Form(True)
):
    """
    构建/更新知识库
    
    支持两种模式:
    1. 文件上传模式(推荐): 通过 multipart/form-data 上传文件
    2. 路径模式(管理员): 通过表单参数指定 directory 或 file_paths
    
    Args:
        files: 上传的文件列表(可选)
        user_id: 用户ID(可选,用于知识库隔离)
        directory: 文档目录路径(可选,管理员模式)
        file_paths: 文件路径列表JSON字符串(可选,管理员模式)
        clear_existing: 是否清空已有数据
        recursive: 是否递归处理子目录
    
    Returns:
        构建结果
    """
    temp_dir = None
    try:
        # 获取用户专属知识库实例
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库初始化失败",
                    "user_id": user_id or "public"
                }
            )
        
        # 优先处理文件上传
        if files and len(files) > 0:
            # 创建临时目录: temp_uploads/{user_id or 'public'}/{timestamp}/
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_folder = user_id or "public"
            temp_dir = os.path.join(KNOWLEDGE_BASE_DOCS_DIR, "temp_uploads", user_folder, timestamp)
            os.makedirs(temp_dir, exist_ok=True)
            
            # 保存所有上传的文件
            saved_files = []
            for file in files:
                if file.filename:
                    file_path = os.path.join(temp_dir, file.filename)
                    with open(file_path, "wb") as f:
                        content = await file.read()
                        f.write(content)
                    saved_files.append(file.filename)
            
            # 使用临时目录构建知识库
            result = kb.build_from_directory(
                directory=temp_dir,
                recursive=recursive,
                clear_existing=clear_existing
            )
            result["uploaded_files"] = saved_files
            result["file_count"] = len(saved_files)
        
        # 降级处理: 从目录构建
        elif directory:
            result = kb.build_from_directory(
                directory=directory,
                recursive=recursive,
                clear_existing=clear_existing
            )
        
        # 降级处理: 从文件列表构建
        elif file_paths:
            import json
            try:
                paths_list = json.loads(file_paths)
            except:
                paths_list = [file_paths]  # 单个路径
            
            result = kb.build_from_files(
                file_paths=paths_list,
                clear_existing=clear_existing
            )
        
        else:
            # 使用默认目录
            result = kb.build_from_directory(
                directory=KNOWLEDGE_BASE_DOCS_DIR,
                recursive=recursive,
                clear_existing=clear_existing
            )
        
        # 在响应中包含用户ID信息
        result["user_id"] = user_id or "public"
        return JSONResponse(content=result)
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"构建失败: {str(e)}",
                "user_id": user_id or "public"
            }
        )
    
    finally:
        # 清理临时文件和目录
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                # 同时清理空的父目录
                parent_dir = os.path.dirname(temp_dir)
                if os.path.exists(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            except Exception as e:
                print(f"清理临时目录失败: {str(e)}")


@app.post("/agent/knowledge_base/info")
async def get_knowledge_base_info(request: KnowledgeBaseInfoRequest):
    """
    获取知识库信息
    
    Args:
        request: 知识库信息查询请求
    
    Returns:
        知识库统计信息
    """
    try:
        user_id = request.user_id
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库未初始化",
                    "user_id": user_id or "public"
                }
            )
        
        info = kb.get_info()
        return JSONResponse(content={
            "success": True,
            **info
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"获取信息失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@app.post("/agent/knowledge_base/search")
async def search_knowledge_base(request: KnowledgeBaseSearchRequest):
    """
    检索知识库
    
    Args:
        request: 检索请求
    
    Returns:
        检索结果
    """
    try:
        # 获取用户专属知识库实例
        user_id = request.user_id
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库未初始化",
                    "user_id": user_id or "public"
                }
            )
        
        results = kb.search(
            query=request.query,
            top_k=request.top_k,
            with_score=True
        )
        
        return JSONResponse(content={
            "success": True,
            "user_id": user_id or "public",
            "query": request.query,
            "results": results
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"检索失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@app.post("/agent/knowledge_base/clear")
async def clear_knowledge_base(request: KnowledgeBaseClearRequest):
    """
    清空知识库
    
    Args:
        request: 知识库清空请求
    
    Returns:
        操作结果
    """
    try:
        user_id = request.user_id
        kb = get_knowledge_base(user_id)
        if not kb:
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "message": "知识库未初始化",
                    "user_id": user_id or "public"
                }
            )
        
        kb.clear()
        
        return JSONResponse(content={
            "success": True,
            "message": "知识库已清空",
            "user_id": user_id or "public"
        })
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"清空失败: {str(e)}",
                "user_id": request.user_id or "public"
            }
        )


@app.post("/agent/process_task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    """
    处理用户任务
    
    Args:
        request: 任务请求，包含任务描述和可选的文件路径
    
    Returns:
        任务处理结果
    """
    try:
        # 如果提供了文件路径，将其添加到任务描述中
        task = request.task
        if request.file_path and request.file_path not in task:
            task = f"{task}\n文件路径: {request.file_path}"
        
        # 获取用户专属 Agent 实例
        agent = get_agent(
            user_id=request.user_id,
            temperature=request.temperature,
            use_rag=request.use_rag
        )
        
        # 执行任务
        result = agent.run(task)
        
        return TaskResponse(
            success=True,
            result=result,
            error=None
        )
    
    except Exception as e:
        # 返回错误信息
        return TaskResponse(
            success=False,
            result="",
            error=str(e)
        )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 启动 Qwen2.5-VL Agent API 服务")
    print("=" * 60)
    print(f"\n📝 LLM 模型: {LLM_MODEL}")
    print(f"🔧 Qwen2.5-VL API: {Qwen25VLTools().base_url}")
    print(f"\n🌐 API 服务将在以下地址启动:")
    print(f"   http://127.0.0.1:8003")
    print(f"   http://0.0.0.0:8003")
    print(f"\n📚 API 文档:")
    print(f"   http://127.0.0.1:8003/docs")
    print(f"   http://127.0.0.1:8003/redoc")
    print("\n" + "=" * 60)
    
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info"
    )


