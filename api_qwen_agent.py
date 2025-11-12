"""
Qwen2.5-VL Agent API 服务

提供 HTTP API 接口，接收用户任务并返回 Agent 处理结果
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

# 从各个模块导入需要的组件
from agent_system.config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_SEED
)
from agent_system.tools import Qwen25VLTools, finish_tool
from agent_system.core import Agent


# 创建 FastAPI 应用
app = FastAPI(
    title="Qwen2.5-VL Agent API",
    description="法律文书智能处理 Agent API",
    version="1.0.0"
)


# 请求模型
class TaskRequest(BaseModel):
    """任务请求模型"""
    task: str = Field(..., description="用户任务描述", example="请提取 /Users/linzaizai/Desktop/Agent/doc/附件九.pdf 的关键信息")
    file_path: Optional[str] = Field(None, description="文件路径（可选）", example="/Users/linzaizai/Desktop/Agent/doc/附件九.pdf")
    temperature: Optional[float] = Field(None, description="LLM 温度参数（可选，默认使用配置值）", ge=0.0, le=1.0)


# 响应模型
class TaskResponse(BaseModel):
    """任务响应模型"""
    success: bool = Field(..., description="任务是否成功")
    result: str = Field(..., description="Agent 处理结果")
    error: Optional[str] = Field(None, description="错误信息（如果有）")


# 全局变量：缓存 Agent 实例
_agent_instance = None
_vl_tools_instance = None


def get_vl_tools():
    """获取 Qwen2.5-VL 工具集实例（单例模式）"""
    global _vl_tools_instance
    if _vl_tools_instance is None:
        _vl_tools_instance = Qwen25VLTools()
    return _vl_tools_instance


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


def create_qwen_vl_tools():
    """创建 Qwen2.5-VL 相关的 LangChain 工具"""
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
    
    return [
        extract_key_info_tool,
        extract_text_tool,
        annotate_pdf_tool,
        recognize_form_tool,
        finish_tool
    ]


def get_agent(temperature: Optional[float] = None):
    """
    获取 Agent 实例（单例模式）
    
    Args:
        temperature: LLM 温度参数，如果为 None 则使用默认配置
    
    Returns:
        Agent 实例
    """
    global _agent_instance
    
    # 如果指定了温度参数，或者还没有创建实例，则创建新实例
    if temperature is not None or _agent_instance is None:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=temperature if temperature is not None else LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
            model_kwargs={"seed": LLM_SEED}
        )
        
        tools = create_qwen_vl_tools()
        _agent_instance = Agent(llm=llm, tools=tools)
    
    return _agent_instance


@app.get("/")
async def root():
    """根路径，返回 API 信息"""
    return {
        "name": "Qwen2.5-VL Agent API",
        "version": "1.0.0",
        "description": "法律文书智能处理 Agent API",
        "endpoints": {
            "POST /process_task": "处理用户任务",
            "GET /health": "健康检查",
            "GET /tools": "获取可用工具列表"
        }
    }



@app.get("/tools")
async def get_tools_info():
    """获取可用工具列表"""
    return {
        "tools": [
            {
                "name": "extract_legal_key_info",
                "description": "提取法律文书的关键信息",
                "input": "文件路径",
                "output": "关键信息字典（案号、法院、当事人等）"
            },
            {
                "name": "extract_document_text",
                "description": "提取文档的全部文字内容（OCR）",
                "input": "文件路径",
                "output": "文档每一页的文字内容列表"
            },
            {
                "name": "annotate_legal_pdf",
                "description": "标注 PDF 法律文书，提取指定字段及其位置",
                "input": "文件路径,字段列表（用分号分隔）",
                "output": "字段内容和位置信息"
            },
            {
                "name": "recognize_form",
                "description": "识别法律案件表单/表格信息",
                "input": "文件路径,表单类型",
                "output": "结构化的表单数据"
            }
        ]
    }


@app.post("/process_task", response_model=TaskResponse)
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
        
        # 获取 Agent 实例
        agent = get_agent(temperature=request.temperature)
        
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


