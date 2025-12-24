"""
API 共享依赖

包含全局变量、缓存实例和工具创建函数
"""

from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

from ..config.settings import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_BASE_URL,
    LLM_SEED,
)
from ..tools import Qwen25VLTools, finish_tool, create_rag_search_tool
from ..core import Agent
from ..rag import KnowledgeBase


# ==================== 全局变量：缓存实例 ====================

_agent_instances = {}  # {user_id: Agent实例}
_vl_tools_instance = None
_knowledge_base_instances = {}  # {user_id: KnowledgeBase实例}，支持多用户隔离


# ==================== 工具实例获取 ====================

def get_vl_tools():
    """获取 Qwen2.5-VL 工具集实例（单例模式）"""
    global _vl_tools_instance
    if _vl_tools_instance is None:
        _vl_tools_instance = Qwen25VLTools()
    return _vl_tools_instance


# ==================== 知识库实例获取 ====================

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


# ==================== 工具包装函数 ====================

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


# ==================== 工具创建函数 ====================

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
            示例输入：/path/to/附件九.pdf""",
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
            示例输入：/path/to/附件九.pdf""",
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
        示例输入：/path/to/附件九.pdf,案号;法院;当事人""",
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
        示例输入：/path/to/附件九.pdf,custom""",
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


# ==================== Agent 实例获取 ====================

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

